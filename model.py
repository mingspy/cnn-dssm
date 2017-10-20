#coding:utf-8
import tensorflow as tf
import numpy as np
import random
import os
from scipy.spatial.distance import cosine
from data_helpers import decode_from_tfrecords,get_batch,load_test_dssm_data 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

INIT_EMBEDDING_RANDOMLY = True

class CDssm(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__( self,  sequence_length, embedding_size, vocab_size,
            filter_sizes, num_filters,hidden_sizes, batch_size=100, activeFn=tf.nn.relu):
        # Placeholders for input, output and dropout
        self.query_in = tf.placeholder(tf.int32, [None, sequence_length], name="query_in")
        self.doc_in = tf.placeholder(tf.int32, [None, sequence_length], name="doc_in")
        self.l2_loss = tf.constant(0.0)

        # Keeping track of l2 regularization loss (optional)
        self.sequence_length = sequence_length
        self.filter_sizes = filter_sizes
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.vocab_size = vocab_size
        self.hidden_sizes = hidden_sizes
        self.activeFn = activeFn
        self.batch_size = batch_size

        self.conv_ws = []
        self.conv_bs = []

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if INIT_EMBEDDING_RANDOMLY:
                wsq = np.sqrt(6.0 / (vocab_size + embedding_size))
                self.embedding_W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -wsq, wsq),
                    name="W")
            else:
                emb = self._read_pre_emb('../../data/dssm/char_emb.txt',embedding_size,vocab_size)
                self.embedding_W = tf.Variable(emb,name="W")
            self.l2_loss += tf.nn.l2_loss(self.embedding_W)

        # Create a convolution + maxpool layer for each filter size
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                conv_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                conv_b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.l2_loss += tf.nn.l2_loss(conv_W)
                self.l2_loss += tf.nn.l2_loss(conv_b)
                self.conv_ws.append(conv_W)
                self.conv_bs.append(conv_b)

        # Final (unnormalized) scores and predictions
        num_filters_total = num_filters * len(filter_sizes)
        with tf.name_scope("output"):
            self.out_Ws = []
            self.out_bs = []
            tmp = [num_filters_total] + hidden_sizes
            for i in xrange(0, len(tmp) - 1):
                wsq = np.sqrt(6.0 / (tmp[i] + tmp[i+1]))
                out_W = tf.get_variable(
                    "W_%d"%i,
                    shape=[tmp[i], tmp[i+1] ],
                    #initializer=tf.contrib.layers.xavier_initializer())
                    initializer=tf.random_uniform_initializer(minval=-wsq,maxval=wsq))
                out_b = tf.Variable(tf.constant(0.0001, shape=[tmp[i+1]]), name="b_%d"%i)
                self.l2_loss += tf.nn.l2_loss(out_W)
                self.l2_loss += tf.nn.l2_loss(out_b)
                self.out_Ws.append(out_W)
                self.out_bs.append(out_b)

        self.query_vec_pred = self._predict(self.query_in,'query_doc')
        self.doc_vec_pred = self._predict(self.doc_in)

    def loss_op(self,query_vec,doc_vec,neg=4, l2_reg_lambda=0.05): 
        tmp = tf.tile(doc_vec,[1,1])
        doc_vecs = tf.tile(doc_vec,[1,1])
        for i in xrange(neg):
            rand = random.randint(0,self.batch_size + i) % self.batch_size
            doc_vecs = tf.concat( [doc_vecs, 
                    tf.slice(tmp,[rand,0],[self.batch_size - rand, -1]),
                    tf.slice(tmp,[0,0],[rand, -1])], 0)
        print('after NEG, shape of docvec %s '%doc_vecs.get_shape())

        with tf.name_scope('Cosine_Similarity'):
            # Cosine similarity
            query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_vec), 1, True)), [neg + 1, 1])
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_vecs), 1, True))

            prod = tf.reduce_sum(tf.multiply(tf.tile(query_vec, [neg + 1, 1]), doc_vecs), 1, True)
            norm_prod = tf.multiply(query_norm, doc_norm)

            cos_sim_raw = tf.truediv(prod, norm_prod)
            cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [neg + 1, -1])) * 20

        print('cos_sim shape:%s'%cos_sim.get_shape())
        with tf.name_scope('Loss'):
            # Train Loss
            prob = tf.nn.softmax((cos_sim))
            self.hit_prob = tf.slice(prob, [0, 0], [-1, 1])
            print('hit_prob shape:%s'%self.hit_prob.get_shape())
            self.neg_prob = tf.slice(prob, [0, 1], [-1, -1])
            print('neg_prob shape:%s'%self.neg_prob.get_shape())
            #self.loss = tf.maximum(1 - tf.reduce_mean(tf.log(self.hit_prob)) + tf.reduce_mean(tf.log(self.neg_prob)),0) + self.l2_loss * l2_reg_lambda
            #self.loss = -tf.reduce_mean(tf.log(self.hit_prob)) + self.l2_loss * l2_reg_lambda
            self.loss = -tf.reduce_mean(tf.log(self.hit_prob) + tf.log(1 - self.neg_prob)) + self.l2_loss * l2_reg_lambda
            tf.summary.scalar("loss",self.loss)

        return self.loss


    def _predict(self,X,name=None):
        embedded_chars = tf.nn.embedding_lookup(self.embedding_W, X)
        print('after lookup shape %s'%embedded_chars.get_shape())
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        print('after expanded shape %s'%embedded_chars_expanded.get_shape())

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            # Convolution Layer
            conv = tf.nn.conv2d( embedded_chars_expanded, self.conv_ws[i],
                strides=[1, 1, 1, 1], padding="VALID")

            # Apply nonlinearity
            print("\n===conv:size:%d===" % filter_size)
            print("after conv, shape is %s"%conv.get_shape())
            h = self.activeFn(tf.nn.bias_add(conv, self.conv_bs[i]))
            print("after relu, shape is %s"%h.get_shape())
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool( h,
                ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1], padding='VALID')
            print("after pooled, shape is %s"%pooled.get_shape())
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        print("after concat,conved output shape is %s"%h_pool.get_shape())
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        print("after flat, conved output shape is %s"%h_pool_flat.get_shape())

        scores_flat = h_pool_flat
        for i in xrange(len(self.out_Ws)):
            if (i == (len(self.out_Ws) - 1)) and name:
                scores_flat = self.activeFn(tf.nn.xw_plus_b(scores_flat, self.out_Ws[i], self.out_bs[i]),name=name)
            else:  scores_flat = self.activeFn(tf.nn.xw_plus_b(scores_flat, self.out_Ws[i], self.out_bs[i]))
        print("predict , shape is %s"%scores_flat.get_shape())
        return scores_flat

    def _read_pre_emb(self,path, dim,vocab_size):
        print('reading pre embedding:%s'%path)
        inf = open(path,'r')
        emb = {}
        head = inf.readline()
        v,d = head.split()[:2]
        assert int(v) == vocab_size 
        assert int(d) == dim

        for i in inf:
            its = i.split()
            assert len(its) == dim + 1
            emb[int(its[0])] = [float(j) for j in its[1:]]
        emb = sorted(emb.iteritems(),key=lambda x: x[0])
        ret = []
        for w,v in emb:
            ret.append(v)
        return ret
        
