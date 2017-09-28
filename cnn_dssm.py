import tensorflow as tf
import numpy as np
import random
import os
from data_helpers import batch_iter
from datetime import datetime
import traceback
import json
from scipy.spatial.distance import cosine
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def set_gpu_devices(devices='0,1'):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

def new_gpu_config(fraction=0.5):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = fraction 
    return config

class CDssm(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__( self, sequence_length, embedding_size, vocab_size,
            filter_sizes, num_filters,hidden_sizes, batch_size=100, l2_reg_lambda=0.01,neg = 5):
        # Placeholders for input, output and dropout
        self.query_in = tf.placeholder(tf.int32, [None, sequence_length], name="query_in")
        self.doc_in = tf.placeholder(tf.int32, [None, sequence_length], name="doc_in")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_loss = tf.constant(0.0)


        # Keeping track of l2 regularization loss (optional)
        self.sequence_length = sequence_length
        self.filter_sizes = filter_sizes
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.vocab_size = vocab_size
        self.l2_reg_lambda = l2_reg_lambda
        self.neg = neg
        self.hidden_sizes = hidden_sizes

        self.conv_ws = []
        self.conv_bs = []

        # Embedding layer
        #with tf.device('/cpu:0'), tf.name_scope("embedding"):
        with tf.name_scope("embedding"):
            wsq = np.sqrt(6.0 / (vocab_size + embedding_size))
            self.embedding_W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -wsq, wsq),
                name="W")
            self.l2_loss += tf.nn.l2_loss(self.embedding_W)

        # Create a convolution + maxpool layer for each filter size
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.conv_ws.append(W)
                self.conv_bs.append(b)

        # Final (unnormalized) scores and predictions
        num_filters_total = num_filters * len(filter_sizes)
        with tf.name_scope("output"):
            self.out_Ws = []
            self.out_bs = []
            tmp = [num_filters_total] + hidden_sizes
            for i in xrange(0, len(tmp) - 1):
                wsq = np.sqrt(6.0 / (tmp[i] + tmp[i+1]))
                w = tf.get_variable(
                    "W_%d"%i,
                    shape=[tmp[i], tmp[i+1] ],
                    #initializer=tf.contrib.layers.xavier_initializer())
                    initializer=tf.random_uniform_initializer(minval=-wsq,maxval=wsq))
                b = tf.Variable(tf.constant(0.0001, shape=[tmp[i+1]]), name="b_%d"%i)
                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)
                self.out_Ws.append(w)
                self.out_bs.append(b)

        self.query_vec,self.query_vec_pred = self._predict(self.query_in)
        self.doc_vec,self.doc_vec_pred = self._predict(self.doc_in)
        tmp = tf.tile(self.doc_vec,[1,1])
        doc_vecs = tf.tile(self.doc_vec,[1,1])
        #BS = self.doc_vec.get_shape()[0]
        for i in xrange(neg):
            rand = random.randint(0,batch_size + i) % batch_size
            doc_vecs = tf.concat( [doc_vecs, 
                    tf.slice(tmp,[rand,0],[batch_size - rand, -1]),
                    tf.slice(tmp,[0,0],[rand, -1])], 0)
        print('after NEG, shape of docvec %s '%doc_vecs.get_shape())

        with tf.name_scope('Cosine_Similarity'):
            # Cosine similarity
            query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.query_vec), 1, True)), [neg + 1, 1])
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_vecs), 1, True))

            prod = tf.reduce_sum(tf.multiply(tf.tile(self.query_vec, [neg + 1, 1]), doc_vecs), 1, True)
            norm_prod = tf.multiply(query_norm, doc_norm)

            cos_sim_raw = tf.truediv(prod, norm_prod)
            cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [neg + 1, -1])) * 20

        print('cos_sim shape:%s'%cos_sim.get_shape())
        with tf.name_scope('Loss'):
            # Train Loss
            prob = tf.nn.softmax((cos_sim))
            self.hit_prob = tf.slice(prob, [0, 0], [-1, 1])
            print('hit_prob shape:%s'%self.hit_prob.get_shape())
            neg_prob = tf.slice(prob, [0, 1], [-1, -1])
            print('neg_prob shape:%s'%neg_prob.get_shape())
            #self.loss = -tf.reduce_mean(tf.log(self.hit_prob)) + tf.reduce_mean(tf.log(neg_prob)) * 0.2 + self.l2_loss * self.l2_reg_lambda
            #self.loss = -tf.reduce_mean(tf.log(self.hit_prob)) + self.l2_loss * self.l2_reg_lambda
            self.loss = -tf.reduce_mean(tf.log(self.hit_prob)) + self.l2_loss * self.l2_reg_lambda

        with tf.name_scope('Training'):
            # Optimizer
            #self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_step = tf.train.AdamOptimizer(0.01).minimize(self.loss,global_step = self.global_step)

    def _predict(self,X):
        embedded_chars = tf.nn.embedding_lookup(self.embedding_W, X)
        print('after lookup shape %s'%embedded_chars.get_shape())
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        print('after expanded shape %s'%embedded_chars_expanded.get_shape())
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            # Convolution Layer
            conv = tf.nn.conv2d(
                embedded_chars_expanded,
                self.conv_ws[i],
                strides=[1, 1, 1, 1],
                padding="VALID"
                )
            # Apply nonlinearity
            print("\n===conv:size:%d===" % filter_size)
            print("after conv, shape is %s"%conv.get_shape())
            h = tf.nn.relu(tf.nn.bias_add(conv, self.conv_bs[i]))
            #h = tf.nn.bias_add(conv, self.conv_bs[i])
            print("after relu, shape is %s"%h.get_shape())
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID')
            print("after pooled, shape is %s"%pooled.get_shape())
            #pooled = tf.nn.tanh(pooled)
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        print("after concat,conved output shape is %s"%h_pool.get_shape())
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        print("after flat, conved output shape is %s"%h_pool_flat.get_shape())
        h_pool_flat = tf.nn.softmax(h_pool_flat)
        print("after softmax, conved output shape is %s"%h_pool_flat.get_shape())

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            scores = h_drop
            scores_flat = h_pool_flat
            for i in xrange(len(self.out_Ws)):
                scores = tf.nn.tanh(tf.nn.xw_plus_b(scores, self.out_Ws[i], self.out_bs[i]))
                scores_flat = tf.nn.tanh(tf.nn.xw_plus_b(scores_flat, self.out_Ws[i], self.out_bs[i]))
                #scores = tf.nn.relu6(tf.nn.xw_plus_b(scores, self.out_Ws[i], self.out_bs[i]))
                #scores_flat = tf.nn.relu6(tf.nn.xw_plus_b(scores_flat, self.out_Ws[i], self.out_bs[i]))
        print("predict , shape is %s"%scores.get_shape())
        return scores,scores_flat
def load_vocab():
    vocab = []
    vocab_idx = {}
    inf = open('../../data/dssm/dssm_voc.txt','r')
    vocab.append('<zero_padding>')
    vocab_idx['<zero_padding>'] = '0'
    cnt = 1
    for line in inf:
        c = line.strip('\n')
        if not c: continue
        vocab.append(c)
        vocab_idx[c] = str(cnt)
        vocab_idx[c.decode('utf-8')] = str(cnt)
        cnt += 1
    return vocab, vocab_idx

def load_data(sentence_len = 800):
    inf = open('../../data/dssm/train.txt','r')
    X = []
    Y = []
    for line in inf:
        line = line.strip()
        its = line.split('\t')
        title = [int(i) for i in its[1].split()]
        content = [int(i) for i in its[2].split()]
        x = title[:sentence_len]
        y = content[:sentence_len]
        for i in xrange(len(x),sentence_len):
            x.append(0)
        for i in xrange(len(y),sentence_len):
            y.append(0)
        X.append(x)
        Y.append(y)
    return X,Y

def file_batch_iter(fname,epochs,batch_size,sentence_len):
    ep = 0
    X = []
    Y = []
    while ep < epochs:
        inf = open(fname,'r')
        print datetime.now(), 'open file, epochs',ep, fname
        ep += 1
        for line in inf:
            try:
                line = line.strip()
                if not line: 
                    inf.close()
                    inf = None
                    break

                its = line.split('\t')
                title = [int(i) for i in its[1].split()]
                content = [int(i) for i in its[2].split()]
                x = title[:sentence_len]
                y = content[:sentence_len]
                lx = len(x)
                for i in xrange(lx,sentence_len):
                    x.append(0)
                ly = len(y)
                for i in xrange(ly,sentence_len):
                    y.append(0)
                X.append(x)
                Y.append(y)
                if len(X) >= batch_size: 
                    yield np.asarray(X), np.asarray(Y)
                    X = [] 
                    Y = []
            except:
                traceback.print_exc()
                pass

sentence_len = 500
embedding_size = 128
batch_size = 100
vocab,vocab_idx = load_vocab()
vocab_size = len(vocab)
filter_sizes = [3,4,5]
num_filters = 300
hidden_sizes = [300,300,embedding_size]

def train(restore=False):

    set_gpu_devices('0,1')
    config = new_gpu_config()
    #x_train,y_train = load_data(sentence_len)
    #print 'load finished',len(x_train)

    print datetime.now(), 'building modle'
    model = CDssm(sentence_len,embedding_size,vocab_size,filter_sizes,num_filters,hidden_sizes,batch_size,neg=4)
    print datetime.now(), 'building modle end'
    train_op = model.train_step
    loss = model.loss
    epochs = 20
    init = tf.global_variables_initializer() 
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        if not restore: sess.run(init)
        else: 
            ckpt = tf.train.get_checkpoint_state('./model/')
            print(ckpt.model_checkpoint_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
            #saver.restore(sess,'./model/dssm')
        #batches = batch_iter( list(zip(x_train, y_train)), batch_size, epochs)
        print datetime.now(), 'init batches'
        batches = file_batch_iter('../../data/dssm/train.txt', epochs,batch_size,sentence_len)
        print datetime.now(), 'init batches end'
        for batch in batches:
            x_batch, y_batch = batch
            _,step,rloss = sess.run([train_op,model.global_step,loss],
                    feed_dict={model.query_in:x_batch, model.doc_in: y_batch, model.dropout_keep_prob:0.8})
            if step % 10 == 0: print datetime.now(),step,rloss
            if step % 10000 == 0:
                saver.save(sess,'./model/dssm',global_step=step)
        saver.save(sess,'./model/dssm',global_step=step)
        pass
def predict():
    set_gpu_devices('0')
    config = new_gpu_config()
    #x_train,y_train = load_data(sentence_len)
    #print 'load finished',len(x_train)

    print datetime.now(), 'building modle'
    model = CDssm(sentence_len,embedding_size,vocab_size,filter_sizes,num_filters,hidden_sizes,batch_size,neg=4)
    print datetime.now(), 'building modle end'
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state('./model/')
        print(ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        print datetime.now(), 'init batches'
        batches = file_batch_iter('../../data/dssm/test.txt', 1,batch_size,sentence_len)
        print datetime.now(), 'init batches end'
        X = []
        Y = []
        loss = [] 
        titles = []
        for batch in batches:
            x_batch, y_batch = batch
            title_vecs,doc_vecs,rloss = sess.run([model.query_vec_pred,model.doc_vec_pred,model.loss],
                    feed_dict={model.query_in:x_batch, model.doc_in: y_batch, model.dropout_keep_prob:0.9})
            X += title_vecs.tolist()
            Y += doc_vecs.tolist()
            loss.append(rloss)
            print 'mean test loss', np.mean(loss), 'len X', len(X)
            for x in x_batch:
                t = []
                for i in x:
                    i = int(i)
                    if i == 0: break
                    t.append(vocab[i])
                titles.append(''.join(t))
    urls = []
    inf = open('../../data/dssm/test.txt','r')
    for line in inf:
        line = line.strip()
        if not line: continue
        its = line.split('\t')
        urls.append(its[0])
    print 'urls:',len(urls)
    docs = {"title_vecs":X,"content_vecs":Y,"urls":urls,'titles':titles}
    json.dump(docs,open('./test_doc_vec.json','w'),ensure_ascii=False,indent=1)
    test_sims(urls,X,Y,titles)

def test_sims(urls,X,Y,titles):
    # calc: title content sims
    TOP = 10000
    sims = {}
    for i in xrange(len(urls)):
        sim = 1 - cosine(X[i], Y[i])
        sims[urls[i]] = sim

    sims = sorted(sims.iteritems(),key=lambda x: x[1],reverse=True)
    print json.dumps(sims,indent=1)

    dsims = {}
    # calc titles sims
    for i in xrange(len(urls)):
        ti = titles[i]+','+urls[i]
        dsims.setdefault(ti,{'title_sim':{},'content_sim':{},'combine_sim':{}})
    for i in xrange(len(urls)):
        t = {}
        c = {}
        ti = titles[i]+','+urls[i]
        for j in xrange(i+1,len(urls)):
            tj = titles[j]+','+urls[j]
            sim = 1 - cosine(X[i], X[j])
            dsims[ti]['title_sim'][tj] = sim
            dsims[tj]['title_sim'][ti] = sim
            sim = 1 - cosine(Y[i], Y[j])
            dsims[ti]['content_sim'][tj] = sim
            dsims[tj]['content_sim'][ti] = sim
            sim = 1 - cosine(X[i]+Y[i], X[j]+Y[j])
            dsims[ti]['combine_sim'][tj] = sim
            dsims[tj]['combine_sim'][ti] = sim

        t = sorted( dsims[ti]['title_sim'].iteritems(),key=lambda x: x[1],reverse=True)[:TOP]
        c = sorted( dsims[ti]['content_sim'].iteritems(),key=lambda x: x[1],reverse=True)[:TOP]
        a = sorted( dsims[ti]['combine_sim'].iteritems(),key=lambda x: x[1],reverse=True)[:TOP]

        dsims = {} # for test
        dsims[ti] = {'title_sim':t,'content_sim':c,'combine_sim':a}
        break

    res = {'self_sims':sims,'doc_sims':dsims} 

    json.dump(res,open('./test_doc_sim.json','w'),ensure_ascii=False, indent=1)
def test_sim_by_read():
    docs = json.load(open('./test_doc_vec.json','r'))
    test_sims(docs['urls'],docs['title_vecs'],docs['content_vecs'],docs['titles'])


if __name__ == '__main__':
    train(False)
    #predict()
    #test_sim_by_read()
