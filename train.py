#coding:utf-8
import os
import sys
import json
import glob
import random
import traceback
import numpy as np
from model import *
import tensorflow as tf
from mytf_utils import *
from datetime import datetime
from pyhdfs import HdfsClient
from data_helpers import decode_from_tfrecords,get_batch,load_test_dssm_data 
reload(sys)
sys.setdefaultencoding('utf-8')

tf.app.flags.DEFINE_string('data_dir', "hdfs://ns3-backup/dw_ext/sinarecmd/xiulei/data/dssm","")
tf.app.flags.DEFINE_string('train_dir', "hdfs://ns3-backup/dw_ext/sinarecmd/xiulei/output/dssm","")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.9,"dropout keep prob for docvecs")
tf.app.flags.DEFINE_integer('steps', 1000000,"how many steps run before end")
tf.app.flags.DEFINE_integer('batch_size', 500,"")
tf.app.flags.DEFINE_integer('sentence_len', 1000,"input sentence length")
tf.app.flags.DEFINE_integer('vocab_size', 4469,"vocab size")
tf.app.flags.DEFINE_integer('embedding_size', 128,"input embedding size")
tf.app.flags.DEFINE_string('conv_filter_sizes', "1,2,3,4,5","conv2d sizes")
tf.app.flags.DEFINE_integer('conv_out_channels', 64,"conv2d out channels")
tf.app.flags.DEFINE_string('hidden_sizes', "128","hidden sizes, such as \"256,1000,128\"")
tf.app.flags.DEFINE_integer('NEG', 4,"NEG size")
tf.app.flags.DEFINE_float('learning_rate', 0.001,"")
tf.app.flags.DEFINE_float('l2_reg_lambda', 0.05,"")

activeFn = tf.nn.tanh
FLAGS = tf.app.flags.FLAGS

def train():

    data_dir = FLAGS.data_dir
    print('data_dir: '+data_dir)
    train_files = tf.gfile.Glob(os.path.join(data_dir,'tfrecords.dssm.*'))
    print('train files:%s'%train_files)

    query,doc=decode_from_tfrecords(train_files,FLAGS.sentence_len)
    batch_query,batch_doc=get_batch(query,doc,FLAGS.batch_size)

    filter_sizes =[ int(i) for i in FLAGS.conv_filter_sizes.split(',') ]
    hidden_sizes =[ int(i) for i in FLAGS.hidden_sizes.split(',') ]

    model = CDssm( FLAGS.sentence_len, FLAGS.embedding_size, FLAGS.vocab_size,
                filter_sizes, FLAGS.conv_out_channels, hidden_sizes, 
                FLAGS.batch_size, activeFn=activeFn)
    print (str(datetime.now())+ ' building modle end')

    query_vec = model._predict(batch_query)
    query_vec = tf.nn.dropout(query_vec, FLAGS.dropout_keep_prob)
    doc_vec = model._predict(batch_doc)
    doc_vec = tf.nn.dropout(doc_vec, FLAGS.dropout_keep_prob)
    loss_op = model.loss_op(query_vec,doc_vec,neg=FLAGS.NEG,l2_reg_lambda = FLAGS.l2_reg_lambda)

    # Optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss_op,global_step = global_step)

    summary = tf.summary.merge_all()
    sv = tf.train.Supervisor( logdir = FLAGS.train_dir, global_step = global_step,
            summary_op=summary, save_model_secs = 600, save_summaries_secs = 60)

    with sv.managed_session() as sess:  
        print (str(datetime.now()) + ' init batches')
        while  not sv.should_stop():
            _,step,rloss = sess.run([train_op,global_step,loss_op])
            print ('%s %d %f'%(datetime.now(),step,rloss))
            if step >= FLAGS.steps: break
    sv.stop()

def main(unused_argv):
    train()

if __name__ == '__main__':
    is_hdfs = FLAGS.data_dir.startswith('hdfs://')
    if not is_hdfs: set_gpu_devices('2')
    tf.app.run()
