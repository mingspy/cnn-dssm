#coding:utf-8
import tensorflow as tf
import numpy as np
import random
import os
from datetime import datetime
import traceback
import json
from data_helpers import decode_from_tfrecords,get_batch,load_test_dssm_data 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from pyhdfs import HdfsClient
import glob
from mytf_utils import *
from cdssm_model import *

'''
#tanh
filter_sizes = [4]
num_filters = 300
activeFn = tf.nn.tanh
model_path = "./model_tanh"
summary_path = "./train_summeray_tanh"

'''

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', "hdfs://ns1-backup/user/xiulei/data/dssm","")
tf.app.flags.DEFINE_string('train_dir', "hdfs://ns1-backup/user/xiulei/data/dssm","")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.9,"")
tf.app.flags.DEFINE_integer('steps', 1000000,"")

# relu
sentence_len = 1000
embedding_size = 128 
batch_size = 500
vocab_size=4469
print 'vocab_size',vocab_size
#filter_sizes = [3,4,5]
filter_sizes = [1,2,3,4,5]
num_filters = 64
#hidden_sizes = [300,300,embedding_size]
hidden_sizes = [embedding_size]
NEG = 4 
#activeFn = tf.nn.relu
activeFn = tf.nn.tanh
model_path = "./model"
summary_path = "./train_summary"
learning_rate = 0.001


def train():

    global model_path,summary_path,learning_rate,batch_size
    restore = False
    curdir = os.path.dirname(os.path.realpath(__file__))
    data_dir = FLAGS.data_dir
    train_dir = FLAGS.train_dir
    model_path = os.path.join(train_dir,model_path)
    summary_path = os.path.join(train_dir,summary_path)

    train_files = []
    print('data_dir: '+data_dir)
    is_hdfs  = False
    if data_dir.startswith('hdfs://'):
        is_hdfs  = True
        if data_dir.startswith('hdfs://ns3-backup'):
            client = HdfsClient(hosts='yz48226.hadoop.data.sina.com.cn:50070')
            fdir = data_dir.replace('hdfs://ns3-backup','')
        elif data_dir.startswith('hdfs://ns1-backup'):
            client = HdfsClient(hosts='yz48212.hadoop.data.sina.com.cn:50070')
            fdir = data_dir.replace('hdfs://ns1-backup','')
        fileList = client.listdir(fdir)
        for name in fileList:
            name_new = os.path.join(data_dir, name)
            train_files.append(name_new)
            print('found train_file %s'%name_new)
    else:
        train_files = glob.glob(data_dir+'/tfrecords.dssm.*')
        train_files.sort()
        train_files = train_files
        print ('train_files:%s'%train_files)
    query,doc=decode_from_tfrecords(train_files)
    batch_query,batch_doc=get_batch(query,doc,batch_size)#batch 生成测试  

    model = CDssm(sentence_len,embedding_size,vocab_size,filter_sizes,num_filters,
                hidden_sizes,batch_size,activeFn=activeFn)
    print (str(datetime.now())+ ' building modle end')

    query_vec = model._predict(batch_query)
    query_vec = tf.nn.dropout(query_vec, FLAGS.dropout_keep_prob)
    doc_vec = model._predict(batch_doc)
    doc_vec = tf.nn.dropout(doc_vec, FLAGS.dropout_keep_prob)
    loss_op = model.loss_op(query_vec,doc_vec,neg=NEG,l2_reg_lambda=0.05)
    # Optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op,global_step = global_step)

    saver = tf.train.Saver()
    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer() 
    with tf.Session() as sess:  
        if not is_hdfs:
            train_writer = tf.summary.FileWriter(summary_path, sess.graph) 
        if not restore: sess.run(init)
        else: 
            ckpt = tf.train.get_checkpoint_state(model_path)
            print(ckpt.model_checkpoint_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        print (str(datetime.now()) + ' init batches')
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(coord=coord)  

        for batch in xrange(FLAGS.steps):
            if not is_hdfs:
                _,step,rloss,rsummary = sess.run([train_op,global_step,loss_op,summary])
            else:
                _,step,rloss = sess.run([train_op,global_step,loss_op])

            if step % 10 == 0:
                print ('%s %d %f'%(datetime.now(),step,rloss))
            if not is_hdfs and step % 100 == 0:
                train_writer.add_summary(rsummary, step) 
            if step % 2000 == 0:
                saver.save(sess,model_path,global_step=step)
        saver.save(sess,model_path,global_step=step)
        coord.request_stop()#queue需要关闭，否则报错  
        coord.join(threads)  

def main(unused_argv):
    train()

if __name__ == '__main__':
    is_hdfs = FLAGS.data_dir.startswith('hdfs://')
    if not is_hdfs: set_gpu_devices('1')
    tf.app.run()
