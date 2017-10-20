#coding:utf-8
import tensorflow as tf
import numpy as np
import random
import os
import json
import sys
import glob
import traceback
from model import *
from mytf_utils import *
from datetime import datetime
from data_helpers import decode_from_tfrecords,get_batch,load_test_dssm_data 
from pyhdfs import HdfsClient

reload(sys)
sys.setdefaultencoding('utf-8')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', "hdfs://ns1-backup/user/xiulei/data/dssm","")
tf.app.flags.DEFINE_string('train_dir', "hdfs://ns1-backup/user/xiulei/data/dssm","")
tf.app.flags.DEFINE_string('ps_hosts', "localhost","")
tf.app.flags.DEFINE_string('job_name', "ps","")
tf.app.flags.DEFINE_integer('task_index', 0,"")
tf.app.flags.DEFINE_string('worker_hosts', "localhost","")
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
model_path = "./model_yarn"
summary_path = "./train_summary_yarn"
learning_rate = 0.001

is_sync = 1 

def main(unused_argv):
    global model_path,summary_path,learning_rate,batch_size,FLAGS
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    n_workers = len(worker_hosts)
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        print('cur is ps server:'+FLAGS.ps_hosts)
        server.join()
    elif FLAGS.job_name == "worker":
        restore = False
        data_dir = FLAGS.data_dir
        train_dir = FLAGS.train_dir
        model_path = os.path.join(train_dir,model_path)
        summary_path = os.path.join(train_dir,summary_path)
        with tf.device(tf.train.replica_device_setter(
                        worker_device="/job:worker/task:%d" % FLAGS.task_index,
                        cluster=cluster)):
            # loading data
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
                train_files = train_files[:5]
                print ('train_files:%s'%train_files)


            query,doc=decode_from_tfrecords(train_files)
            batch_query,batch_doc=get_batch(query,doc,batch_size)#batch 生成测试  

            # building model
            model = CDssm(sentence_len,embedding_size,vocab_size,filter_sizes,num_filters,
                        hidden_sizes,batch_size,activeFn=activeFn)
            print (str(datetime.now())+ ' building modle end')

            query_vec = model._predict(batch_query)
            query_vec = tf.nn.dropout(query_vec, FLAGS.dropout_keep_prob)
            doc_vec = model._predict(batch_doc)
            doc_vec = tf.nn.dropout(doc_vec, FLAGS.dropout_keep_prob)
            loss_op = model.loss_op(query_vec,doc_vec,neg=NEG,l2_reg_lambda=0.05)
            opt = tf.train.AdamOptimizer(learning_rate)
            # Optimizer
            global_step = tf.Variable(0, name="global_step", trainable=False)
            if is_sync == 1:
                #同步模式计算更新梯度
                opt = tf.train.SyncReplicasOptimizer(  opt,
                        replicas_to_aggregate = n_workers,  
                        total_num_replicas = n_workers)  
            train_op = opt.minimize(loss_op, global_step=global_step)

            #saver = tf.train.Saver()
            summary = tf.summary.merge_all()
            #init = tf.global_variables_initializer() 

        if is_sync == 1 and  FLAGS.task_index == 0: 
            chief_queue_runner = opt.get_chief_queue_runner()  
            init_token_op = opt.get_init_tokens_op(0) 
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                    logdir = model_path,
                    #init_op = init,
                    summary_op=summary,
                    #saver=saver,
                    global_step=global_step,
                    save_model_secs=600,
                    save_summaries_secs=60)      
        with sv.managed_session(server.target) as sess:
            print (str(datetime.now()) + ' init batches')
            # 如果是同步模式
            if FLAGS.task_index == 0 and is_sync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            while  not sv.should_stop():
                _,step,rloss = sess.run([train_op,global_step,loss_op])
                print ('%s %d %f'%(datetime.now(),step,rloss))
                if step >= FLAGS.steps: break
        sv.stop()


if __name__ == '__main__':
    is_hdfs = FLAGS.data_dir.startswith('hdfs://')
    if not is_hdfs: set_gpu_devices('1')
    tf.app.run()
