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
tf.app.flags.DEFINE_string('ps_hosts', "localhost","")
tf.app.flags.DEFINE_string('job_name', "ps","")
tf.app.flags.DEFINE_integer('task_index', 0,"")
tf.app.flags.DEFINE_string('worker_hosts', "localhost","")

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
is_sync = 1 

def main(unused_argv):
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

        with tf.device(tf.train.replica_device_setter( 
                    worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
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

            opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
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
                    logdir = FLAGS.train_dir,
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
