#coding:utf-8
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import glob


def load_data_and_labels(data_file):
    # Load data from files
    print 'loading',os.path.realpath(data_file)
    examples = list(open(data_file, "r").readlines())
    examples = [s.strip().split() for s in examples]
    exs = []
    for i in examples:
        tmp = [int(j) for j in i]
        exs.append(tmp)
    examples = np.asarray(exs,dtype=np.int64)
    sz,mlen = examples.shape
    data,labels = examples[:,:mlen/2],examples[:,mlen/2:]
    return data,labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if end_index == data_size:
                start_index = data_size - batch_size
            yield shuffled_data[start_index:end_index]
  
def load_test_dssm_data(fname,sentence_len):
    X = []
    Y = []
    print datetime.now(), 'open file, epochs',fname
    inf = open(fname,'r')
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
            x = [1] + title[:sentence_len - 2] + [2]
            y = [1] + content[:sentence_len - 2] + [2]
            lx = len(x)
            for i in xrange(lx,sentence_len):
                x.append(0)
            ly = len(y)
            for i in xrange(ly,sentence_len):
                y.append(0)
            X.append(x)
            Y.append(y)
        except:
            traceback.print_exc()
            pass
    return X,Y

def encode_to_tfrecords(lable_file,data_root,sentence_len=1000,new_name='tfrecords.dssm',split_size = -1):  
    fn = 0
    def get_writer(fn):
        pname = os.path.join(data_root,new_name)
        if split_size > 0:
            pname += '.'+str(fn)
        writer=tf.python_io.TFRecordWriter(pname)
        return writer
    num_example=0  
    writer = get_writer(fn)
    fn += 1
    with open(lable_file,'r') as inf:  
        for l in inf:
            if split_size > 0 and num_example > 1 and num_example % split_size == 0:
                writer.close()
                writer = get_writer(fn)
                fn += 1
            its=l.strip().split('\t')  
            url = its[0]
            title = [int(i) for i in its[1].split()]
            content = [int(i) for i in its[2].split()]
            title = [1] + title[:sentence_len - 2] + [2]
            content = [1] + content[:sentence_len - 2] + [2]
            x = sentence_len - len(title)
            for i in xrange(x):
                title.append(0)
            x = sentence_len - len(content)
            for i in xrange(x):
                content.append(0)
            url = tf.compat.as_bytes(url)
            example=tf.train.Example(features=tf.train.Features(feature={  
                #'url':tf.train.Feature(bytes_list=tf.train.BytesList(value=url)),  
                'query':tf.train.Feature(int64_list=tf.train.Int64List(value=title)),  
                'doc':tf.train.Feature(int64_list=tf.train.Int64List(value=content))
            }))  
            serialized=example.SerializeToString()  
            writer.write(serialized)  
            num_example+=1  
            #if num_example == 100: break
    writer.close()  

#读取tfrecords文件  
def decode_from_tfrecords(filename,sentence_len=1000,num_epoch=None):  
    if not isinstance(filename,list):
        filename = [filename]
    filename_queue=tf.train.string_input_producer(filename,num_epochs=num_epoch)
    reader=tf.TFRecordReader()  
    _,serialized=reader.read(filename_queue)  
    example=tf.parse_single_example(serialized,features={  
        #'url':tf.FixedLenFeature([],tf.string),  
        'query':tf.FixedLenFeature([sentence_len],tf.int64),  
        'doc':tf.FixedLenFeature([sentence_len],tf.int64)  
        })  
    #url = tf.cast(example['url'], tf.string)  
    query = example['query']
    doc = example['doc']
    return query,doc
def get_batch(query,doc, batch_size):  
    #生成batch  
    #shuffle：capacity用于定义shuttle的范围，
    #如果是对整个训练数据集，获取batch，那么capacity就应该够大  
    #保证数据打的足够乱  
    querys, docs = tf.train.shuffle_batch([query, doc],batch_size=batch_size,  
            num_threads=16,capacity=2000+batch_size*3,min_after_dequeue=batch_size*2)  
    return querys,docs

def test():  
    #encode_to_tfrecords("../../data/dssm/train.txt","../../data/dssm/",split_size = 1000000)  

    files = glob.glob('../../data/dssm/tfrecords.dssm*')  

    query,doc=decode_from_tfrecords(files)
    batch_query,batch_doc=get_batch(query,doc,1000)#batch 生成测试  
    init=tf.global_variables_initializer()
    with tf.Session() as session:  
        session.run(init)  
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(coord=coord)  
        for l in range(5):
            #每run一次，就会指向下一个样本，一直循环  
            #url_np,query_np,doc_np=session.run([url,query,doc])#每调用run一次，那么  
            batch_query_np,batch_doc_np=session.run([batch_query,batch_doc])  
            print batch_doc_np
            print '------------------'
        coord.request_stop()#queue需要关闭，否则报错  
        coord.join(threads)  
