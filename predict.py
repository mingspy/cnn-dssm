#coding:utf-8
import os
import argparse   
from mytf_utils import *
from data_helpers import *
from datetime import datetime
from scipy.spatial.distance import cosine
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def load_vocab():
    global args
    vocab = []
    vocab_idx = {}
    inf = open(args.vocab_path,'r')
    cnt = 0 
    for line in inf:
        c = line.strip('\n')
        if not c: continue
        vocab.append(c)
        vocab_idx[c] = str(cnt)
        vocab_idx[c.decode('utf-8')] = str(cnt)
        cnt += 1
    return vocab, vocab_idx


def predict(graph, query_in,query_doc):
    def test_phrases_sim(sess,query_doc,vocab_idx):
        test_fname = './test_phrases.txt'
        if not os.path.exists(test_fname): return 
        inf = open(test_fname,'r')
        for line in inf:
            line = line.strip()
            test_phrases = [i.decode('utf-8') for i in line.split('|')]
            test_vecs = []
            for p in test_phrases:
                x = [1]
                for i in p:
                    if i in vocab_idx: x.append(int(vocab_idx[i]))
                x = x[:args.sentence_len - 2]
                x.append(2)
                r = args.sentence_len - len(x)
                for i in xrange(r):
                    x.append(0)
                test_vecs.append(x)
                pass
            test_vecs = sess.run(query_doc, feed_dict={query_in:test_vecs})
            print test_phrases[0],':',
            for i,w in enumerate(test_phrases):
                if i == 0: continue
                print  w, 1 - cosine(test_vecs[0],test_vecs[i]),
            print '\n---------------------'
        pass
    batch_size = 200
    vocab,vocab_idx = load_vocab()

    config = new_gpu_config()

    with tf.Session(graph=graph, config=config) as sess:
        print datetime.now(), 'init batches'
        batches = load_test_dssm_data(args.test_path, args.sentence_len)
        print datetime.now(), 'init batches end'
        X = []
        Y = []
        titles = []
        bz = len(batches[0]) / batch_size
        for batch in xrange(bz):
            x_batch, y_batch = batches[0][batch*batch_size:(batch+1) * batch_size],batches[1][batch*batch_size:(batch+1) * batch_size]
            title_vecs = sess.run(query_doc, feed_dict={query_in:x_batch})
            doc_vecs = sess.run(query_doc, feed_dict={query_in:y_batch})
            X += title_vecs.tolist()
            Y += doc_vecs.tolist()
            print( 'len X %d'% len(X))
            for x in x_batch:
                t = []
                for i in x:
                    i = int(i)
                    if i == 0: break
                    t.append(vocab[i])
                titles.append(''.join(t))

        test_phrases_sim(sess,query_doc,vocab_idx)

    urls = []
    inf = open(args.test_path,'r')
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
    #print json.dumps(sims,indent=1)

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


parser = argparse.ArgumentParser()  
parser.add_argument("--model_dir", default="./model_yarn/", type=str, help="Frozen model file to import")  
parser.add_argument("--frozen_model_name", default="frozen_dssm.pb", type=str, help="Frozen model file to import")  
parser.add_argument("--sentence_len", default="1000", type=int, help="Frozen model file to import")  
parser.add_argument("--vocab_path", default="../../data/dssm/dssm_voc.txt", type=str, help="vocab")  
parser.add_argument("--test_path", default="../../data/dssm/test.txt", type=str, help="vocab")  
args = parser.parse_args()  

set_gpu_devices('2')

freeze_graph(args.model_dir,frozen_model_name = args.frozen_model_name)
graph = load_graph(os.path.join(args.model_dir,args.frozen_model_name))

for op in graph.get_operations():  
    print(op.name,op.values())  

query_in = graph.get_tensor_by_name('prefix/query_in:0')  
query_doc = graph.get_tensor_by_name('prefix/query_doc:0')  

predict(graph, query_in,query_doc)

print('done!')
