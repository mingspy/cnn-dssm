#coding:utf-8

#from util.settings import *
from datetime import datetime
#from util.kafka_handler import *
from collections import Counter
import numpy as np
import traceback
import time
import glob
import json
import sys
import os
import logging
import copy
reload(sys)
sys.setdefaultencoding('utf-8')

MIN_SIM = 0.9
MAX_IDX_LEN = 10000
MAX_META_KEYS = 100



'''
model: cluster model
model format:{fp1:{'lda':center,'entity':entityset,'docs':title_fp}}

fp: doc fp
entity: enityt set
lda: lda value
length: format lda to certain length, i.e. 1000
min_sim: min similarity for cluster, less than this value will be a new cluster
'''
class ClusterMeta(object):
    def __init__(self, cid = '', center=0, center_sqrt = 0, docs=[],  docs_num = 0, keys = {}):
        self.set_values(cid,center,center_sqrt,docs,docs_num, keys)

    def set_values(self, cid = '', center=[], center_sqrt = 0, docs=[], docs_num=0, keys ={} ):
        self.cid = cid
        self.center = np.asarray(center)
        self.center_sqrt = center_sqrt
        self.docs = docs
        self.keys = Counter(dict(keys))
        self.docs_num = docs_num
        # histery not save docs_num, so, 
        # when loading, reset by actually docs len.
        if self.docs_num == 0:
            self.docs_num = len(self.docs)

    def  __string__(self):
        return json.dumps(self.to_dict())
    def to_dict(self):
        return {'cid':self.cid,'center':self.center.tolist(),
                'center_sqrt':float(self.center_sqrt),
                'docs':self.docs,
                'keys':dict(self.keys),
                'docs_num':self.docs_num}
    @staticmethod
    def from_dict(dict_value):
        return ClusterMeta( 
                dict_value.get('cid',''), 
                dict_value.get('center',[]),
                dict_value.get('center_sqrt',0.0),
                dict_value.get('docs',[]),
                dict_value.get('docs_num',0),
                dict_value.get('keys',{}) )

    def add_doc(self, docid, doc_vec,keys):
        ndocs = self.docs_num
        if ndocs == 0: 
            self.center = np.asarray(doc_vec)
        else:
            self.center = ( ndocs * self.center + np.asarray(doc_vec) ) /(ndocs + 1)
        self.center_sqrt = np.sqrt(np.sum(np.dot(self.center, self.center)))
        self.docs.append(docid)
        if keys:
            for k in keys:
                self.keys.setdefault(k,0)
                self.keys[k] += 1
        self.docs_num += 1
    def merge(self,another):
        self.center = (self.center * self.docs_num + another.center * another.docs_num)
        self.docs += another.docs
        self.docs_num += another.docs_num
        self.center /= self.docs_num
        self.center_sqrt = np.sqrt(np.sum(np.dot(self.center, self.center)))
        self.keys += another.keys
        if len(self.keys) > MAX_META_KEYS:
            self.keys = Counter(dict(sorted(self.keys.iteritems(), key = lambda x:x[1],reverse=True)[:MAX_META_KEYS])) 

    def similarity(self, doc_vec, doc_vec_sqrt):
        return np.dot(doc_vec, self.center) / self.center_sqrt / doc_vec_sqrt

class HCluster(object):
    def __init__(self,min_sim=MIN_SIM,model = None,model_path=None):
        self.log = logging.getLogger('HCluster')
        self.min_sim = min_sim
        self._meta = {}
        self._idx = {}
        self._black_words = set()
        self.max_cid = 0
        self.load(model,model_path)
        
    def load(self,model,model_path):
        if not model and model_path:
            model = json.load(open(model_path,'r'))
        # load model 
        if not model: return

        if 'min_sim' in model: self.min_sim = model.get('min_sim',MIN_SIM)
        if 'max_cid' in model: self.max_cid = model['max_cid']
        if '_black_words' in model: self._black_words = set(model['_black_words'])

        meta = model.get("_meta",{})
        for k,v in  meta.iteritems():
            self._meta[k] = ClusterMeta.from_dict(v)
        for k,v in model.get("_idx",{}).iteritems():
            self._idx[k] = set(v)
        self.log.info(' loaded clusters:%d'%len(self._meta))
    def save(self,path):
        model = {}
        model = {"min_sim":self.min_sim,
                "max_cid":self.max_cid }
        idx = {}
        for k,v in self._idx.iteritems():
            idx[k] = list(v)
        model['_idx'] = idx

        meta = {}
        for k,v in self._meta.iteritems():
            meta[k] = v.to_dict()
        model['_meta'] = meta
        model['_black_words'] = list(self._black_words)
        json.dump(model, open(path,'w'),ensure_ascii = False,indent=1)

    #倒排索引
    def _build_idx(self,words,cid):
        to_rm = set()
        for i in words:
            i = i
            if i in self._black_words: continue
            self._idx.setdefault(i,set())
            self._idx[i].add(cid)
            if len(self._idx[i]) > MAX_IDX_LEN:
                self._black_words.add(i)
                to_rm.add(i)
        for i in to_rm:
            self._idx.pop(i)

    def cluster(self,docid,doc_vec,keys):
        if isinstance(keys,list):
            tmp = {}
            for i in keys: tmp[i] = 1
            keys = tmp
        if keys:
            clusters = {}
            for i in keys:
                if i in self._idx: 
                    for j in self._idx[i]:
                        clusters.setdefault(j,0)
                        clusters[j] += 1
            clusters = sorted(clusters.iteritems(), key=lambda x:x[1],reverse=True)
            clusters = clusters[:100]
            tmp = set()
            for k in clusters: tmp.add(k[0])
            clusters = tmp
        else: 
            clusters = self._meta.keys()

        doc_vec = np.asarray(doc_vec)
        doc_vec_sqrt = np.sqrt(np.sum(np.dot(doc_vec,doc_vec)))
        sims = {}
        for k in clusters:
            if k not in self._meta: continue
            meta = self._meta[k]
            sim = meta.similarity(doc_vec, doc_vec_sqrt)
            if sim > self.min_sim: sims[k] = sim

        if not sims:
            cid = str(self.max_cid)
            self.max_cid += 1
            self._meta[cid] = ClusterMeta(cid, doc_vec, doc_vec_sqrt, [docid],1,keys)
        else:
            sims = sorted(sims.iteritems(),key=lambda x:x[1],reverse=True)
            cid = sims[0][0]
            self._meta[cid].add_doc(docid,doc_vec,keys)

        self._build_idx(keys,cid)
        self.merge_sims(sims)

        return cid
    def merge_sims(self,sims):
        if len(sims) < 2: return 
        cid = sims[0]
        if cid not in self._meta: return
        m = self._meta[cid]
        for i in xrange(1,len(sims)):
            cid2 = sims[i][0]
            if cid2 not in self._meta: continue
            m2 = self._meta[cid2]
            if m.similarity(m2.center,m2.center_sqrt) > self.min_sim:
                m.merge(m2)
                self._meta.pop(cid2)

    def merge(self,min_sim,anothers=None,full_compare=False):
        if anothers:
            if not isinstance(anothers,list): anothers = [anothers]
            for a in anothers:
                self._black_words |= a._black_words
                for k,v in a._idx.iteritems():
                    tmp = set()
                    for i in v:
                        tmp.add(str(self.max_cid + int(i)))
                    if k in self._idx: self._idx[k] |= tmp
                    else: self._idx[k] = tmp
                for k,v in a._meta.iteritems():
                    k = str(self.max_cid + int(k))
                    self._mata[k] = v
                self.max_cid += a.max_cid
        # merge meta
        cids = self._meta.keys()
        cids = sorted(cids, key=lambda x:int(x))
        lc = len(cids)
        print datetime.now(),'merge by min_sim', min_sim
        cnt = 0
        for i in xrange(lc):
            cid1 = cids[i]
            if cid1 not in self._meta: continue
            m1 = self._meta[cid1]
            merged = []
            if full_compare: cids2 = cids[i+1:]
            else:
                cids2 = set()
                for k in m1.keys:
                    if k in self._idx: cids2 |= self._idx[k]

            for cid2 in cids2:
                if cid2 == cid1 : continue
                if cid2 not in self._meta: continue
                m2 = self._meta[cid2]
                sim = m1.similarity(m2.center,m2.center_sqrt)
                if sim > min_sim:
                    m1.merge(m2)
                    self._meta.pop(cid2)
                    merged.append((cid2,sim))
            if merged: print datetime.now(),'merged',cid1, merged
        self.min_sim = min_sim
        print datetime.now(), 'before merge, clusters:', lc, 'after merge,clusters:', len(self._meta)
        # rebuild index:
        self._idx = {}
        for k,v in self._meta.iteritems():
            self._build_idx(v.keys.keys(),k)


ip = os.popen("ifconfig -a | grep inet |head -1 | awk '{print $2}'").read().strip()

def run_cluster():
    model = HCluster()
    cnt = 0
    docs = json.load(open('./test_doc_vec.json','r'))
    urls = docs['urls']
    tv = docs['title_vecs']
    cv = docs['content_vecs']
    titles = docs['titles']

    for i in xrange(len(urls)):
        ti = titles[i] + ',' + urls[i]
        model.cluster(ti,tv[i]+cv[i],[])
    model.merge(MIN_SIM, full_compare = False)
    model.save('./dist_cluster_result_%s.json'%ip)
            


def merge_results():
    m = HCluster(model_path = './dist_cluster_result_%s.json'%ip)
    Last = 40
    step = 5
    start_sim = int (MIN_SIM * 100)
    for i in xrange(0, Last, step):
        s = start_sim - i
        #hi =  ( s != start_sim )
        full = (i == Last - step)
        m.merge(s / 100.0,full_compare = full)
        m.save('merged_result_%s_%d.json'%(ip,s))

def dump_results():
    fname = sys.argv[1]
    limit = int(sys.argv[2])
    docs = json.load(open(fname,'r'))
    ret = {}
    docs = docs['_meta']
    for k,v in docs.iteritems():
        if v['docs_num'] >= limit:
            keys = sorted(v['keys'].iteritems(),key=lambda x: -x[1])
            ret[k] = [v['docs_num'],v['docs'],keys[:10]]
    print 'total clusters:',len(ret)
    print json.dumps(ret,ensure_ascii=False,indent=1)

def estimate_cluster_performance(cluster_result, docvecs, min_docs = 10):
    selected = {}
    for k,m in cluster_result.iteritems():
        if m.docs_num < min_docs: continue
        selected[k] = m 

    cluster_len = len(selected)
    print 'after filter len clusters:',cluster_len
    cls = selected.keys()
    dist = []
    for i in range(cluster_len):
        m =  selected[cls[i]]
        for j in range(i+1, cluster_len):
            m2 =  selected[cls[j]]
            d = 1 - m.similarity(m2.center,m2.center_sqrt)
            dist.append(d)
    ind = np.mean(dist)
    print len(dist),'cluster independence', ind 
    cohesion = []
    for k,m in selected.iteritems():
        for i in m.docs:
            vec = docvecs[i]
            vec_sqrt = np.sqrt(np.sum(np.dot(vec, vec)))
            s = m.similarity( vec,vec_sqrt)
            cohesion.append(s)
    co = np.mean(cohesion)
        
    print len(cohesion),'cluster cohesion', co
    print co,ind,co * ind  * 2 /(co + ind)
    return co,ind,co * ind  * 2 /(co + ind)
def estimate():
    m = HCluster(model_path = './dist_cluster_result_%s.json'%ip)
    cluster_result = m._meta
    docs = json.load(open('./test_doc_vec.json','r'))
    urls = docs['urls']
    tv = docs['title_vecs']
    cv = docs['content_vecs']
    titles = docs['titles']
    docvecs = {}
    for i,title in enumerate(titles):
        ti = title+','+urls[i]
        docvecs[ti] = tv[i]+cv[i]
    estimate_cluster_performance(cluster_result, docvecs, 5)


if __name__ == "__main__":
    #merge_results()
    dump_results()
    #run_cluster()
    #estimate()
