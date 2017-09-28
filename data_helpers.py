import numpy as np
import re
import itertools
from collections import Counter
import os




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

if __name__ == '__main__':
    pass
    x,xl = load_data_and_labels('../data/segment/test.txt')
    print x.shape,xl.shape
    print x[0]
    print xl[0]
