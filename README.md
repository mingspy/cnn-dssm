# cnn-dssm
A tensorflow version of dssm using textcnn as feature extractor.
training data using document's title as query, document's content as hit doc in dssm.
training data size 1300w+

## notations
### Q means query words, here using document's title
### D+ means hit docs, here using document's content
### D- mean negative docs, here using negative document's content
### p(Q,D+) = cosine(Q,D+)

## cnn_dssm.py is version 1
  loss function =  - p(Q,D+)
## cdssm2.py is version 2
  loss function = max{1 - p(Q,D+) + p(Q,D-),0}

## tricks
In my practice:
    little learning_rate, such as 0.001, easier to converge
    small conv output,such as 64, easier to training
    active function tanh get better,relu usally got nan loss
    
    
## cdssm2.py tanh - loss 
![Alt loss](https://github.com/mingspy/cnn-dssm/blob/master/cdssm2_loss_lr0.001_fout64_cf12345.png) 

sentence_len = 100
embedding_size = 128
batch_size = 500
vocab,vocab_idx = load_vocab()
vocab_size = len(vocab) # 4469
print 'vocab_size',vocab_size
#filter_sizes = [3,4,5]
filter_sizes = [1,2,3,4,5]
num_filters = 64
hidden_sizes = [embedding_size]
NEG = 4 
learning_rate = 0.001
activeFn = tf.nn.tanh
model_path = "./model"
summary_path = "./train_summeray"

