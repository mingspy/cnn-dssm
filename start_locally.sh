#########################################################################
# File Name: start_train.sh
# Author: xiulei
# mail: mingspy@163.com
# Created Time: Mon 24 Jul 2017 10:28:10 AM CST
#########################################################################
#!/bin/bash

#word2vec -train ./data/word2vec_train.txt -save-vocab ./data/pre_vocab.txt -min-count 7
#python prepair_train_data.py reduce_vocab

#mv ./data/tmp_vocab.txt ./data/pre_vocab.txt

#python prepair_train_data.py remove_unkown

#word2vec -train train.txt -output vec.txt -size 256 -sample 1e-4 -window 7 -threads 10 -negative 5 -hs 1 -binary 0 -iter 100
if [ $1 = train ]; then
    echo 'start training '
    nohup python -u train.py --data_dir=../../data/dssm --train_dir=./model >& d2v.log  &
elif [ $1 = train2 ]; then
    echo 'start training2 '
    nohup python -u train.py --data_dir=../../data/dssm2 --train_dir=./model2 --sentence_len=600 --vocab_size=437403 >& dssm2.log  &
elif [ $1 = predict ]; then
    echo 'start predict'
    python -u predict.py --model_dir=./model >& ttt &
elif [ $1 = predict2 ]; then
    echo 'start predict2'
    python -u predict.py --model_dir=./model2 --sentence_len=600 --vocab_path=../../data/dssm2/dssm_voc2.txt --test_path=../../data/dssm2/test.txt>& ttt &
elif [ $1 = predict_hdfs ]; then
    echo 'start predict hdfs'
    python -u predict.py --model_dir=./hdfs/dssm1_model >& ttt &
elif [ $1 = predict_hdfs2 ]; then
    echo 'start predict hdfs'
    python -u predict.py --model_dir=./hdfs/dssm2_model --sentence_len=600 --vocab_path=../../data/dssm2/dssm_voc2.txt --test_path=../../data/dssm2/test.txt >& ttt &
elif [ $1 = board ]; then
    tensorboard --logdir ./model &
fi


