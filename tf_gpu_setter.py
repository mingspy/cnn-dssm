import os
import tensorflow as tf

def set_gpu_devices(devices='0,1'):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

def new_gpu_config(fraction=0.4):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = fraction 
    return config

