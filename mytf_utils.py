#-*- coding:utf-8 -*-  
import os, argparse  
import tensorflow as tf  
from tensorflow.python.framework import graph_util  

curdir = os.path.dirname(os.path.realpath(__file__))  
  
def set_gpu_devices(devices='0,1'):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

def new_gpu_config(fraction=0.5):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = fraction 
    return config

  
def freeze_graph(model_folder, output_node_names = "query_doc", frozen_model_name="/frozen_model.pb"):  
    # We retrieve our checkpoint fullpath  
    checkpoint = tf.train.get_checkpoint_state(model_folder)  
    input_checkpoint = checkpoint.model_checkpoint_path  
    print input_checkpoint
      
    # We precise the file fullname of our freezed graph  
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])  
    print absolute_model_folder
    output_graph = os.path.join(absolute_model_folder,frozen_model_name)
  
    # Before exporting our graph, we need to precise what is our output node  
    # this variables is plural, because you can have multiple output nodes  
    #freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点  
    #输出结点可以看我们模型的定义  
    #只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃  
    #所以,output_node_names必须根据不同的网络进行修改  
    #output_node_names = "output"  
  
    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated  
    clear_devices = True  
      
    # We import the meta graph and retrive a Saver  
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)  
  
    # We retrieve the protobuf graph definition  
    graph = tf.get_default_graph()  
    input_graph_def = graph.as_graph_def()  
  
    #We start a session and restore the graph weights  
    #这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen  
    #相当于将参数已经固化在了图当中   
    with tf.Session() as sess:  
        saver.restore(sess, input_checkpoint)  
  
        # We use a built-in TF helper to export variables to constant  
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess,   
            input_graph_def,   
            output_node_names.split(",") # We split on comma for convenience  
        )   
  
        # Finally we serialize and dump the output graph to the filesystem  
        with tf.gfile.GFile(output_graph, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
        print("%d ops in the final graph." % len(output_graph_def.node))  
  
def load_graph(frozen_graph_filename):  
    # We parse the graph_def file  
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())  
  
    # We load the graph_def in the default graph  
    with tf.Graph().as_default() as graph:  
        tf.import_graph_def(  
            graph_def,   
            input_map=None,   
            return_elements=None,   
            name="prefix",   
            op_dict=None,   
            producer_op_list=None  
        )  
    return graph  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--frozen_model_filename", default="./model_yarn/frozen_model.pb", type=str, help="Frozen model file to import")  
    args = parser.parse_args()  
    #加载已经将参数固化后的图  
    graph = load_graph(args.frozen_model_filename)  
  
    # We can list operations  
    #op.values() gives you a list of tensors it produces  
    #op.name gives you the name  
    #输入,输出结点也是operation,所以,我们可以得到operation的名字  
    for op in graph.get_operations():  
        print(op.name,op.values())  
        # prefix/Placeholder/inputs_placeholder  
        # ...  
        # prefix/Accuracy/predictions  
    #操作有:prefix/Placeholder/inputs_placeholder  
    #操作有:prefix/Accuracy/predictions  
    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字  
    #注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字  
    x = graph.get_tensor_by_name('prefix/query_in:0')  
    y = graph.get_tensor_by_name('prefix/query_doc:0')  
          
    with tf.Session(graph=graph) as sess:  
        y_out = sess.run(y, feed_dict={  
            x: [[1,3, 5, 7, 4, 5, 1, 1, 1, 1, 1,2,0,0]] # < 45  
        })  
        print(y_out) # [[ 0.]] Yay!  
    print ("finish") 
