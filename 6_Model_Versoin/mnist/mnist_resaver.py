import tensorflow as tf
import os
# https://blog.csdn.net/guyuealian/article/details/82218092


#通过传入 CKPT 模型的路径得到模型的图和变量数据
#通过 import_meta_graph 导入模型中的图
#通过 saver.restore 从模型中恢复图中各个变量的数据
#通过 graph_util.convert_variables_to_constants 将模型持久化
#https://blog.csdn.net/guyuealian/article/details/82218092

MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"

def freeze_graph(input_checkpoint, output_graph, output_node_names="InceptionV3/Logits/SpatialSqueeze"):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    # graph = tf.get_default_graph()  # 获得默认的图
    # input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def= sess.graph_def,  # ==:input_graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        # 下面这两句是为了保存和序列化输出
        with tf.gfile.GFile(output_graph, "wb") as f:
           f.write(output_graph_def.SerializeToString())

        # 如果要输出可读模型，就放开这个注释。
        # tf.train.write_graph(output_graph_def, './', output_graph + "-textual", as_text=True)
        print("%d ops in the final graph." % len(output_graph_def.node))

# 输入ckpt模型路径
input_checkpoint = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
# 输出pb模型的路径
out_pb_path= os.path.join(MODEL_SAVE_PATH, MODEL_NAME) + ".pb"
# 调用freeze_graph将ckpt转为pb

output_node_names = "train"

freeze_graph(input_checkpoint, out_pb_path, output_node_names)
