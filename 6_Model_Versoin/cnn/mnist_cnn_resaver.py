#https://jdarpinian.blogspot.com/2017/04/tensorflow-tidbit-incompatible-with.html
import tensorflow as tf
import os

MODEL_SAVE_PATH = "CNN_model/"
MODEL_NAME = "cnn_model"

def freeze_graph(input_checkpoint, output_graph, output_node_names):

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    # graph = tf.get_default_graph()  # 获得默认的图
    # input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # ==:input_graph_def
            output_node_names=output_node_names.split(","),
            variable_names_blacklist= ["import/Variable:0"]
        )  # 如果有多个输出节点，以逗号隔开

        # 下面这两句是为了保存和序列化输出
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        # 如果要输出可读模型，就放开这个注释。
        # tf.train.write_graph(output_graph_def, './', output_graph + "-textual", as_text=True)
        print("%d ops in the final graph." % len(output_graph_def.node))

freeze_graph(input_checkpoint= "CNN_model/cnn_model",
             output_graph= "CNN_model/cnn_model.pb",
             output_node_names= "train")