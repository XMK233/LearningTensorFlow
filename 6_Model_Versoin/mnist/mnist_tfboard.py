import tensorflow as tf
from tensorflow.python.platform import gfile
import os
#这是从文件格式的meta文件加载模型

MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
# graphdef.ParseFromString(gfile.FastGFile("/data/TensorFlowAndroidMNIST/app/src/main/expert-graph.pb", "rb").read())
# _ = tf.import_graph_def(graphdef, name="")
_ = tf.train.import_meta_graph(os.path.join(MODEL_SAVE_PATH, MODEL_NAME) + ".meta")
summary_write = tf.summary.FileWriter(os.path.join(MODEL_SAVE_PATH, MODEL_NAME) , graph)