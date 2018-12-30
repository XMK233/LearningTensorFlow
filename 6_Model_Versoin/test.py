# 系数都以八进制存在pb文件中
# 八进制：octal
import numpy as np
import tensorflow as tf
import os

#q = np.fromstring(b"\020\003\000\000\364\001\000\000",dtype=np.int32)
#print(q)

# https://stackoverflow.com/questions/53085007/re-train-a-frozen-pb-model-in-tensorflow
print(os.getcwd())
with tf.gfile.GFile('MNIST_model/mnist_model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name = "")
    writer = tf.summary.FileWriter('MNIST_model/out/', graph)
    writer.close()
# then $ tensorboard --logdir out/

# https://blog.csdn.net/TwT520Ly/article/details/80228970
# model = 'MNIST_model/mnist_model.pb'
# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
# graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
# tf.import_graph_def(graph_def, name='graph')
# summaryWriter = tf.summary.FileWriter('log/', graph)