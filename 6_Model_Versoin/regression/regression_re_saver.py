# https://www.jianshu.com/p/7ebee4d10e49
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

def save_mode_pb(pb_file_path):
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)
    # 这里的输出需要加上name属性
    op = tf.add(xy, b, name='op_to_store')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    path = os.path.dirname(os.path.abspath(pb_file_path))
    if os.path.isdir(path) is False:
        os.makedirs(path)

    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

    # with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
    #     f.write(constant_graph.SerializeToString())

    tf.train.write_graph(constant_graph, './', pb_file_path , as_text=True)

    # test
    feed_dict = {x: 2, y: 3}
    print(sess.run(op, feed_dict))

def restore_mode_pb(pb_file_path):
    sess = tf.Session()
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        writer = tf.summary.FileWriter('RGRS_model/out/', sess.graph)
        writer.close()

    print(sess.run('b:0'))

    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')

    op = sess.graph.get_tensor_by_name('op_to_store:0')

    ret = sess.run(op, {input_x: 5, input_y: 5})
    print(ret)

save_mode_pb("RGRS_model/rgrs.pb")
#restore_mode_pb("RGRS_model/rgrs.pb")