import tensorflow as tf
import os

MODEL_SAVE_PATH = "CNN_model/"
MODEL_NAME = "cnn_model"

def restore_mode_pb(pb_file_path):
    sess = tf.Session()
    with tf.gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def)

        #writer = tf.summary.FileWriter('RGRS_model/out/', sess.graph)
        #writer.close()

    #print(sess.run('b:0'))

    #input_x = sess.graph.get_tensor_by_name('x:0')
    #input_y = sess.graph.get_tensor_by_name('y:0')

    #op = sess.graph.get_tensor_by_name('op_to_store:0')

    #ret = sess.run(op, {input_x: 5, input_y: 5})
    #print(ret)

restore_mode_pb("CNN_model/cnn_model.pb")