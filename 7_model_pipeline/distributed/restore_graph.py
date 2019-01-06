# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

#
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

#
MODEL_SAVE_PATH = "model_storage"
MODEL_NAME = "readable_graph"
#
DATA_PATH = "/tmp/data"


def main(argv=None):
    # è§£æžflagså¹¶é€šè¿‡tf.train.ClusterSpecé…ç½®TensorFlowé›†ç¾¤ã€‚
    # parse the flags, and use tf.train.ClusterSpec to configure the TensorFlow cluster

    # å‚æ•°æœåŠ¡å™¨åªéœ€è¦ç®¡ç†TensorFlowä¸­çš„å˜é‡ï¼Œä¸éœ€è¦æ‰§è¡Œè®­ç»ƒçš„è¿‡ç¨‹ã€‚server.join()ä¼š
    # ä¸€è‡´åœåœ¨è¿™æ¡è¯­å¥ä¸Šã€‚
    # Parameter server will only manage the parameters, they will not execute the training and computation
    # So use the server.join(), make the program stops here.

    # if the program is run on worker nodes, then the following code will be executed.
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

    # tf.train.replica_device_setter function:
    # tf.train.replica_device_setter will deploy all of
    # the parameters to parameter server.
    # At the same time it will deploy the computation to worker servers
    #

    '''x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    global_step, loss, train_op = build_model(x, y_)'''

    hooks = [tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False)

    #
    saver = tf.train.import_meta_graph("model_storage/model.ckpt-10001.meta",
                                       clear_devices=True)
    with tf.train.MonitoredTrainingSession(checkpoint_dir=MODEL_SAVE_PATH,
                                           hooks=hooks,
                                           save_checkpoint_steps=1000,
                                           save_summaries_secs=60,
                                           config=sess_config) as mon_sess:

        global_step = tf.contrib.framework.get_or_create_global_step()
        print("session started.")
        step = 0
        start_time = time.time()

        graph = tf.get_default_graph()
        print([n.name for n in tf.get_default_graph().as_graph_def().node])
        y_ = graph.get_tensor_by_name("y-input:0")
        x = graph.get_tensor_by_name("x-input:0")
        train_op = graph.get_operation_by_name("train")
        loss = graph.get_tensor_by_name("loss")
        #
        #
        #
        while not mon_sess.should_stop():
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, global_step_value = mon_sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            # we can get the global step of training.
            # global_step_value
            if step > 0 and step % 100 == 0:
                duration = time.time() - start_time
                sec_per_batch = duration / global_step_value
                format_str = "After %d training steps (%d global steps), " + \
                             "loss on training batch is %g. (%.3f sec/batch)"
                print(format_str % (step, global_step_value, loss_value, sec_per_batch))
            step += 1

        saver.export_meta_graph(os.path.join(MODEL_SAVE_PATH, MODEL_NAME) + ".json", as_text=True)


if __name__ == "__main__":
    tf.app.run()

# how to run the code:
# python xxxx.py --job_name='xxx' --task_id=x --ps_hosts='xxxx:xxxx' --worker_hosts='xxxx:xxxx,xxxx:xxxx'
# (Remember, the port is necessary)
#
#

