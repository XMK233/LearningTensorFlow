# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


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

#
#
#
FLAGS = tf.app.flags.FLAGS

def build_model(x, y_):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    #
    y = inference(x, regularizer)
    global_step = tf.contrib.framework.get_or_create_global_step()

    #
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        60000 / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

    #
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(
        tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, train_op]):
        train_op = tf.no_op()
    return global_step, loss, train_op


def main(argv=None):
    # 解析flags并通过tf.train.ClusterSpec配置TensorFlow集群。
    # parse the flags, and use tf.train.ClusterSpec to configure the TensorFlow cluster

    # 参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程。server.join()会
    # 一致停在这条语句上。
    # Parameter server will only manage the parameters, they will not execute the training and computation
    # So use the server.join(), make the program stops here.

    # if the program is run on worker nodes, then the following code will be executed.
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

    # tf.train.replica_device_setter function:
    # tf.train.replica_device_setter will deploy all of
    # the parameters to parameter server.
    # At the same time it will deploy the computation to worker servers
        #
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    global_step, loss, train_op = build_model(x, y_)

    hooks = [tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False)

    #
    saver = tf.train.Saver()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=MODEL_SAVE_PATH,
                                           hooks=hooks,
                                           save_checkpoint_steps=1000,
                                           save_summaries_secs=60,
                                           config=sess_config) as mon_sess:
        print("session started.")
        step = 0
        start_time = time.time()

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

