# save a model using tf.train.Saver()
# and then export a sevicable model from a tf.train.Saver().
import tensorflow as tf

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


from tensorflow.examples.tutorials.mnist import input_data
import os
from tensorflow.contrib.session_bundle import exporter

BATCH_SIZE = 80 #100
LEARNING_RATE_BASE = 0.9 #0.8
LEARNING_RATE_DECAY = 0.89 #0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 500 #100
MOVING_AVERAGE_DECAY = 0.92 #0.99
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"


def train(mnist):
    x = tf.placeholder(tf.float32
                       , [None, INPUT_NODE]
                       , name='x-input')
    y_ = tf.placeholder(tf.float32
                        , [None, OUTPUT_NODE]
                        , name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, regularizer)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
        (logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    train_step = \
        tf.train.GradientDescentOptimizer(LEARNING_RATE_BASE) \
            .minimize(loss)

    with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _ = sess.run(train_op
                                           , feed_dict={x: xs, y_: ys})

        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        saver.export_meta_graph(os.path.join(MODEL_SAVE_PATH, MODEL_NAME) + ".json", as_text=True)


def main(argv=None):
    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)
    print(os.getcwd())

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()