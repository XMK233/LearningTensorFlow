# save a model using tf.train.Saver()
# and then export a sevicable model from a tf.train.Saver().
# 代码解读： https://www.cnblogs.com/yinzm/p/7110870.html
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def save_model(sess, signature_def_map, output_dir):
  """Saves the model with given signature def map."""
  builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map=signature_def_map)
  builder.save()

def build_signature_def_from_tensors(inputs, outputs, method_name):
  """Builds signature def with inputs, outputs, and method_name."""
  return tf.saved_model.signature_def_utils.build_signature_def(
      inputs={
          key: tf.saved_model.utils.build_tensor_info(tensor)
          for key, tensor in inputs.items()
      },
      outputs={
          key: tf.saved_model.utils.build_tensor_info(tensor)
          for key, tensor in outputs.items()
      },
      method_name=method_name)

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

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 500
MOVING_AVERAGE_DECAY = 0.99
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

    global_step = tf.Variable(0, trainable=False)

    variable_averages = \
        tf.train.ExponentialMovingAverage \
            (MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = \
        variable_averages.apply \
            (tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
        (logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = \
        tf.train.GradientDescentOptimizer(learning_rate) \
            .minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step]
                                           , feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

        signature_def_map = {
            "serving_default":  # tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                build_signature_def_from_tensors({"x_input": x},
                                                 {"y_output": tf.argmax(y, 1)},
                                                 tf.saved_model.signature_constants.PREDICT_METHOD_NAME),
        }
        with open("deployed_version.txt", "r") as f:
            last_version = str(int(f.readline().strip()))
            new_version = str(int(last_version) + 1)
            save_model(sess, signature_def_map, os.path.join(MODEL_SAVE_PATH,
                                                             MODEL_NAME + "/%s" %(new_version)))
        with open("deployed_version.txt", "w") as f:
            f.write(new_version)



def main(argv=None):
    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)
    print(os.getcwd())

    mnist = input_data.read_data_sets("/tmp/MNIST_data"
                                      , one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()