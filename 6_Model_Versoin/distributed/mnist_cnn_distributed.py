import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os, time
# https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
# other cnn codes: https://www.datacamp.com/community/tutorials/cnn-tensorflow-python

MODEL_SAVE_PATH = "CNN_model/"
MODEL_NAME = "cnn_model"
KEEP_RATE = 0.8
TRAINING_STEPS = 3

N_CLASSES = 10
BATCH_SIZE = 128

FLAGS = tf.app.flags.FLAGS

#
#
tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
#
tf.app.flags.DEFINE_string(
    'ps_hosts', ' scale05.eecs.yorku.ca:9994,scale05.eecs.yorku.ca:9995',
    'Comma-separated list of hostname:port for the parameter server jobs. e.g. "tf-ps0:2222,tf-ps1:1111" ')
#
tf.app.flags.DEFINE_string(
    'worker_hosts', ' scale05.eecs.yorku.ca:9996,scale05.eecs.yorku.ca:9997',
    'Comma-separated list of hostname:port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:1111" ')
#
#
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')

#
#

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, N_CLASSES]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([N_CLASSES]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, KEEP_RATE)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def build_model(x, y): #train_neural_network
    prediction = convolutional_neural_network(x)
    global_step = tf.contrib.framework.get_or_create_global_step()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= prediction, labels= y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.control_dependencies([optimizer]):
        train_op = tf.no_op(name='train')

    return cost, train_op, global_step

    '''hm_epochs = 3
    saver = tf.train.Saver()

    with tf.control_dependencies([optimizer]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / BATCH_SIZE)):
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
                _, c = sess.run([train_op, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        saver.export_meta_graph(os.path.join(MODEL_SAVE_PATH, MODEL_NAME) + ".json", as_text=True)'''

def main(argv=None):
    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)
    print(os.getcwd())

    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # use tf.train.ClusterSpec and current task to initialize tf.train.Server。
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        with tf.device("/cpu:0"):
            server.join()

    is_chief = (FLAGS.task_id == 0)
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    device_setter = tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_id,
        cluster=cluster)

    with tf.device(device_setter):
        x = tf.placeholder(tf.float32, [None, 784], name="x-input")
        y = tf.placeholder(tf.float32, name="y-input")

        loss, train_op, global_steps = build_model(x, y)
        hooks = [tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False)

        #
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=MODEL_SAVE_PATH,
                                               hooks=hooks,
                                               save_checkpoint_secs=60,
                                               config=sess_config) as mon_sess:
            print("session started.")
            step = 0
            start_time = time.time()

            while not mon_sess.should_stop():
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                _, loss_value, global_step_value = mon_sess.run(
                    [train_op, loss, global_steps], feed_dict={x: xs, y: ys})

                # we can get the global step of training.
                # global_step_value
                if step > 0 and step % 1 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / global_step_value
                    format_str = "After %d training steps (%d global steps), " +\
                                 "loss on training batch is %g. (%.3f sec/batch)"
                    print (format_str % (step, global_step_value, loss_value, sec_per_batch))
                step += 1

    #train_neural_network(mnist)

if __name__ == '__main__':
    tf.app.run()