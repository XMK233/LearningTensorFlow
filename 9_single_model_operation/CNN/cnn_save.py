from cnn_settings import *

# https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
# other cnn codes: https://www.datacamp.com/community/tutorials/cnn-tensorflow-python

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32]),
                                      name= 'W_conv1'),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64]),
                                      name = 'W_conv2'),
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]),
                                   name= 'W_fc'),
               'out': tf.Variable(tf.random_normal([1024, N_CLASSES]),
                                  name= 'out')}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32]),
                                     name= 'b_conv1'),
              'b_conv2': tf.Variable(tf.random_normal([64]),
                                     name= 'b_conv2'),
              'b_fc': tf.Variable(tf.random_normal([1024]),
                                  name= 'b_fc'),
              'out': tf.Variable(tf.random_normal([N_CLASSES]),
                                 name= 'out')}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, KEEP_RATE)

    output = tf.add(tf.matmul(fc, weights['out']),
                    biases['out'],
                    name= "predict_output")

    return output

tnd1, tnl1, tnd2, tnl2, tstd, tstl = Dataset().tri_split(0.45, 0.9)


def train_neural_network1(train_data, train_label, test_data, test_label):
    x = tf.placeholder(tf.float32, [None, 784], name="x-input")
    y = tf.placeholder(tf.float32, name="y-input")

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction,
            labels=y),
        name="cost"
    )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    saver = tf.train.Saver(max_to_keep= 20)

    with tf.control_dependencies([optimizer]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(len(train_data) / BATCH_SIZE)):
                epoch_x, epoch_y = get_Batch1(train_data,
                                              train_label,
                                              BATCH_SIZE)
                _, c = sess.run([train_op, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            # 普通ckpt保存
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=epoch)
            saver.export_meta_graph(os.path.join(MODEL_SAVE_PATH, MODEL_NAME) + "-%d" % (epoch) + ".json",
                                    as_text=True)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_data, y: test_label}))

        # visualize the graph
        writer = tf.summary.FileWriter(os.path.join(SUMMARY_PATH, "original"),
                                       tf.get_default_graph())
        writer.close()

        # confusion_matrix,用来计算model performance
        confusion_matrix = tf.contrib.metrics.confusion_matrix(
            tf.argmax(y, 1),
            tf.argmax(prediction, 1)
        )
        cm = confusion_matrix.eval({x: test_data,
                                    y: test_label}
                                   )
        print(cm)
        return cm

if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)
print(os.getcwd())

# mnist dataset: https://blog.csdn.net/gaoyueace/article/details/79056085
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#train_neural_network(mnist)

cm_origin = train_neural_network1(tnd1, tnl1, tstd, tstl)
performance_metrics(cm_origin)
