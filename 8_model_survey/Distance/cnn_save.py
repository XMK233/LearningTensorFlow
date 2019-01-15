import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import pandas as pd

# https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
# other cnn codes: https://www.datacamp.com/community/tutorials/cnn-tensorflow-python

MODEL_SAVE_PATH = "CNN_model/"
MODEL_NAME = "cnn_model"
SUMMARY_PATH = "CNN_Logs"
KEEP_RATE = 0.8

N_CLASSES = 10
BATCH_SIZE = 128


def performance_metrics1(confusion_matrix):
    # https://blog.csdn.net/sihailongwang/article/details/77527970
    # 好像不一定对。值得斟酌。

    accu = [0 for i in range(N_CLASSES)]
    column = [0 for i in range(N_CLASSES)]
    line = [0 for i in range(N_CLASSES)]
    accuracy = 0
    recall = 0
    precision = 0
    for i in range(0, N_CLASSES):
        accu[i] = confusion_matrix[i][i]
    for i in range(0, N_CLASSES):
        for j in range(0, N_CLASSES):
            column[i] += confusion_matrix[j][i]
    for i in range(0, N_CLASSES):
        for j in range(0, N_CLASSES):
            line[i] += confusion_matrix[i][j]
    # for i in range(0, N_CLASSES):
    #     accuracy += float(accu[i])/len_labels_all
    for i in range(0, N_CLASSES):
        if column[i] != 0:
            recall += float(accu[i]) / column[i]
    recall = recall / N_CLASSES
    for i in range(0, N_CLASSES):
        if line[i] != 0:
            precision += float(accu[i]) / line[i]
    precision = precision / N_CLASSES
    f1_score = (2 * (precision * recall)) / (precision + recall)
    print("Average Precision(AP): %f, \n"
          "Average Recall(AR): %f, \n"
          "F1 of AP and AR: %f"
          % (precision, recall, f1_score)
          )
    return precision, recall, f1_score


def performance_metrics(confusion_matrix):
    # 上一个方法precision和recall在行列关系上好像有点奇怪，可能弄反了。
    # 而且计算的是各个类的平均值。
    # 不过基本思想是对的，在弄清楚弄反与否之前暂时不要删掉前一个方法。
    # 这个方法基本上是正统的计算方法了。而且会给出每一个类别的所有metrics。
    # 行列代表的含义和该链接所述的相同：https://zhuanlan.zhihu.com/p/33273532
    accu = [0 for i in range(N_CLASSES)]
    column = [0 for i in range(N_CLASSES)]
    line = [0 for i in range(N_CLASSES)]
    for i in range(0, N_CLASSES):
        accu[i] = confusion_matrix[i][i]
    for i in range(0, N_CLASSES):
        for j in range(0, N_CLASSES):
            column[i] += confusion_matrix[j][i]
    for i in range(0, N_CLASSES):
        for j in range(0, N_CLASSES):
            line[i] += confusion_matrix[i][j]
    total_num = sum(line)

    for i in range(N_CLASSES):
        TP = accu[i]
        FP = column[i] - accu[i]
        FN = line[i] - accu[i]
        TN = total_num - TP - FP - FN

        recall = float(TP) / (TP + FN)
        precision = float(TP) / (TP + FP)
        f1_score = (2 * (precision * recall)) / (precision + recall)
        print("For class %d: \n"
              "Average Precision(AP): %f, \n"
              "Average Recall(AR): %f, \n"
              "F1 of AP and AR: %f \n\n"
              % (i, precision, recall, f1_score)
              )


def get_Batch1(data, label, batch_size):
    # https://blog.csdn.net/sinat_35821976/article/details/82668555
    x_batch = data.sample(batch_size)
    y_batch = label.loc[list(x_batch.index)]
    return x_batch, y_batch


def get_Batch(data, label, batch_size):
    # https://blog.csdn.net/sinat_35821976/article/details/82668555
    print(data.shape, label.shape)
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32)
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32,
                                      allow_smaller_final_batch=False)
    return x_batch, y_batch


class Dataset:
    data = None
    label = None

    def __init__(self, path="/tmp/train.csv"):
        df = pd.read_csv(path, header=0)
        cols = df.columns.values.tolist()
        cols.pop(0)
        self.data = df[cols]
        self.label = pd.get_dummies(df['label'])

    def bi_split(self, percentage=0.3):
        length = round(len(self.data) * percentage)

        return self.data[0:length], self.label[0:length], \
               self.data[length + 1:], self.label[length + 1:]

    def tri_split(self, p1=0.3, p2=0.9):
        l1 = round(len(self.data) * p1)
        l2 = round(len(self.data) * p2)

        return self.data[0:l1], self.label[0:l1], \
               self.data[l1 + 1:l2], self.label[l1 + 1:l2], \
               self.data[l2 + 1:], self.label[l2 + 1:]

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
