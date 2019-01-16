import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import pandas as pd

MODEL_SAVE_PATH = "CNN_Model/"
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