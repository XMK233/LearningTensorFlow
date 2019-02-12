#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
from rnn_settings import *
import os
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

def recurrent_neural_network(x):
    with tf.name_scope("weight_and_bias"):
        layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    with tf.name_scope("lstm_cells"):
        lstm_cell = rnn_cell.LSTMCell(rnn_size,state_is_tuple=True)

    with tf.name_scope("output"):
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        #output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
        output = tf.add(
            tf.matmul(outputs[-1], layer['weights']),
            layer['biases'],
            name="predict_output"
        )

    return output

def train_neural_network(train_data, train_label, test_data, test_label):
    with tf.name_scope("input_things"):
        x = tf.placeholder('float', [None, n_chunks, chunk_size], name= "x-input")
        y = tf.placeholder('float', name= "y-input")
    prediction = recurrent_neural_network(x)
    with tf.name_scope("loss_functions"):
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels= y,
                                                       logits= prediction
                                                       ),
            name= "cost"
        )

    with tf.name_scope("train_steps"):
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        with tf.control_dependencies([optimizer]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver(max_to_keep=20)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            hm_epochs = 20
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(int(len(train_data) / batch_size)):
                    epoch_x, epoch_y = get_Batch1(train_data,
                                                  train_label,
                                                  BATCH_SIZE)
                    epoch_x = epoch_x.values[:, :]
                    epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                    _, c = sess.run([train_op, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=epoch)
                saver.export_meta_graph(os.path.join(MODEL_SAVE_PATH, MODEL_NAME) + "-%d" % (epoch) + ".json",
                                        as_text=True)

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',
                  accuracy.eval({
                      x: test_data.values[:, :].reshape((-1, n_chunks, chunk_size)),
                      y: test_label
                  }))

            writer = tf.summary.FileWriter(os.path.join(SUMMARY_PATH, "original"),
                                           tf.get_default_graph())
            writer.close()
            # confusion_matrix,用来计算model performance
            confusion_matrix = tf.contrib.metrics.confusion_matrix(
                tf.argmax(y, 1),
                tf.argmax(prediction, 1)
            )
            cm = confusion_matrix.eval({x: test_data.values[:, :].reshape((-1, n_chunks, chunk_size)),
                                        y: test_label}
                                       )
            print(cm)
            return cm

if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)
print(os.getcwd())

tnd1, tnl1, tnd2, tnl2, tstd, tstl = Dataset().tri_split(0.45, 0.9)

cm_origin = train_neural_network(tnd1, tnl1, tstd, tstl)
performance_metrics(cm_origin)