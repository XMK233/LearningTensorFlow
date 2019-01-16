from rnn_settings import *
import os

def restore_model1(input_checkpoint,
                   test_data, test_label):
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)
    # graph = tf.get_default_graph()  # 获得默认的图
    # input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x-input:0")
        y = graph.get_tensor_by_name("y-input:0")
        prediction = graph.get_tensor_by_name("predict_output:0")

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({
            x: test_data.values[:, :].reshape((-1, n_chunks, chunk_size)),
            y: test_label
        }))

        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME + "_restored"))
        saver.export_meta_graph(os.path.join(MODEL_SAVE_PATH, MODEL_NAME + "_restored") + ".json", as_text=True)

        confusion_matrix = tf.contrib.metrics.confusion_matrix(
            tf.argmax(y, 1),
            tf.argmax(prediction, 1)
        )
        cm = confusion_matrix.eval({x: test_data.values[:, :].reshape((-1, n_chunks, chunk_size)),
                                    y: test_label}
                                   )
        print(cm)
        return cm

tnd1, tnl1, tnd2, tnl2, tstd, tstl = Dataset().tri_split(0.45, 0.9)
cm_restore = restore_model1(os.path.join(MODEL_SAVE_PATH, MODEL_NAME + "-1"), tstd, tstl)
performance_metrics(cm_restore)