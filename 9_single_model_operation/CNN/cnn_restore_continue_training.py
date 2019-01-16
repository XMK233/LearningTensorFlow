from cnn_settings import *

def restore_and_retrain_model1(input_checkpoint,
                               train_data, train_label,
                               test_data, test_label):

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)
    # graph = tf.get_default_graph()  # 获得默认的图
    # input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        graph = tf.get_default_graph()
        cost = graph.get_tensor_by_name("cost:0")
        x = graph.get_tensor_by_name("x-input:0")
        y = graph.get_tensor_by_name("y-input:0")
        prediction = graph.get_tensor_by_name("predict_output:0")
        train_op = graph.get_operation_by_name("train")
        hm_epochs = 2
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(len(train_data) / BATCH_SIZE)):
                epoch_x, epoch_y = get_Batch1(train_data, train_label,
                                              BATCH_SIZE)
                _, c = sess.run([train_op, cost], feed_dict={x:
                                                                 epoch_x,
                                                             y:
                                                                 epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_data, y: test_label}))

        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME + "_retrained"))
        saver.export_meta_graph(os.path.join(MODEL_SAVE_PATH, MODEL_NAME + "_restrained") + ".json", as_text=True)

        # visualize the graph
        writer = tf.summary.FileWriter(os.path.join(SUMMARY_PATH, "retrained"),
                               tf.get_default_graph())
        writer.close()

        confusion_matrix = tf.contrib.metrics.confusion_matrix(
            tf.argmax(y, 1),
            tf.argmax(prediction, 1)
        )
        cm = confusion_matrix.eval({x: test_data,
                                 y: test_label}
                                   )
        print(cm)
        return cm

tnd1, tnl1, tnd2, tnl2, tstd, tstl = Dataset().tri_split(0.45, 0.9)
cm_retrain = restore_and_retrain_model1("CNN_model/cnn_model-1", tnd2, tnl2, tstd, tstl)
performance_metrics(cm_retrain)