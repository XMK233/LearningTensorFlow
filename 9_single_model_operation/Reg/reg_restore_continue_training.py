from reg_settings import *

model = os.path.join(MODEL_SAVE_PATH, MODEL_NAME + "-1")
saver = tf.train.import_meta_graph(model + '.meta',
                                   clear_devices=True)

with tf.Session() as sess:
    saver.restore(sess, model)  # 恢复图并得到数据
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input:0")
    train_op = graph.get_operation_by_name("training/train")

    # 随机生成若干个点，围绕在y=0.1x+0.3的直线周围
    num_points = 100
    vectors_set = []
    for i in range(num_points):
        x1 = np.random.normal(0.0, 0.55)
        y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        vectors_set.append([x1, y1])
    # 生成一些样本
    x_data1 = [v[0] for v in vectors_set]
    y_data1 = [v[1] for v in vectors_set]

    plt.scatter(x_data1, y_data1, c='r')
    plt.show()

    for step in range(5):
        sess.run(train_op, feed_dict={x: x_data1})

    saver.save(sess, model + "_retrained")
    saver.export_meta_graph(model + "_retrained" + ".json", as_text=True)
    writer = tf.summary.FileWriter(os.path.join(SUMMARY_PATH, "retrained"),
                               tf.get_default_graph())
    writer.close()