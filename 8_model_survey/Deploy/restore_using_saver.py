import tensorflow as tf

saver = tf.train.import_meta_graph("model.ckpt-10001.meta")
with tf.Session() as sess:
    saver.restore(sess, "model.ckpt-10001")