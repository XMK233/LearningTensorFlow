import tensorflow as tf
import os

print(os.getcwd())

if not os.path.exists("Easy_model"):
    os.mkdir("Easy_model")

v1 = tf.Variable(tf.random_normal([1], stddev=1, seed=1), name= "v1")
v2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1), name= 'v2')
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "Easy_model/model.ckpt")
    saver.export_meta_graph("Easy_model/model.ckpt.meta.json", as_text= True)