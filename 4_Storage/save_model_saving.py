#save using save_model
import tensorflow as tf
import os
print(os.getcwd())
# define the tensorflow network and do some trains
x = tf.placeholder("float", name="x")
w = tf.Variable(2.0, name="w")
b = tf.Variable(0.0, name="bias")

h = tf.multiply(x, w)
y = tf.add(h, b, name="y")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# save the model
export_path =  './savedmodel'
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

'''tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_y = tf.saved_model.utils.build_tensor_info(y)'''

prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'x_input': tf.saved_model.utils.build_tensor_info(x)},#tensor_info_x
      outputs={'y_output': tf.saved_model.utils.build_tensor_info(y)},#tensor_info_y
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder.add_meta_graph_and_variables(
  sess, [tf.saved_model.tag_constants.SERVING],
  signature_def_map={
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          prediction_signature
  },
  )
builder.save()