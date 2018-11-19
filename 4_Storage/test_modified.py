#save the model in save_model method
#changed the layout. Essentially not different from test.py

import tensorflow as tf
import os

def save_model(sess, signature_def_map, output_dir):
  """Saves the model with given signature def map."""
  builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map=signature_def_map)
  builder.save()


def build_signature_def_from_tensors(inputs, outputs, method_name):
  """Builds signature def with inputs, outputs, and method_name."""
  return tf.saved_model.signature_def_utils.build_signature_def(
      inputs={
          key: tf.saved_model.utils.build_tensor_info(tensor)
          for key, tensor in inputs.items()
      },
      outputs={
          key: tf.saved_model.utils.build_tensor_info(tensor)
          for key, tensor in outputs.items()
      },
      method_name=method_name)

x = tf.placeholder("float", name="x")
w = tf.Variable(2.0, name="w")
b = tf.Variable(0.0, name="bias")

h = tf.multiply(x, w)
y = tf.add(h, b, name="y")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# save the model
export_path =  './linear_model/00000123'

signature_def_map = {
        "letsgo":#tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            build_signature_def_from_tensors({"x_input": x},
                                             {"y_output": y},
                                             tf.saved_model.signature_constants.PREDICT_METHOD_NAME),
    }

save_model(sess, signature_def_map, export_path)