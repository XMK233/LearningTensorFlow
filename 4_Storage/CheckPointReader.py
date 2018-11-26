import tensorflow as tf

reader = tf.train.NewCheckpointReader("Easy_model/model.ckpt")

global_variables = reader.get_variable_to_shape_map()

for variable_name in global_variables:
    print(variable_name, global_variables[variable_name], reader.get_tensor("v1"))

print("Value for variable v1 is ", reader.get_tensor("v1"))
