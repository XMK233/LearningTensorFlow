import tensorflow as tf

MODEL_SAVE_PATH = "CNN_model"
MODEL_NAME = "cnn_model"
SUMMARY_PATH = "CNN_Logs"

def transform_saved_graph_to_readable(model_path):
    saver = tf.train.import_meta_graph(model_path,
                                       clear_devices=True)
    saver.export_meta_graph(model_path + ".json",
                            as_text=True)


transform_saved_graph_to_readable("%s/%s-%d.meta" %(MODEL_SAVE_PATH, MODEL_NAME, 2))