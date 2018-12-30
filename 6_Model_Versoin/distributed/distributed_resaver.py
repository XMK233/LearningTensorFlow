import tensorflow as tf
import os, shutil

#https://github.com/tensorflow/tensorflow/issues/4436
#https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/my-zswFMhgQ
# https://github.com/tensorflow/tensorflow/issues/6081

'''def freeze_graph(graph_dir, data_dir, output_graph, output_node_names):

    saver = tf.train.import_meta_graph(graph_dir, clear_devices=True)

    # graph = tf.get_default_graph()  # 获得默认的图
    # input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, data_dir)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # ==:input_graph_def
            output_node_names=output_node_names.split(","),
            #variable_names_blacklist= ["import/Variable:0"]
        )  # 如果有多个输出节点，以逗号隔开

        # 下面这两句是为了保存和序列化输出
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        # 如果要输出可读模型，就放开这个注释。
        # tf.train.write_graph(output_graph_def, './', output_graph + "-textual", as_text=True)
        print("%d ops in the final graph." % len(output_graph_def.node))'''

'''shutil.copy("distributed0/model_storage/model.ckpt-10002.data-00000-of-00001", "model.ckpt-10002.data-00000-of-00001")
shutil.copy("distributed1/model_storage/model.ckpt-10002.meta", "model.ckpt-10002.meta")
shutil.copy("distributed1/model_storage/checkpoint", "checkpoint")'''
#module_file = tf.train.latest_checkpoint('distributed1/model_storage/')
'''freeze_graph(graph_dir= "distributed1/log_sync/model.ckpt-10000.meta",
             data_dir = "distributed0/log_sync/model.ckpt-10000.data-00000-of-00001",
             output_graph= "distributed_model.pb",
             output_node_names= "train")'''





'''MODEL_SAVE_PATH = "model_storage"

FLAGS = tf.app.flags.FLAGS

#
#
tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
#
tf.app.flags.DEFINE_string(
    'ps_hosts', ' scale05.eecs.yorku.ca:9994,scale05.eecs.yorku.ca:9995',
    'Comma-separated list of hostname:port for the parameter server jobs. e.g. "tf-ps0:2222,tf-ps1:1111" ')
#
tf.app.flags.DEFINE_string(
    'worker_hosts', ' scale05.eecs.yorku.ca:9996,scale05.eecs.yorku.ca:9997',
    'Comma-separated list of hostname:port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:1111" ')
#
#
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')


ps_hosts = FLAGS.ps_hosts.split(',')
worker_hosts = FLAGS.worker_hosts.split(',')
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_id)

if FLAGS.job_name == 'ps':
    with tf.device("/cpu:0"):
        server.join()

is_chief = (FLAGS.task_id == 0)

device_setter = tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_id,
        cluster=cluster)

with tf.device(device_setter):
    hooks = [tf.train.StopAtStepHook(last_step=20000)]
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False)
    saver = tf.train.Saver()
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           checkpoint_dir=MODEL_SAVE_PATH,
                                           hooks=hooks,
                                           save_checkpoint_steps=1000,
                                           save_summaries_secs=60,
                                           config=sess_config) as mon_sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(mon_sess, ckpt.model_checkpoint_path)
            if is_chief:
                output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
                    sess=mon_sess,
                    input_graph_def=mon_sess.graph_def,  # ==:input_graph_def
                    output_node_names=asdf.split(","),
                    variable_names_blacklist=["import/Variable:0"]
                )

                with tf.gfile.GFile("frozen.pb", "wb") as f:
                    f.write(output_graph_def.SerializeToString())

with tf.train.MonitoredTrainingSession(
                                       checkpoint_dir=MODEL_SAVE_PATH,
                                               save_checkpoint_steps=1000,
                                               save_summaries_secs= 60,
                                               config=sess_config) as mon_sess:'''