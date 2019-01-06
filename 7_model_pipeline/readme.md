So....
---

* I want to build the pipeline for model training, saving, restoring. 

* Train a easy model and restore it. Train a more complex one. 

* Train a model on distributed working environment. 

* The jupyter notebook I used: 
scale05.eecs.yorku.ca, 8881, d9559ffb6837ac01591fd79f187ae19c0f01304b8529b1c7

cnn_model.ipynb
---
* To do: accuracy[done], precision[done], recall[done]....

* Function: model save in ckpt file, restore, retrain. 
Data is the csv mnist dataframe from kaggle.com. 
The model is not that easy because the model is CNN. 

* Operations and variables can be given names. When restore the model, 
the name can be used to locate the ops or variables you want. 

* The ffeeding data should be in nparray or pd.Dataframe format. 
Not in variable format. 

* operation.eval() can return the nparray, rather than the Tensor.

distributed
---
* 分布式训练：通过run.async.sh脚本，yml，async.py文件就可以实现一键分布式训练。
* （似乎必须要把最新的模型的data, index文件放到meta的文件夹下才能正常读取。
我去，真麻烦）
* 如果这个模型是以MonitoredTrainingSession做的保存，那么这样
操作就能在原有的ckpt基础上继续训练：
* 第一次训练的时候，有记录一个global_step,
这个记录表明到目前为止总共训练了多少个step。如果要继续训练，就要先把存在ps上的
data, index最新记录移动到worker:0里面，然后改一下代码里面的训练总step数到一个
更大的数字，然后运行和原先一样的命令就可以了。
* 比如：我原本训练了10000 step，结果获得了10000轮的ckpt。然后，我需要
把ps里面的data,index移动到worker0里面，跟meta放到一起。然后改动代码，把训练
总step数量改为20000，然后按照正常的Distributed tensorflow来运行。结果发现，
模型继续训练了，然后保存ckpt文件直到20000的记录为止。
* 目前可以做出这样的判断：monitoredTrainingSession保存的模型可能只有用这种方法
才能读取；而且读取的时候也必须要把data, index, meta放到一起才行。
* test_model.py: 建立了一个完全一样的模型，但却是非分布式的，而且也用
monitoredTrainingSession去读取已经存储的模型，发现可以继续训练。这说明，
分布式训练得到的ckpt不见得就一定得是分布式才能读取回来，只要用存储ckpt相同
的方法去读取就OK。
* 在restore_graph.py中尝试不建立模型，而是直接读取图，然后获取参数，继续训练。
发现图读取得不对，x-input, y-input提示说找不到。这个方法算是失败了吧。



