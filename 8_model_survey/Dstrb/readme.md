## 关于restore 用MonitoredTrainingSession保存的模型
* 其实是可以用普通saver来读取的。只要ckpt四件套放在同一个目录下。代码可见
restore_using_saver.py

## 关于模型共享
有这样一个scenario：coders is the people that has the ability to train
and deploy the model.
### model saving and restore
#### single machine: 
* in single machine, they can just save the model in ckpt or 
any other kind of format and use corresponding methods to restore it. 
For example, coder A and B can train and save model, restore model, continue 
train the model in ckpt format using tf.train.Saver. 
#### distributed: 
* coder A and B can train model on 3 machines. .data saved on PS node, 
.index, "checkpoint" and .meta which is graph is saved on chief worker node. 
They can move .data, .index, "checkpoint" and .meta files together and use 
tf.MonitoredTrainingSession to continue training the model. 
* This scenario is a very rudimentary scenario. All the engaging people should 
know how to save and use TF model. 
### model deploying
#### single machine
* the typical way to save the servable model is to use tf.saved_model. The model will 
be saved into a .pb format. 
* I can train a model and save the servable model on my machine. Each time when I 
generate a new model, I will set a version of it. Then I start a TF serving container. 
The local directory is loaded into container's directory. When I generate a new version 
of my model, the containers will find the latest version of the model. Then I can send 
some HTTP post request to send some data to model and get a prediction result return. 
* But this kind of method only share the "prediction" part of a model, no continue training
or some other things. That is being said, the prediciton part can be very easy for 
people to get a result. Even he or she doesn't know everything of TF model.
#### distributed 
(可以在分布式情景下训练一个回归模型，然后将所有的保存文件放在一起，
再将其转换为.pb模型进行serving。如果可以的话，那么这一条路就是一条
可行的“分布式训练+部署”的流程)
