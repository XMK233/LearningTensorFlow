## 关于restore 用MonitoredTrainingSession保存的模型
* 其实是可以用普通saver来读取的。只要ckpt四件套放在同一个目录下。代码可见
restore_using_saver.py

## model shared 
### for further training
#### single machine
Coders can simply save checkpoint files by using the basic tf.train.Saver. Then 
coders can restore the model's important variables and tensors then do the 
further training with either the new data or not. 

Whoever have the access to saved
 ckpt files can further training the models. 
#### distributed
Distributed models are typically trained under some supervision, like using
tf.train.Supervisor or tf.train.MonitoredTrainingSession. 
Take MonitoredTrainingSession as an example. 
The training graph and parameters will be stored on different nodes. If coders
want to continue training, they can change the total steps of training, or data 
using part of the code and then 
run the code in the same way again, the MTS will automatically restore the latest version 
checkpoint and continue training. But as far as I know, you must move the 4 ckpt files
to the save directory and then you can start the continue training like said before.

Whoever has the access to saved ckpt files and the permission to move the files and 
change the code can further train the models. 
### for prediction
#### Not Serving
* single machine: 
The most easy way to use the trained model to prediction is restoring the model, 
restoring and running the prediction operations. That can be achieved by using 
tf.train.Saver methods. Whoever has the access to ckpt files can restore the 
model and do the prediction. 
* distributed: For example. A model is trained and the ckpts are saved by MTS. 
Then a coder can move the ckpt files together and then restore it by simply using
tf.train.Saver. And then the prediction will act like in a single machine. After 
the model is restored, coders can use tf.save_model to save a servable model. This 
model is ready to be deployed by TF Serving. Once the model is deployed, users can 
send a HTTP POST request with proper body information, and then they can get prediction
result. 
#### TensorFlow Serving
* The model, no matter trained on a single machine or distributedly, can be 
restored as it is trained on a single machine by using tf.train.Saver. 
* TF Serving has a very good feature. If you have deployed a model saved in directory
 "/home/Model/V1" by using container method, and then you train a new version of the model 
 and save the model in "/home/Model/V2". The TF Serving container will automatically use
 the latest V2 model to handle the prediction request.   

----------------------------------------
----------------------------------------
----------------------------------------
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
we can train a model in a distributed manner, and then move the files together. 
We then restore the model and save it as a servable model, like in .pb format.
And then we can serve the model. 
