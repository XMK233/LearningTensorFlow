## 关于RNN的层和cell个数：
https://blog.csdn.net/Jerr__y/article/details/61195257
 讲解了什么是多层循环神经网络。
 
 所谓多层循环神经网络，是这样的：
 一层的话有好多cell，每一个cell就是一个那个···rnn的经典结构。
 然后多层就是多个层串联起来。由此判断，rnn_save里面写的那个代码
 是单层的，并且每一层有128个cell。
 
## rnn_save_with_namespace.py
这个代码是用来生成加了namespace的模型。主要用来研究tensorboard的。 