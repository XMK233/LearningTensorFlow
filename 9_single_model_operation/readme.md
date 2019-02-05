## 在diff文件里面
发现一个问题：rnn里面有一类叫“beta%d_power”的tensor。这个tensor得到之后是一个
numpy.float32类型的数字而非其他tensor一样是一个ndarray。对它们使用.shape之后得到
的是一个空的list。

## 关于免费的modeldepot/percept镜像
TEST.py里面有一个方法叫transform_image_to_base46，
用来转换图片为base64编码，这样的编码可以用来放进modeldepot/percept里面训练。