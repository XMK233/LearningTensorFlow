diff.py用来做pb模型的diff的。

但是发现，tensor_content数据中间有双引号，会导致json读取失败。
目前找不到如何把一个字符串按照八位数来读取。认怂吧青年。