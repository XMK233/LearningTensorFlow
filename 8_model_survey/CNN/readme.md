## 关于将ckpt模型导出为可读形式：
使用脚本运行得到的可读计算图，在做diff的时候，发现：图没有发生变化。

变量的diff耗时贼久，如果没有死循环的话，难道就是这么慢了？卧槽。