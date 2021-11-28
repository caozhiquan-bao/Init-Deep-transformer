项目主要内容：

1）实现Transformer的程序编写，并在Iwslt14数据集上进行试验，但是自己完成的Transformer效果由于缺少很多优化策略，效果很差，实现代码在MyTransformer文件夹中。

2）基于Fairseq工具，从深层神经网络模型结构、参数初始化策略两个方面进行了实验。使用和实现DLCL网络模型、T-Fixup初始化方法、DS-Init初始化方法，并在DLCL上进行初始化方法T-Fixup和DS-Init的融合实验，包括浅层和深层实验。最终实现的融合模型提升了DLCL的性能。除MyTransformer文件夹以外的文件，主要贡献为MODEL/models/目录下的模型文件(以fixup开头的模型文件)。

更多细节请看项目报告。
