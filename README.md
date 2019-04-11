# Self_Attention
基于Keras的self-attention复现

其输入：上一层的输出张量。

输出：一个三维张量，维度为（samples，maxlen，attention）,第一个维度含义代表句子数目，maxlen维度代表每一个句子长度，attention维度代表每个句子中每个分词构成的attention编码值个数。

运行截图如下（1000条情感分类数据应用下）：

![图1 code](https://github.com/yanhan19940405/Self_Attention/blob/master/image/11.png)

![图2 code](https://github.com/yanhan19940405/Self_Attention/blob/master/image/12.png)

在8万条情感分析数据场景下，网络结构为cnn+ATT。学习率设置为0.01时，自身复现的att训练过程达成收敛，分类结果好于单独CNN网络，结果展示如下：

![图4 code](https://github.com/yanhan19940405/Transformer/blob/master/image/fin.png)

训练收敛情况展示如下：

![图5 code](https://github.com/yanhan19940405/Transformer/blob/master/image/att.png)

[Attention详细解读请参阅](https://yanhan19940405.github.io/2019/03/18/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86Attention%E6%9C%BA%E5%88%B6%E7%BB%BC%E8%BF%B0%E4%B8%8E%E5%9F%BA%E4%BA%8Ekeras%E5%A4%8D%E7%8E%B0/)

# Transformer模型复现（3层Transformer结构)

模型结构如下：
![图3 Transformer](https://github.com/yanhan19940405/Transformer/blob/master/image/model.png)

原理：1.分类场景下，通过Transformer Encoder作为动态词向量提取器+softmax作为文本分类baseline，进行微调即可。

     2.序列生成场景,将transformer Encoder作为序列隐藏状态层（state）生成器即可，decoder结构没有固定，可以随自身需求定义。
      
参数说明：在Transformer类中，creat_model方法为模型创建方法。其中maxlen0表示文本长度，wordindex代表词典索引数目，matrix代表词向量

注意：此处已经去掉常用数据预处理部分
