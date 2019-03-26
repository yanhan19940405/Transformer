# Self_Attention
基于Keras的self-attention复现

其输入：上一层的输出张量。

输出：一个三维张量，维度为（samples，maxlen，attention）,第一个维度含义代表句子数目，maxlen维度代表每一个句子长度，attention维度代表每个句子中每个分词构成的attention编码值个数。

运行截图如下（1000条情感分类数据应用下）：

![图1 code](https://github.com/yanhan19940405/Self_Attention/blob/master/image/11.png)

![图2 code](https://github.com/yanhan19940405/Self_Attention/blob/master/image/12.png)

[详细解读请参阅](https://yanhan19940405.github.io/2019/03/18/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86Attention%E6%9C%BA%E5%88%B6%E7%BB%BC%E8%BF%B0%E4%B8%8E%E5%9F%BA%E4%BA%8Ekeras%E5%A4%8D%E7%8E%B0/)
