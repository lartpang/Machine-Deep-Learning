## Inception V3

Christian的团队确实很厉害，2015年2月他们又发表了新的文章关于在googleNet中加入一个[Batch-normalized](http://arxiv.org/abs/1502.03167)层。Batch-normalized层归一化计算图层输出处所有特征图的平均值和标准差，并使用这些值对其响应进行归一化。这对应于“白化”数据非常有效，并且使得所有神经层具有相同范围并且具有零均值的响应。这有助于训练，因为下一层不必学习输入数据中的偏移，并且可以专注于如何最好地组合特征。

2015年12月，他们发布了一个新版本的GoogLeNet([Inception V3](http://arxiv.org/abs/1512.00567))模块和相应的架构，并且更好地解释了原来的GoogLeNet架构，GoogLeNet原始思想：

* 通过构建平衡深度和宽度的网络，最大化网络的信息流。在进入pooling层之前增加feature maps
* 当网络层数深度增加时，特征的数量或层的宽度也相对应地增加
* 在每一层使用宽度增加以增加下一层之前的特征的组合
* 只使用3x3卷积

因此最后的模型就变成这样了：

![img](https://chenzomi12.github.io/2016/12/13/CNN-Architectures/inceptionv3.jpg)

网络架构最后还是跟GoogleNet一样使用pooling层+softmax层作为最后的分类器。