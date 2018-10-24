## Inception V3

2015年12月，他们发布了一个新版本的GoogLeNet([Inception V3](http://arxiv.org/abs/1512.00567))模块和相应的架构，并且更好地解释了原来的GoogLeNet架构，GoogLeNet原始思想：

* 通过构建平衡深度和宽度的网络，最大化网络的信息流。在进入pooling层之前增加feature maps
* 当网络层数深度增加时，特征的数量或层的宽度也相对应地增加
* 在每一层使用宽度增加以增加下一层之前的特征的组合
* 只使用3x3卷积

因此最后的模型就变成这样了：

![img](https://chenzomi12.github.io/2016/12/13/CNN-Architectures/inceptionv3.jpg)

网络架构最后还是跟GoogleNet一样使用pooling层+softmax层作为最后的分类器。