## ResNet

2015年12月[ResNet](https://arxiv.org/pdf/1512.03385v1.pdf)发表了，时间上大概与Inception v3网络一起发表的。其中ResNet的一个重要的思想是：输出的是两个连续的卷积层，并且输入时绕到下一层去。这句话不太好理解可以看下图。

![img](https://chenzomi12.github.io/2016/12/13/CNN-Architectures/resnetb.jpg)

但在这里，他们绕过两层，并且大规模地在网络中应用这中模型。在2层之后绕过是一个关键，因为绕过单层的话实践上表明并没有太多的帮助，然而绕过2层可以看做是在网络中的一个小分类器！看上去好像没什么感觉，但这是很致命的一种架构，因为通过这种架构最后实现了神经网络超过1000层。傻了吧，之前我们使用LeNet只是5层，AlexNet也最多不过7层。

![img](https://chenzomi12.github.io/2016/12/13/CNN-Architectures/resnetbottleneck.jpg)

该层首先使用1x1卷积然后输出原来特征数的1/4，然后使用3×3的卷积核，然后再次使用1x1的卷积核但是这次输出的特征数为原来输入的大小。如果原来输入的是256个特征，输出的也是256个特征，但是这样就像Bottleneck Layer那样说的大量地减少了计算量，但是却保留了丰富的高维特征信息。

ResNet一开始的时候是使用一个7x7大小的卷积核，然后跟一个pooling层。当然啦，最后的分类器跟GoogleNet一样是一个pooling层加上一个softmax作为分类器。下图左边是VGG19拥有190万个参数，右图是34层的ResNet只需要36万个参数：

[![img](https://chenzomi12.github.io/2016/12/13/CNN-Architectures/ResNet.jpg)](https://chenzomi12.github.io/2016/12/13/CNN-Architectures/ResNet.jpg)

**ResNet网络特征**

* ResNet可以被看作并行和串行多个模块的结合
* ResNet上部分的输入和输出一样，所以看上去有点像RNN，因此可以看做是一个更好的生物神经网络的模型