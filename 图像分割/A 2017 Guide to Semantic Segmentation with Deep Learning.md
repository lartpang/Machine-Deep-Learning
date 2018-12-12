# A 2017 Guide to Semantic Segmentation with Deep Learning

> 关键点:
>
> * 主要数据库: VOC, COCO
> * 像素级分类
> * 几个语义分割的典型网络
> * 语义风格的方法

> 原文: http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review
>
> 翻译: http://www.chenzhaobin.com/notes/semantic-segmentation-deep-learning-review

## What exactly is semantic segmentation（究竟什么是语义分割）?

语义分割就是“在像素级别去理解一副图像”,比如我们想知道图像中每个像素具体属于哪个目标物体。

![biker](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21.jpg)

![biker](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21_class.png)

除了要**识别出车子和人**以外，我们还要**获取每个目标的具体轮廓**。

因此，和分类任务不同, 语义分割要求我们可以从模型里面获取到像素级别的识别信息。

VOC2012 和MSCOCO 是语义分割方面最重要的数据库.

## What are the different approaches(有哪些方法)?

在深度学习还没有在机器视觉领域独占鳌头以前，人们通常采用[TextonForest](http://mi.eng.cam.ac.uk/~cipolla/publications/inproceedings/2008-CVPR-semantic-texton-forests.pdf)或者 [基于随机森林的分类器](http://www.cse.chalmers.se/edu/year/2011/course/TDA361/Advanced%20Computer%20Graphics/BodyPartRecognition.pdf)  来处理语义分割任务。和图像分类任务一样,卷积神经网络也在处理分割任务方面取得了巨大的成功。

比较原始的深度学习处理方式是基于像素块来分类的，也就是基于某个像素周围像素组成的像素块(patch)来判断某个像素的类别。这样做的一个**主要原因是当时采用的分类网络通常都有全连接层**，要求使用固定尺寸的图像。

在2014年，来自伯克利（美国加利福尼亚州西部城市）的 Long et al 提出了全卷积神经网络(FCN）,在没有采用任何全连接层的情况下完成了像素级别的预测。FCN允许输入任意尺寸的图片，并且比基于像素块的分类方法要快很多。几乎所有后续的语义分割方法都采用了FCN的这种范式。

用CNN做分割任务，除了全连接层的影响以外，另外一个主要的问题是池化层。*池化层可以在丢失一部分位置信息的情况下增大感受野并能够获取更广泛的上下文信息*。但是，*语义分割是每个像素和类别的映射关系，所以需要保留位置信息的*。下面论述两种不同的改进架构可以解决这个问题。

**第一种 编码器-解码器(encoder-decoder) 架构**, 编码器通过池化层逐渐的减少空间维度，然后解码器逐逐渐的恢复物体的详细信息和空间维度。通常在*编码器和解码器之间有转换关系*，方便解码器更好的恢复物体的详细信息。U-Net就是这类方法的一种流行的架构。

![U-Net architecture](http://blog.qure.ai/assets/images/segmentation-review/unet.png)

**另外一种架构采用扩张卷积/空洞卷积，并且没有池化层，**不通过pooling也能有较大的感受野看到更多的信息

![Dilated/atrous convolutions](http://blog.qure.ai/assets/images/segmentation-review/dilated_conv.png)

当空洞的rate=1的时候，就是常规的卷积网络。【知乎：[如何理解空洞卷积](https://www.zhihu.com/question/54149221)】

**条件随机场(CRF)后处理**通常被用来改进分割。CRFs是*基于周边像素来‘平滑’分割的图模型*。它们的工作原理是<u>相似的像素趋向于分为同一个类别</u>。CRFs可以提高算法1-2%。

接下来,我将总结一些语义分割方面从FCN开始演进的代表性论文。所有的这些论文都是将VOC2012评估服务器作为基准的

## Summaries(论文总结)

> Following papers are summarized (in chronological order):

以下是我按照时间顺序总结的论文列表：

1. [FCN](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#fcn)
2. [SegNet](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#segnet)
3. [Dilated Convolutions](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#dilation)
4. [DeepLab (v1 & v2)](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#deeplab)
5. [RefineNet](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#refinenet)
6. [PSPNet](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#pspnet)
7. [Large Kernel Matters](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#large-kernel)
8. [DeepLab v3](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#deeplabv3)

对于接下来的每片文献，我列举并阐释了它们的主要贡献, 并且进行适当的解释, 也展示了他们的对应的结果(mean IOU 在VOC2012测试集上)

### FCN

- Fully Convolutional Networks for Semantic Segmentation（用于语义分割的全卷积神经网络）
- Submitted on 14 Nov 2014(提交于2014-11-14)
- [Arxiv Link](https://arxiv.org/abs/1411.4038)（论文链接）

**主要贡献**：

- 普及了端到端的神经网络在语义分割上的应用
- 重用imagenet预训练网络，使其可以用于分割任务
- 采用反卷积层来上采样
- 介绍跳跃层来改进上采样的失真问题

**解释**：

论文主要的发现是把分类网络中的全连接层看成了卷积核覆盖整个区域的卷积层。这就相当于通过交错的像素块来评估原始的分类网络，但是由于在像素块之间的计算是共享的，所以相比来说效率更高。尽管这不是仅有的一篇发现此观点的论文(比如其他的论文[overfeat](https://arxiv.org/abs/1312.6229)）,但是这篇论文提高了了VOC2012最优的实验结果.

![FCN architecture](http://blog.qure.ai/assets/images/segmentation-review/FCN%20-%20illustration.png)

对预训练的imagenet神经网络(比如VGG)的全连接层进行卷积化以后，由于池化的原因，特征映射图仍然需要利用反卷积层来进行上采样操作。*反卷积层可以学习插值的参数，而不是采用简单的双线性插值*。反卷积层又被称为上卷积层、全卷积层、transposed convolution 或者 fractionally-strided convolution。

然而，由于在池化过程中丢失了一部分信息，导致上采样(即使利用了反卷积层)会产生较为粗糙的分割映射。因此，论文中又提到了在较高分辨率的特征映射图中抽取的跳跃层进行优化。

*Benchmarks (VOC2012)*:

ScoreCommentSource

62.2-[leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=6103#KEY_FCN-8s)

67.2More momentum. Not described in paper[leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=6103#KEY_FCN-8s-heavy)

*My Comments*:

- This was an important contribution but state of the art has improved a lot by now though.

### SegNet

- SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
- Submitted on 2 Nov 2015
- [Arxiv Link](https://arxiv.org/abs/1511.00561)

*Key Contributions*:

- 对最大池化进行了索引记录, 在解码的时候提升了分割的分辨率.

*Explanation*:

FCN, despite upconvolutional layers and a few shortcut connections produces coarse segmentation maps. Therefore, more shortcut connections are introduced. 

**但是，不像FCN那样复制编码器功能，而是复制maxpooling的索引**。这使得SegNet比FCN具有更高的内存效率。

![SegNet Architecture](http://blog.qure.ai/assets/images/segmentation-review/segnet_architecture.png)

Segnet Architecture. [Source](https://arxiv.org/abs/1511.00561).

*Benchmarks (VOC2012)*:

ScoreCommentSource

59.9-[leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_SegNet)

*My comments*:

- FCN and SegNet are one of the first encoder-decoder architectures.
- 基准测试比较差, 所以不怎么用了

### Dilated Convolutions

- Multi-Scale Context Aggregation by Dilated Convolutions
- Submitted on 23 Nov 2015
- [Arxiv Link](https://arxiv.org/abs/1511.07122)

*Key Contributions*:

- Use dilated convolutions, a convolutional layer for dense predictions.
- 提出“上下文模块”，它使用扩张卷积进行多尺度聚合。

*Explanation*:

Pooling helps in classification networks because receptive field increases. But this is not the best thing to do for segmentation because pooling decreases the resolution. Therefore, authors use *dilated convolution* layer which works like this:

pooling有助于分类网络，因为增加了感受野。但这不是分割的最佳选择，因为会降低分辨率。因此，作者使用扩张卷积层，其工作方式如下：

![Dilated/Atrous Convolutions](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/dilation.gif)

扩张卷积层（在DeepLab中也称为迂回卷积）允许感受野的指数增加而不减小空间维度。来自预训练分类网络（此处为VGG）的最后两个pooling layer被移除，随后的卷积层被扩张卷积替换。特别地，pool-3和pool-4之间的卷积具有膨胀系数2，并且pool-4之后的卷积具有膨胀系数4. 利用该模块（本文中称为*frontend module*），获得密集预测而不增加任何参数数量。

模块（本文中称为*context module*）用*frontend module*的输出作为输入分开训练。该模块是不同扩张的扩张卷积的级联，因此聚合了多尺度上下文并且改善了来自前端的预测。

*Benchmarks (VOC2012)*:

ScoreCommentSource

71.3frontendreported in the paper

73.5frontend + contextreported in the paper

74.7frontend + context + CRFreported in the paper

75.3frontend + context + CRF-RNNreported in the paper

*My comments*:

- 请注意，预测的分割图的大小是图像的1/8。几乎所有方法都是这种情况。对它们进行插值以获得最终的分割图。

### DeepLab (v1 & v2)

- **v1** : Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
- Submitted on 22 Dec 2014
- [Arxiv Link](https://arxiv.org/abs/1412.7062)
- **v2** : DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
- Submitted on 2 Jun 2016
- [Arxiv Link](https://arxiv.org/abs/1606.00915)

*Key Contributions*:

- 使用扩张卷积
- 提出了扩张空间金字塔池（*atrous spatial pyramid pooling (ASPP)*）
- Use Fully connected CRF

*Explanation*:

**在不增加参数数量的情况下，Atrous / Dilated卷积增加了视野。** 

通过将多个重新缩放版本的原始图像传递到并行CNN分支（图像金字塔）和/或通过使用具有不同采样率（ASPP）的多个并行的迂回卷积层来实现多尺度处理。

结构化预测由完全连接的CRF完成。CRF作为后处理步骤单独训练/调整。

![DeepLab2 Pipeline](http://blog.qure.ai/assets/images/segmentation-review/deeplabv2.png)

DeepLab2 Pipeline. [Source](https://arxiv.org/abs/1606.00915).

*Benchmarks (VOC2012)*:

ScoreCommentSource

79.7ResNet-101 + atrous Convolutions + ASPP + CRF[leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=6103#KEY_DeepLabv2-CRF)

### RefineNet

- RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
- Submitted on 20 Nov 2016
- [Arxiv Link](https://arxiv.org/abs/1611.06612)

*Key Contributions*:

- 具有深思熟虑的解码器模块的编码器 - 解码器架构(Encoder-Decoder architecture with well thought-out decoder blocks)
- 所有组件都遵循残差连接设计

*Explanation*:

使用扩张卷积的方法并非没有缺点。扩张卷积在计算上是昂贵的并且占用大量内存，因为它们必须应用于大量高分辨率特征映射。这妨碍了高分辨率预测的计算。例如，DeepLab的预测是原始输入大小的1/8。

因此，本文提出使用编码器 - 解码器架构。编码器部分是ResNet-101块, 解码器具有RefineNet块，这样的结构, 可以连接编码器的高分辨率功能和先前RefineNet模块的低分辨率功能。

![RefineNet Architecture](http://blog.qure.ai/assets/images/segmentation-review/refinenet%20-%20architecture.png)

RefineNet Architecture. [Source](https://arxiv.org/abs/1611.06612).

每个RefineNet模块都有一个组件，通过对低分辨率功能进行上采样来融合多分辨率功能，以及一个基于重复的5 x 5的步长为1的pooling层捕获上下文的组件。这些组件中的每一个都采用遵循恒等匹配的思维模式的残差连接设计。

![RefineNet Block](http://blog.qure.ai/assets/images/segmentation-review/refinenet%20-%20block.png)
RefineNet Block. [Source](https://arxiv.org/abs/1611.06612).

*Benchmarks (VOC2012)*:

ScoreCommentSource

84.2Uses CRF, Multiscale inputs, COCO pretraining[leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_Multipath-RefineNet)

### PSPNet

- Pyramid Scene Parsing Network
- Submitted on 4 Dec 2016
- [Arxiv Link](https://arxiv.org/abs/1612.01105)

*Key Contributions*:

- 提出金字塔池化来聚合上下文
- 使用辅助损失( auxiliary loss)

*Explanation*:

全局场景类别很重要，因为它提供了分割类别的分布的线索。金字塔池化模块通过应用大核池化层来捕获此信息。

扩张卷积用于扩张卷积的论文来修改Resnet，并且向其添加一个金字塔池化模块。该模块将来自ResNet的特征映射与并行池化层的上采样输出连接，其中并行池化层的核覆盖整个，一半和一小部分的图像。

在ResNet的第四阶段之后（即输入到金字塔池模块），应用除了主分支损失之外的一个辅助损失。这个想法在其他地方也被称为中间监督。

![PSPNet Architecture](http://blog.qure.ai/assets/images/segmentation-review/pspnet.png)
PSPNet Architecture. [Source](https://arxiv.org/abs/1612.01105).

*Benchmarks (VOC2012)*:

ScoreCommentSource

85.4MSCOCO pretraining, multi scale input, no CRF[leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_PSPNet)

82.6no MSCOCO pretraining, multi scale input, no CRFreported in the paper 

### Large Kernel Matters

- Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
- Submitted on 8 Mar 2017
- [Arxiv Link](https://arxiv.org/abs/1703.02719)

*Key Contributions*:

- 提出一种具有非常大的核的卷积的编码器 - 解码器架构

*Explanation*:

语义分割需要分割对象的分割和分类。由于完全连接的层不能存在于分割架构中，因此采用具有非常大的核的卷积。

采用大核的另一个原因是虽然像ResNet这样的更深层次的网络具有非常大的感受野，但研究表明网络倾向于从更小的区域（有效的接收域）收集信息。

较大的核计算量很大并且具有许多参数。因此，k×k卷积用1×k + k×1和k×1和1×k卷积的和来近似。该模块在本文中称为全局卷积网络（*Global Convolutional Network* (GCN) ）。

继续看架构，ResNet（没有任何扩张卷积）形成架构的编码器部分，而GCN和反卷积形成解码器。还使用称为边界细化（ *Boundary Refinement* (BR) ）的简单残差块。

![GCN Architecture](http://blog.qure.ai/assets/images/segmentation-review/large_kernel_matter.png)
GCN Architecture. [Source](https://arxiv.org/abs/1703.02719).

*Benchmarks (VOC2012)*:

ScoreCommentSource

82.2-reported in the paper

83.6Improved training, not described in the paper[leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_Large_Kernel_Matters)

### DeepLab v3

- Rethinking Atrous Convolution for Semantic Image Segmentation
- Submitted on 17 Jun 2017
- [Arxiv Link](https://arxiv.org/abs/1706.05587)

*Key Contributions*:

- *atrous spatial pyramid pooling (ASPP)*改进
- 在级联中采用空洞卷积的模块

*Explanation*:

ResNet模型被修改为使用扩张卷积，如DeepLabv2和扩张卷积。改进的ASPP涉及图像级特征的连接，1x1卷积和3个不同速率的3x3 atrous卷积。在每个并行卷积层之后使用批量归一化。

级联模块是一个resnet块，除了组件卷积层以不同的rate的空洞卷积组成。此模块类似于扩张卷积论文中使用的上下文模块，但这直接应用于中间特征映射而不是信念映射（**信念映射是具有等于类数的通道的最终CNN特征映射**）。

两个提出的模型都是独立评估的，并且尝试将两者结合起来, 但是并没有改善性能。它们两个在验证集熵上表现非常相似，ASPP表现稍好一些。不使用CRF。

这两个模型都优于DeepLabv2的最佳模型。作者指出，改进来自BN和更好的编码多规模上下文的方法。

![DeepLabv3 ASPP](http://blog.qure.ai/assets/images/segmentation-review/deeplabv3.png)
DeepLabv3 ASPP (used for submission). [Source](https://arxiv.org/abs/1706.05587).

*Benchmarks (VOC2012)*:

ScoreCommentSource

85.7used ASPP (no cascaded modules)[leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_DeepLabv3)