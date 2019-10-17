# SAC-Net: Spatial Attenuation Contextfor Salient Object Detection

- [SAC-Net: Spatial Attenuation Contextfor Salient Object Detection](#sac-net-spatial-attenuation-contextfor-salient-object-detection)
  - [主要工作](#主要工作)
  - [主要结构](#主要结构)
  - [实验细节](#实验细节)
  - [一些讨论](#一些讨论)
  - [参考链接](#参考链接)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1570357903246-e30a6804-da28-4d60-b254-cbe38a3c4d4c.png#align=left&display=inline&height=116&name=image.png&originHeight=116&originWidth=682&search=&size=23035&status=done&width=682)

## 主要工作

这是arxiv上的一篇文章, 不知道是不是因为投稿会议被拒了.

总体来看, 想法有些类似于PiCANet, 尝试集成各个方向的不同距离的依赖关系从而实现对于上下文信息的利用. 这里有一点, 就是对于各个方向的远距离特征信息的利用, 这里设置了一个参数, 配合着一个可学习的注意力权重实现了对于不同距离信息利用的自适应计算.

至于架构本身而言, 创新也不是太大, 依旧走的是FPN的路子, 不过这个一般确实不好有新的变化. 虽然挂着attention的名头, 但是看上去也只是个简单的softmax归一化处理后的针对分支的预测权重.

## 主要结构

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1570358217457-415da670-062d-4ce6-b17e-45b119a8c4c9.png#align=left&display=inline&height=348&name=image.png&originHeight=348&originWidth=811&search=&size=89837&status=done&width=811)

总体框图很直接, 主要讲解其中的SAC Module.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1570358233870-eecd9fef-8455-4f3d-971d-3c7ca748739a.png#align=left&display=inline&height=408&name=image.png&originHeight=408&originWidth=820&search=&size=100960&status=done&width=820)

这里单独看图其实大体可以了解其中的构思. 这里对于特征的传播采取了两个重复的阶段的设置. 同时一条并行的支路预测了注意力作为下面的传播支路中的各个分支的权重. 所有的传播分支乘以各自的权重W的结果拼接后送入1x1卷积进一步融合. 对于各个传播分支, 则是各自使用了四个方向的传播机制, 同时可以看出来, 这其中会涉及到一个因子alpha, 具体有什么关系, 接下来进一步解释.

首先, 对于通过1x1卷积后的特征通过n个支路进行处理. 之后对于各个分支中的特征使用四个方向的依次处理从而传播融合特征. 这里给出了up方向的计算过程, 其他三个方向可以进行类比.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1570362180945-23f3b15d-9ea6-4ef3-8a0b-dcb5b11e8cdf.png#align=left&display=inline&height=50&name=image.png&originHeight=50&originWidth=395&search=&size=7475&status=done&width=395)

* fijup表示对于up分支中, (i, j)位置处更新后的特征矢量, 其与rjup有关系
* rijup表示加上了特定权重加权(1-alphak)的来自上一位置(i-1, j)更新后的特征fijup后, 对于位置(i, j)而言的混合特征, 注意, 这里的**fij始终表示原始的输入的特征图上的(i, j)位置上的特征矢量, 四个方向计算使用的都是相同的fij**
* 这里的beta是一个权重, 用来提供一定的非线性(reduce the feature magnitude when it is negative)(使得正负数据的对应的结果有差异, 感觉这里是在模仿Leaky ReLU和PReLU的计算), 试验中这个参数初始化为0.1, 设置为可学习的参数, 对于各个通道而言是不同的(learn the value of β for **each feature channel**)
* 这里的alpha从图中可以看出来, 是一个分数, 对于第k个分支(一共是n个), 其对应的alphak=(n-k)/n, 也就是说, 随着k的增大, (1-alphak)=(k/n)会逐渐增大, 趋于1, 也就是对于过去的信息进行了完全的保留, 实现了超长距离的信息的记忆与传播

这里通过使用对于之前的结果的一个加权的"记忆"实现了特征在不同方向上的传播.(有点类似于滑动平均, 但是这里对于f和fup两类不同的数据的滑动过程, 而非一般的, 针对自身数据序列的滑动窗口)

关于这里的注意力的计算, 使用的如下的softmax计算过程:

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1570363146697-06945f9e-dc18-4abb-978d-dfde866df181.png#align=left&display=inline&height=75&name=image.png&originHeight=136&originWidth=731&search=&size=24111&status=done&width=402)

这里表示的是:

1. 处时输入特征F
1. 通过一个结构为 `Conv3x3-GroupNorm-ReLU-C`  `onv3x3-GroupNorm-ReLU-Conv1x1` 的注意力计算模块Fattention
1. 得到未归一化的注意力输出{A}
1. 通过softmax沿着所有的A的同一位置(i,j)进行归一化，使得对于(i,j)这个位置而言，归一化后的权重集合{W}加和为1

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1570363476689-687d5666-6b4b-40e0-a78e-e893fb129ac6.png#align=left&display=inline&height=66&name=image.png&originHeight=116&originWidth=713&search=&size=21530&status=done&width=406)

对于得到的每个分支中, 各个方向处理后的特征, 通过这里的$$\oplus$$操作, 也就是按照通道拼接的计算方式进行组合后, 乘以权重, 再拼接所有分支的结果, 从而得到了最终的该阶段的输出特征.

第一阶段结束之后, 通过使用一个1x1卷积减小通道数, 之后再重复进行一次上面提到的计算过程. 这里会使用另一组attention weight, 但是按照图上的展示, 似乎是用的始终是同一个特征计算出来的. 可能实际实现有所不同. 因为文中提到: Then, we repeat the same process in the second round of recurrent translation **using another set of attention weight.**
最后接一个 `Conv1x1-GroupNorm-ReLU`  的结构进行特征的集成, 产生最终的SAC module的输出, 即所谓的"spatial attenuation context"(空间衰减上下文信息).

## 实验细节

* Backbone: 由于saliency detection在深度学习中的发展的原因, 很多模型仍然使用的是vgg16, resnet50来进行模型的比较与试验, 所以为了更公平的比较, 实际上在实验中也应该保证backbone的一致性. 但是本文使用的竟然是resnet101来作为backbone, 这就有点不厚道了
* The number of layers:we set the channel number of each FPN or SAC layer as 256 and did not use the feature maps at the first layer in both the FPN or SAC module due to the large memory footprint
* 损失函数使用的依然是交叉熵函数, 这里使用加和的形式
* 使用水平翻转作为训练时的数据增强策略
* 使用了梯度累积的策略进行训练, mini-batch为1, 每迭代10次更新一次权重参数
* 使用的是DUTS-TR训练模型
* We trained and tested our network on a single GPU (TITAN Xp) using input images of size **400×400**

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1570365005126-01122de2-a424-45c6-b843-c4b2522f7f02.png#align=left&display=inline&height=574&name=image.png&originHeight=574&originWidth=787&search=&size=186884&status=done&width=787)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1570365030491-dd71247c-9aae-4796-8714-c24a77aa1e72.png#align=left&display=inline&height=316&name=image.png&originHeight=316&originWidth=783&search=&size=87607&status=done&width=783)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1570365069806-a77d35e7-8290-4f77-9b0c-376741f25c6d.png#align=left&display=inline&height=207&name=image.png&originHeight=207&originWidth=383&search=&size=34702&status=done&width=383)

## 一些讨论

文中关于一些方法的讨论总结的蛮好的, 这里记录下:

There has been a lot of works on exploiting spatial context in deep CNNs for image analysis.

1. Dilated convolution [71], [72] takes context from larger regions by inserting holes into the convolution kernels, but the context information in use still has a fixed range in a local region.
1. ASPP [73], [74] and PSPNet [75] adopt multiple convolution kernels with different dilated rates or multiple pooling operations with different scales to aggregate spatial context using different region sizes; however, their designed kernel or pooling sizes are fixed, less flexible, and not adaptable to different inputs.
1. DSC [76], [77] adopts the attention weights to indicate the importance of context features aggregated from different directions, but it only obtains the global context with a fixed influence range over the spatial domain.
1. The non-local network [32] computes correlations between every pixel pair on the feature map to encode the global image semantics, but this method ignores the spatial relationship between pixels in the aggregation; for salient object detection, features of opposite semantics may, however, be important; see Figure 1.
1. PSANet [78] adaptively learns attention weights for each pixel to aggregate the information from different positions; however, it is unable to capture the context on lower-level feature maps in high resolutions due to the huge time and memory overhead.

Compared to these methods, our SAC-Net explores and adaptively aggregates context features implicitly with variable influence ranges; it is flexible, fast, and computationally friendly for efficient salient object detection. 

Lastly, we also analyzed the failure cases, for which we found to be highly challenging, also for the other state-of-theart methods. For instance, our method may fail for

* multiple salient objects in very different scales (see Figure 7 (top)),  where the network may regard the small objects as non-salient background;
* dark salient objects (see Figure 7 (middle)),  where there are insufficient context to determine whether the regions are salient or not;
* salient objects over a complex background (see Figure 7 (bottom)), where high-level scene knowledge is required to understand the image.

## 参考链接

* 论文:[https://arxiv.org/pdf/1903.10152.pdf](https://arxiv.org/pdf/1903.10152.pdf)
