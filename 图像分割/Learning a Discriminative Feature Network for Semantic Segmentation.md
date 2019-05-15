# Learning a Discriminative Feature Network for Semantic Segmentation

* [Learning a Discriminative Feature Network for Semantic Segmentation](#learning-a-discriminative-feature-network-for-semantic-segmentation)
  * [设计思想](#设计思想)
    * [Smooth Network](#smooth-network)
      * [Channel attention block](#channel-attention-block)
      * [Refinement residual block](#refinement-residual-block)
    * [Border Network](#border-network)
  * [实验](#实验)
    * [损失](#损失)
    * [细节](#细节)
    * [Ablation study](#ablation-study)
  * [总结](#总结)
  * [参考文章](#参考文章)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557649154750-30c7ce99-f6fe-4157-ae7d-7f294542b961.png#align=left&display=inline&height=418&name=image.png&originHeight=418&originWidth=1586&size=106105&status=done&width=1586)

## 设计思想

本文总结了现有语义分割方法仍然有待解决的两类 Challenge（如图所示）：

- Intra-class Inconsistency类内不一致（具有相同的语义标签，不同的表观特征的区域）
- Inter-class Indistinction类间模糊（具有不同的语义标签，相似的表观特征的区域）

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557652542294-cbe6cd95-4c3c-4902-ba25-00648d68ed9a.png#align=left&display=inline&height=557&name=image.png&originHeight=557&originWidth=599&size=198590&status=done&width=599)

所以，本文从宏观角度出发重新思考语义分割任务，提出应该将同一类的像素考虑成一个整体，也就需要增强类内一致性，增大类间区分性。总结而言，**需要更具有判别力的特征**。为了处理这样的两个问题，提出了新的Discriminative Feature Network (DFN)结构。

> Specifically, to handle the intra-class inconsistency problem, we specially **design a Smooth Network with Channel Attention Block and global average pooling to select the more discriminative features.**
> Furthermore, we **propose a Border Network to make the bilateral features of boundary distinguishable with deep semantic boundary supervision**.

下面是主要的结构：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557652674121-ad146029-6345-4fe0-ba62-318e044992de.png#align=left&display=inline&height=692&name=image.png&originHeight=692&originWidth=1260&size=188017&status=done&width=1260)
提出的网络主要是有两个重要的分支网络，一个是Border Network网络，一个是Smooth Network网络。这里的特征从中间的预训练网络送入，在各个阶段分别送出，送入到两侧的边界网络和平滑网络。

- 左侧使用了一个自下而上的过程，从开始的输入逐渐融合后面层级的特征，与以往的类似与FCN，UNet的自上而下的方向正好相反，因为这里主要想使用高层级的语义信息引导细化低层级的特征，进而得到更具有辨别力的特征信息（更多的是在强调边缘）。要注意这里始终分辨率和Res-1的输出是一致的，各阶段输出都会上采样。进而一定程度上解决类间不易辨别的问题。
- 右侧使用了一个自上而下的过程，通过反复的细化残差块以及通道注意力模块，获得了更多具有判别能力的特征，继而一定程度上解决了类内不一致的问题。

### Smooth Network

在语义分割的任务中，大多数现代方法将其视为密集预测问题。但是，预测有时在某些部分会产生不正确的结果，特别是大区域和复杂场景的部分，这被称为**类内不一致问题**。

类内不一致问题**主要是由于缺乏上下文信息**。

1. 因此，使用全局平均池化引入全局上下文信息。然而，**全局上下文信息只具有高语义信息，这对于恢复空间信息没有帮助**需要多尺度的感受野和上下文信息来改进空间信息**，正如大多数现代方法所做的那样。
1. 然而，存在一个问题，即**感受野的不同尺度产生具有不同程度辨别能力的特征，导致不一致的结果**。

因此，**需要选择更多的有判别能力的特征来预测某个特定类别的统一语义标签。**
将使用的ResNet网络分成五个阶段，每个阶段对应着不同分辨率的特征图。根据观察，不同阶段有着不同的识别能力，会导致不同的一致性表现。

- 在较低阶段，网络编码更精细的空间信息，然而，由于其较小的感受野和没有空间上下文信息的引导，它具有较差的语义一致性。
- 在高阶段，它具有很强的语义一致性，因为它感受野较大，然而预测的结果空间上是粗糙的。
- **总体而言，较低阶段可以进行更准确的空间预测，而较高阶段可以提供更准确的语义预测。**

基于这一观察，结合它们的优势，提出了一个平滑网络，利用高阶段的一致性来指导低阶段的最优预测。

观察到，在当前流行的语义分割体系结构中，主要有两种样式。

1. 第一个是“Backbone-Style”，如PSPNet，Deeplab v3。它嵌入了不同尺度的上下文信息，以通过金字塔空间池化模块或ASPP模块来**提高网络的一致性**。
1. 另一个是“Encoder-Decoder-Style”，如RefineNet，Global Convolutional Network。这类网络利用了不同阶段固有的多尺度上下文信息，但缺乏一致性最强的的全局上下文信息。
   1. 此外，当网络结合相邻阶段的特征时，它只是按通道加和这些特征。该操作忽略了不同阶段的不同一致性。
   2. 为了解决这个缺陷，文章首先嵌入了一个全局平均池化层，将U形架构扩展为V形架构。通过全局平均池化层，将最强的一致性约束引入网络中作为指导。
   3. 此外，为了提高一致性，设计了一个通道注意力模块，如图2c所示。该设计结合了相邻级的特征来计算信道注意向量3b。**高阶段的特征提供了一个强的一致性引导，而低阶段的特征给出了不同的特征判别信息。**这样，信道注意力矢量可以选择判别特征。

#### Channel attention block

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557751146907-9c425828-cacc-40a0-bbb4-e70271048746.png#align=left&display=inline&height=325&name=image.png&originHeight=325&originWidth=572&size=22402&status=done&width=572)

通道注意块CAB**旨在改变每个阶段的功能权重，以增强一致性**，如图3所示。

在FCN架构中，卷积运算符输出一个得分图，给出每个像素对于每个类的概率。

在等式1中，得分图的最终得分仅来自于对特征图的所有通道求和。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557719151822-8bd22621-48fc-4ba5-bdb6-730fe902f15b.png#align=left&display=inline&height=99&name=image.png&originHeight=99&originWidth=648&size=9446&status=done&width=648)这里的x是网络输出的特征图，w表示卷积核，k表示通道序号，一共有K个通道，D表示像素坐标。![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557719164461-357377cc-64a2-4053-9f24-b15e6b3638f3.png#align=left&display=inline&height=98&name=image.png&originHeight=98&originWidth=648&size=11626&status=done&width=648)这里的delta表示预测概率，y是网络的输出。也就是对于K个通道计算softmax。

正如式子1和2，最终的预测标签是最高概率的类别。因此，假定预测结果是y0，而真值标签是y1。因此，可以引入参数来更改最高概率值从y0变到y1，如式子3所示。![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557719180615-50334da0-a65e-4ae2-8801-7b4441300b65.png#align=left&display=inline&height=147&name=image.png&originHeight=147&originWidth=633&size=13461&status=done&width=633)yba是网络的新的预测并且这里的alpha是sigmoid(x;w)。也就是对于x通过参数w处理后的结果进行了sigmoid处理。

在等式1中，它隐含地表明不同信道的权重（在最终预测中所起到的作用）是相等的。但是，如第1节所述，**不同阶段的特征具有不同程度的区分能力，这导致预测的不同一致性。为了获得类内一致的预测，应该提取具有辨别力的特征并抑制不具有判别能力的特征。**因此，在方程3中，该值适用于特征映射x，它表示使用CAB的特征选择。**通过这种设计，可以使网络逐阶段获得判别特征，使预测内部一致。**

```python
# 个人实现
class CAB(nn.Module):
    def __init__(self, in_c):
        """
        :param in_c: lin_feat与din_feat通道数相同，模块输出通道数等于任一个
        :param k: 通道缩减数量
        """
        super(CAB, self).__init__()

        self.conv1x1_1 = nn.Conv2d(in_channels=2 * in_c,
                                   out_channels=in_c,
                                   kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv2d(in_channels=in_c,
                                   out_channels=in_c,
                                   kernel_size=1)

    def forward(self, lin_feat, din_feat):
        """
        :param lin_feat: 来自同级的特征
        :param din_feat: 来自低分辨率层级的特征
        :return: 注意力加权后的特征
        """
        din_feat = F.interpolate(din_feat,
                                 size=lin_feat.size()[2:],
                                 mode='bilinear',
                                 align_corners=True)
        main_feat = lin_feat

        x_ori = torch.cat((din_feat, lin_feat), dim=1)
        x = F.adaptive_avg_pool2d(x_ori, (1, 1)).expand_as(x_ori)
        x = self.conv1x1_1(x)
        x = self.relu(x)
        x = self.conv1x1_2(x)
        x = torch.sigmoid(x)
        return din_feat + main_feat * x


# 官方
class SELayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion + x2

        return fm
```

#### Refinement residual block

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557751312855-b81917d7-4891-471b-a956-46167802631a.png#align=left&display=inline&height=300&name=image.png&originHeight=300&originWidth=575&size=21542&status=done&width=575)

特征网络中每个阶段的特征图都通过细化残差块，如图2b所示。

- 该块的第一个分量是1x1卷积层。使用它将通道数统一到512。同时，它可以**组合所有通道的信息**。
- 然后是基本残差块，可以重新精细化特征图（相当于分支学习了一个残差，而这个残差认为是一些需要细致考虑的部分，仔细思考，多算算）。
- 此外，这个区块可以增强每个阶段的区分能力，这得益于ResNet的结构。

```python
# 个人实现
class RRB(nn.Module):
    def __init__(self, in_c, mid_c=512):
        super(RRB, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_channels=in_c,
                                 out_channels=mid_c,
                                 kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels=mid_c,
                                   out_channels=mid_c,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bn = nn.BatchNorm2d(mid_c)
        self.conv3x3_2 = nn.Conv2d(in_channels=mid_c,
                                   out_channels=mid_c,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, in_feat):
        in_feat = self.conv1x1(in_feat)

        main_feat = in_feat
        x = self.conv3x3_1(in_feat)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3x3_2(x)
        return self.relu(main_feat + x)


# 官方
class RefineResidual(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-5):
        super(RefineResidual, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                  stride=1, padding=0, dilation=1,
                                  bias=has_bias)
        self.cbr = ConvBnRelu(out_planes, out_planes, ksize, 1,
                              ksize // 2, has_bias=has_bias,
                              norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv_1x1(x)
        t = self.cbr(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x
```

### Border Network

在语义分割任务中，在具有相似外观的不同类别之间的预测结果容易产生混乱，尤其是当它们在空间上相邻时。**因此需要放大特征的区别**。因为这种动机，采用语义边界来指导特征学习。

为了提取准确的语义边界，应用语义边界的显式监督，使网络学习具有强大的类间区分能力的特征。因此提出了一个边界网络，以扩大特征的类间区分。**它直接学习具有显式语义边界监督的语义边界，类似于语义边界检测任务。**这使得语义边界两侧的特征可以区分。

特征网络具有不同的阶段。低阶段功能具有更详细的信息，而高阶段功能具有更高的语义信息。这里需要具有更多语义信息的语义边界。因此设计了一个自下而上的边界网络，这个网络可以同时获得来自低层级的较为准确的边界信息，也可以获得来自较高层级的语义信息，这消除了一些缺乏语义信息的原始边缘。这样，高层级的语义信息可以逐步细化来自低层级的边界信息。**网络的监督信号是通过传统的图像处理方法，如Canny从语义分割的真值中获得。**

为了弥补正负样本的不平衡，使用focal loss来监督边界网络的输出，如公式4所示。调整focal loss的参数alpha和gamma，以获得更好的性能。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557752150666-53613f25-1660-4932-bb43-692392bfa0f9.png#align=left&display=inline&height=68&name=image.png&originHeight=68&originWidth=635&size=6734&status=done&width=635)

这里pk是类别k的估计概率。一共有K个类别。

边界网络主要关注语义边界，其将边界两侧的类分开。为了提取准确的语义边界，双方的特征将变得更加可辨。这恰好达到了目标，即尽可能地使这些特征类间类间可辨。

```python
# 官方
class BNRefine(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-5):
        super(BNRefine, self).__init__()
        self.conv_bn_relu = ConvBnRelu(in_planes, out_planes, ksize, 1,
                                       ksize // 2, has_bias=has_bias,
                                       norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        t = self.conv_bn_relu(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x
```

## 实验

### 损失

在平滑网络中，使用softmax loss来**监督每个阶段的上采样输出**，不包括全局平均池化层，使用focal loss来监督边界网络的输出。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557754302048-f7830fed-1c51-4afa-a777-e089eeed9582.png#align=left&display=inline&height=123&name=image.png&originHeight=123&originWidth=634&size=18564&status=done&width=634)

### 细节

- We evaluate our approach on two public datasets: PAS-CAL VOC 2012 and Cityscapes.
    - Pascal VOC 2012：The original dataset is augmented by the Semantic Boundaries Dataset[Semantic contours from inverse detectors], resulting in 10,582 images for training. 
    - on the Cityscapes dataset. In training, our crop size of image is800800.We observe that for the high resolution of image the large crop size is useful.
- Our proposed network is based on the ResNet-101 pre-trained on ImageNet. And we use the FCN4 as our base segmentation framework. 
- SGD with batch size 32, momentum 0.9, weight decay 0.0001.
- "Poly" learning rate policy, multiplied by ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557754302052-1b2c358d-313b-457b-a791-70cbed16aab4.png#align=left&display=inline&height=40&name=image.png&originHeight=40&originWidth=236&size=5367&status=done&width=236) with power 0.9, initial learning rate 4e-3.
- lambda = 0.1(比较结果)
- use the mean IOU as the metric
- We use **mean subtraction and random horizontal flip**randomly scale the input images**use 5 scales {0.5, 0.75, 1, 1.5, 1.75}** on both datasets. 
- use the ResNet-101 as our base feature network, and directly upsample the ouput.

### Ablation study

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557754750281-309b7b1d-3917-426f-bedf-da038023ef2f.png#align=left&display=inline&height=195&name=image.png&originHeight=195&originWidth=642&size=33884&status=done&width=642)

基线网络的性能。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557754828396-5d294ed5-a038-4c65-89b5-466069326315.png#align=left&display=inline&height=423&name=image.png&originHeight=423&originWidth=650&size=82916&status=done&width=650)

改进结构后的比较。Then we extend the base network to FCN4 structure with our proposed Refinement Residual Block (RRB), which improves the performance from 72.86% to 76.65%. 从上表中可以看出，提出的GP，RRB， CAB都是有效果提升的。而且这里的深监督也起到了一定的作用。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557755168624-db85ab46-e738-495e-a01c-8694011f40f0.png#align=left&display=inline&height=535&name=image.png&originHeight=535&originWidth=656&size=225124&status=done&width=656)

展示了smooth network的影响。Obviously, **our Smooth Network can effectively make the prediction more consistent. **

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557755423166-5c87459c-3838-4dd7-bade-87026513670a.png#align=left&display=inline&height=323&name=image.png&originHeight=323&originWidth=650&size=66796&status=done&width=650)

展示了两个子网络的影响。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557755519248-290d7287-b7d4-4619-89e1-f084a72e9961.png#align=left&display=inline&height=569&name=image.png&originHeight=569&originWidth=648&size=215670&status=done&width=648)

展示了Border Network的影响。实际上使用真值的边界引导最终的预测，确实可以在一定程度上降低类间的混淆。因为边界也算是一种强约束，反过来对于高层语义信息的影响也是较为直接的。由于精确的边界监督信号，网络放大了双边特征的区别，提取了语义边界。边界网络优化了语义边界，这是整个图像中相对较小的一部分，因此这种设计略有改进。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557755812114-27e93460-987b-4c45-aaab-3c3b521cc28d.png#align=left&display=inline&height=619&name=image.png&originHeight=619&originWidth=645&size=268660&status=done&width=645)

展示了边界网络的预测结果。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557755869305-7396f9bb-a103-40e1-85cf-702cb16dd5d6.png#align=left&display=inline&height=520&name=image.png&originHeight=520&originWidth=658&size=69439&status=done&width=658)

对于不同的lambda的测试。最终选择使用lambda=0.1。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557755976142-bd159580-395c-4567-bd4e-222279fe1ccb.png#align=left&display=inline&height=818&name=image.png&originHeight=818&originWidth=650&size=305889&status=done&width=650)

各阶段细化过程的展示。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557756039761-e0b43e4a-8836-48ed-be69-ba3e87e714e5.png#align=left&display=inline&height=235&name=image.png&originHeight=235&originWidth=655&size=40647&status=done&width=655)

In evaluation, we **apply the multi-scale inputs and also horizontally flip the inputs** to further improve the performance. In addition, since thePASCAL VOC2012 dataset provides higher quality of annotation than the augmented datasets, we **further fine-tune our model on PASCAL VOC 2012 train set for evaluation on validation set**. More performance details are listed in Table 4.

And then for evaluation on test set, we use the PASCAL VOC 2012 trainval set to further fine-tune our proposed method.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557756532735-0e476fde-eec8-40b7-ac79-d3d9d4c7a5f5.png#align=left&display=inline&height=705&name=image.png&originHeight=705&originWidth=656&size=106036&status=done&width=656)

In the end, our proposed approach respectively achieves performance of 82.7% and 86.2% with and without MS-COCO fine-tuning, as shown in Table 5. Note that, we **do not use Dense-CRF post-processing for our method.**

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557756532803-da189882-1fc9-47e3-b924-ef6a30feafa2.png#align=left&display=inline&height=523&name=image.png&originHeight=523&originWidth=655&size=84989&status=done&width=655)

展示了在Cityscapes的测试结果。

## 总结

最后总结一下，本文的贡献主要有 4 个方面：

- 从一个新的宏观视角重新思考语义分割，将其看作一项**把一致的语义标签分配给一类物体**（而不仅仅是在像素层面）的任务。
- 提出 **DFN 同时解决类内一致和类间差别问题**。DFN 分别在 PASCAL VOC 2012 和 Cityscapes 数据集上取得 86.2% 和 80.3% 的当前最优 mean IOU，证实了该方法的有效性。
- 提出 Smooth Network，通过全局语境和通道注意力模块提升类内一致性。
- 提出一种自下而上的 Border Network，利用多层边界监督信号增大语义边界两边的特征变化，同时优化预测的语义边界。

## 参考文章

- [https://zhuanlan.zhihu.com/p/55263898](https://zhuanlan.zhihu.com/p/55263898)
- 文章解析：[https://mp.weixin.qq.com/s/4csAftUmJErdA4qn26mO3Q](https://mp.weixin.qq.com/s/4csAftUmJErdA4qn26mO3Q)
- 相关操作的代码：[https://github.com/ycszen/TorchSeg/blob/master/furnace/seg_opr/seg_oprs.py](https://github.com/ycszen/TorchSeg/blob/master/furnace/seg_opr/seg_oprs.py)
