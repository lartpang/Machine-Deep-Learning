# FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation

* [FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation](#fastfcn-rethinking-dilated-convolution-in-the-backbone-for-semantic-segmentation)
  * [相关工作](#相关工作)
    * [DUpsample](#dupsample)
  * [主要工作](#主要工作)
    * [Joint Pyramid Upsampling (JPU)](#joint-pyramid-upsampling-jpu)
    * [Joint Upsampling](#joint-upsampling)
    * [使用JPU模拟带有扩张卷积的基础网络的卷积块](#使用jpu模拟带有扩张卷积的基础网络的卷积块)
    * [与ASPP的差异](#与aspp的差异)
  * [实验细节](#实验细节)
    * [验证扩张卷积和上采样模块的有效性](#验证扩张卷积和上采样模块的有效性)
    * [FPS](#fps)
    * [模型比较](#模型比较)
  * [关键代码](#关键代码)
  * [参考链接](#参考链接)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557228377384-5391d6a9-8346-433a-907e-aefb0e11fefb.png#align=left&display=inline&height=464&name=image.png&originHeight=464&originWidth=1432&size=97616&status=done&width=1432)

## 相关工作

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557298174100-c31c6cd3-5f3a-4761-a86a-d3c9d3dcd020.png#align=left&display=inline&height=472&name=image.png&originHeight=472&originWidth=1250&size=197380&status=done&width=1250)

随着FCNs在语义分割方面取得了巨大成功，FCN之后，有两个显着的方向，即DilatedFCN和Encoder-Decoder模型。

1. DilatedFCN利用扩张的卷积来保持感知视野，并采用多尺度上下文模块来处理高级特征图。
1. Encoder-Decoder提出利用的编码器来提取多级特征图，然后由解码器将其组合生成最终预测。

### DUpsample

数据相关的上采样DUpsampling [Decoders matter for semantic segmentation: Data-dependent decoding enables flexible feature aggregation]也与文中的方法有关，它利用了分割标签空间中的冗余，并且能够从CNN的低分辨率输出中恢复像素级别预测。与文中的方法相比，DUpsampling对标签空间具有很强的依赖性，这对于更大或更复杂的标签空间来说很难概括。

## 主要工作

为了应对在backbone中使用扩张卷积的时候（一般就会取消对应位置的下采样，使得输出的是缩放8倍的尺寸），造成的过大的计算复杂度和存储占用，**文章提出了一个联合金字塔上采样模块（Joint Pyramid Upsampling (JPU) ）来实现这个输出缩放8倍的操作。用其替换扩张卷积，并在后面跟着其他的已有的工作的模块，实现了较好的效果。**

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557229657409-f652b286-2de4-4d53-a3cc-6d06aaa64b34.png#align=left&display=inline&height=737&name=image.png&originHeight=737&originWidth=1508&size=228864&status=done&width=1508)

### Joint Pyramid Upsampling (JPU)

所提出的JPU被设计用于生成特征图，该特征图近似于来自DilatedFCN的主干的最终特征图的尺寸。下面是JPU的主要结构，先简单说下过程。
![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557233691744-a0b01281-2e7d-43ad-af9c-7889f9d0bdde.png#align=left&display=inline&height=430&name=image.png&originHeight=430&originWidth=593&size=50364&status=done&width=593)

1. a：来自卷积块3/4/5的输出特征图送入JPU中，三者分别通过进一步卷积处理后
1. b：上采样到原图的八分之一，之后拼接，使用一个深度分离的扩张金字塔，输出后拼接
1. c：进一步卷积输出对应于原图八分之一尺寸的特征

这里看似流程很简单，实际上这里是在使用CNN来模拟之前的Joint Upsampling操作。

### Joint Upsampling

给定一个低分辨率的目标图像，和一个高分辨率的引导图像，联合上采样旨在通过迁移来自引导图像的细节和结构信息来生成一个高分辨率的目标图像。

> [Deep joint image filtering]构建了一个基于CNN的联合过滤器，它学会了恢复指导图像中的结构细节。[Fast end-to-end trainable guided filter]提出了一种端到端可训练的引导过滤模块，其有条件地对低分辨率图像进行上采样。我们的方法与上述方法有关。然而，**提出的JPU用于处理具有大量通道的特征图**，而前面的两种方法是专门为处理3通道图像而设计的，这些图像无法捕获高维特征图中的复杂关系。此外，方法的动机和目标是完全不同的。


一般的，低分辨率目标图像yl可以使用转换函数f和低分辨率引导图像xl来生成，也就是yl=f(xl)，由于引导图像和目标图像之间的这种关系是具有一定的稳定性的，对于二者对应的高分辨率图像而言，也该存在yh=f(xh)的类似关系。一般是使用一个近似的转换关系hatf来近似f，而且hatf的计算复杂度也要低一些，例如如果f是一个MLP，那么hatf就可以简化为一个线性转换。于是对于高分辨率之间的关系转化为yh=hatf(xh)。

这样，给定低分辨率的引导图像xl，目标图像yl和高分辨率的引导图像xh，联合上采样可以被定义如下：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557234558363-82c43686-8f1f-4eda-b233-ea5f5eee18c5.png#align=left&display=inline&height=52&name=image.png&originHeight=52&originWidth=569&size=10855&status=done&width=569)
这里的H是一个所有可能的转换函数的集合，而||.||表示预定义的距离计算。

### 使用JPU模拟带有扩张卷积的基础网络的卷积块

这里首先简单分析了一下一些常用的手段——扩张卷积和跨步卷积。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557234735061-88dd5bcb-952c-46d5-856d-59851c115bc0.png#align=left&display=inline&height=537&name=image.png&originHeight=537&originWidth=1223&size=132346&status=done&width=1223)

可以看到，扩张卷积 `Cd` 可以被拆分为 `Split+Conv+Merge` 的系列操作，简化表示为 `SCrM` ，其中的 `Cr` 表示通常的卷积操作。而右边的跨步卷积 `Cs` 可以拆分为 `Conv+Reduce` 的系列操作，简化表示为 `CrR` 。

于是在**扩张卷积改造的backbone中**，在某一个改造后的卷积块中，由输入特征得到输出特征的过程可以简化表示如下：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557235047332-327eb8ef-c819-451a-9da3-a04c57e11f4b.png#align=left&display=inline&height=288&name=image.png&originHeight=288&originWidth=567&size=31638&status=done&width=567)

很直观，输入的特征x通过正常的卷积之后，经过一系列的扩张卷积（这里是扩张卷积改造的backbone的惯用方法），对扩张卷积拆分合并之后得到了第三行的式子，n个Cr可以合并为一个连续的正常卷积的处理。而其中第四行，使用正常卷积处理x得到了较高分辨率的特征ym，它通过S分解可以认为是分成了y0和y1两种特征。说到这里，这里的分离操作像不像不同扩张率的金字塔的结构呢？

而**在本文的方法**中（也就是正常的下采样的过程，这里的**下采样使用的是跨步卷积**），在某一个对应的卷积块中，输出特征ys可以表示为：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557235082969-77e01a4c-b6f7-435e-a335-296f95379e01.png#align=left&display=inline&height=167&name=image.png&originHeight=167&originWidth=565&size=20458&status=done&width=565)

同样的，通过拆分合并，将操作化归到了最后一行的ym/R/Cr的操作序列。ym是输入特征通过正常卷积输出的结果，可以认为与式子2中的ym是一致的。而这里对于ym有着一个缩减reduce的操作（实际上就是下采样的部分），得到了ym0，这个可以认为是“高分辨率”的ym对应的“低分辨率”的ym0之间的一个关系。

这里之所以强调高分辨率和低分辨率，主要是为了和上一小结中的 `Joint Upsampling` 相照应，因为实际上就是在模拟这个过程。

于是对于已经给定的x和ys（正常的backbone可以得到），近似与特征yd的特征y可以如下方式获得：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557235986679-7eccd596-97dd-455b-a8e6-abc888d42ed1.png#align=left&display=inline&height=152&name=image.png&originHeight=152&originWidth=588&size=18180&status=done&width=588)

这里和前面提到的联合上采样是一致的。也即是使用低分辨率的ys和ym0之间的关系近似获取一个较优的转换函数hath，用于高分辨率的（引导）特征ym来生成较高分辨率的（目标）特征y。

似乎问题就这么解决了，但是要注意的是，这里式子4是一个忧患微调，使用迭代梯度下降来收敛，要耗费大量的时间，文章就提出了JPU模块，来模拟这个整体的映射关系。

![](https://cdn.nlark.com/yuque/0/2019/png/192314/1557233691744-a0b01281-2e7d-43ad-af9c-7889f9d0bdde.png#align=left&display=inline&height=430&originHeight=430&originWidth=593&status=done&width=593)
为了实现前面描述的映射关系，需要通过给定x生成ym，然后来自ym0和ys的特征需要收集起来，来学习映射hath，最终一个卷积块要用来转换聚集的特征到最终的预测y。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557237239691-f255b885-2482-47dd-b096-e18677f008f6.png#align=left&display=inline&height=432&name=image.png&originHeight=432&originWidth=608&size=57213&status=done&width=608)

按照这样的分析，文章提出了图4所示的JPU模块。具体而言：

1. 每一个输入特征图（相当于ym）首先被使用标准卷积进行处理，这就是相当于给定x来生成ym，并且将ym转换到了一个嵌入空间降低了维度，这样所有的输入特征映射到相同的空间（通道维度是一样的），能够更好的融合并降低计算复杂度。
1. 然后生成的特征图上采样之后拼接，这也就生成了途中所示的yc。四个深度分离卷积（扩张率不同）被应用到并行分支提取特征。不同的扩张率实际上模拟的是不同尺度的特征。
    1. 例如这里扩张率为1的时候，可以捕获特征ym0与ym剩余部分的关系。可以看图5中右侧的蓝色框。
    1. 另外的扩张率为2/4/8的分支被设计来学习映射hath，来转换ym0到ys，如图中绿色框显示的那样。（这里的ym0认为是途中左侧所示的数据，而映射得到的ys是深红色数据，注意右图中的ys只有有标示的地方是ys，ys是低分辨率的）
3. 提取的特征编码了ym0和ys之间的映射（**低分辨率的引导特征和目标特征之间的关系**引导特征的高分辨率和低分辨率之间的关系**结合两种关系，可以得到高分辨率的目标特征**）。

### 与ASPP的差异

这里使用了多尺度特征信息，但是不同于ASPP只是利用了最后一个特征图的信息，而这里使用来自多级特征图的信息来进行多尺度信息的提取。

## 实验细节

- Dataset: Pascal Context dataset, ADE20K,
    - For training on Pascal Context, we follow the protocol presented in [Context encoding for semantic segmentation].
    - ADE20K dataset is a scene parsing bench-mark, which contains 150 stuff/object categories. The dataset includes **20K/2K/3K images for training (train), val-idation (val), and testing (test).**
- Concretely, we set the learning rate to 0.001 initially, which gradually decreases to 0 by following the ”**poly**” strategy (power = 0.9).
- For data augmentation:
    - **randomly scale (from 0.5 to 2.0)**
    - **left-right flip the input images**.
- The images are then cropped to 480x480 and grouped with batch size 16.
- The network is trained for 80 epochs with SGD
    - the momentum is set to 0.9
    - weight decay is set to 1e-4
- All the experiments are conducted in a workstation with 4 Titan-Xp GPUs (12G per GPU).
- We employ pixel-wise cross-entropy as the loss function.
- ResNet-50 and ResNet-101 are used as the backbone, which are widely used in most existing segmentation methods as the standard backbones.

由于网络设计的实际上是个中间的部分，对于头部的部分没有改造，可以使用Encoding或者ASPP或者PSP的结构。

### 验证扩张卷积和上采样模块的有效性

对于os=32的，也即是下采样32倍的部分，对于原本使用扩张卷积的网络，这里使用图中所示上采样方式进行上采样到原图1/8大小，再使用原网络的头部。这里只是替换了encoding网络，发现效果很替换后效果不好，不如使用扩张卷积，但是这里却又可以看出JPU的有效性。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557238557780-aef0a822-e869-49d2-8295-ba2427599e76.png#align=left&display=inline&height=397&name=image.png&originHeight=397&originWidth=643&size=74506&status=done&width=643)![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557238890263-76586328-9da9-4d76-8b7e-f2551ea4e0ce.png#align=left&display=inline&height=440&name=image.png&originHeight=440&originWidth=590&size=91251&status=done&width=590)

一些结果如图6所示。

1. Encoding-32-Bilinear（图6c）成功捕获全局语义信息，从而对鸟类和天空进行粗略分割。然而，鸟的边界是不准确的，并且树枝的大部分部分未被标记出来。
1. 当用FPN替换双线性插值时（图6d），鸟和树枝用准确的边界成功标出，这显示了组合低级和高级特征图的效果。
1. 扩张卷积可以获得稍好的结果（图6e）。
1. 对于提出的方法（图6f），它准确地标出了主分支和侧射，这表明了所提出的联合上采样模块的有效性。特别是，侧面反映出JPU从多级特征图中提取多尺度背景的能力。

### FPS

为了比较计算复杂度，使用了FPS作为评估指标。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557239124435-fb44e103-6692-4c52-bdd3-e199670060f2.png#align=left&display=inline&height=647&name=image.png&originHeight=647&originWidth=580&size=113634&status=done&width=580)

JPU也造成了一定的计算复杂度的增加。

### 模型比较

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557297441009-7768f86f-0713-44bd-a10f-ad094c82c26d.png#align=left&display=inline&height=533&name=image.png&originHeight=533&originWidth=620&size=96015&status=done&width=620)

Following [Context encoding for semantic segmentation], the mIoU reported in Table 1 is on 59 classes w/o background. **In this table, the mIoU is measured on 60 classes w/ back-ground for a fair comparison with other methods.** Besides, we average the network prediction in **multiple scales for evaluation** in this table.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557297452985-37625a7d-ec41-4de7-b55d-2754cd7d7213.png#align=left&display=inline&height=450&name=image.png&originHeight=450&originWidth=593&size=84934&status=done&width=593)![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557297471655-4007fd4d-ddc0-4ced-8a7d-7582bb8dcfe5.png#align=left&display=inline&height=302&name=image.png&originHeight=302&originWidth=599&size=60639&status=done&width=599)

Our method (ResNet-101) performs a little worse than EncNet, and we attribute this to the spatial resolution of the training images. **Concretely, in our method, the training images arecropped to 480x480 for processing 4 images in a GPU with 12G memory. However, EncNet is trained with 576x576 images on GPUs with memory larger than 12G.** We then fine-tune our network on the trainset and valset for another **20 epochs with learning rate 0.001**. The predictions on the testset are submitted to the evaluation server. As shown in Table 5, our method outperforms two winning entries from the COCO-Place challenge 2017.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557298107437-74015e57-9c31-4210-bcf9-8f44421c8d9a.png#align=left&display=inline&height=551&name=image.png&originHeight=551&originWidth=599&size=131392&status=done&width=599)

## 关键代码

```python
# https://github.com/wuhuikai/FastFCN/blob/master/encoding/nn/customize.py

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.upsample(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return inputs[0], inputs[1], inputs[2], feat
```

## 参考链接

- 代码：[https://github.com/wuhuikai/FastFCN](https://github.com/wuhuikai/FastFCN)
