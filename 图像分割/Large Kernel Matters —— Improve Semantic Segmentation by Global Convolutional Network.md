# Large Kernel Matters —— Improve Semantic Segmentation by Global Convolutional Network

* [Large Kernel Matters —— Improve Semantic Segmentation by Global Convolutional Network](#large-kernel-matters--improve-semantic-segmentation-by-global-convolutional-network)
  * [主要工作](#主要工作)
  * [实验细节](#实验细节)
    * [验证Large Kernel Conv的有效性](#验证large-kernel-conv的有效性)
    * [GCN的核增大性能提升，是否在于参数的增加？](#gcn的核增大性能提升是否在于参数的增加)
    * [是否可以通过堆叠多个Small Kernel Size的Conv来替代GCN？](#是否可以通过堆叠多个small-kernel-size的conv来替代gcn)
    * [GCN对于分割结果有怎样的贡献](#gcn对于分割结果有怎样的贡献)
    * [在预训练模型上的嵌入](#在预训练模型上的嵌入)
    * [Pascal VOC 2012](#pascal-voc-2012)
    * [Cityscapes](#cityscapes)
  * [总结](#总结)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557307676936-4c9dfc2f-da0b-405c-96de-12c54a140e44.png#align=left&display=inline&height=288&name=image.png&originHeight=288&originWidth=1264&size=79278&status=done&width=1264)

## 主要工作

GCN 主要将 Semantic Segmentation分解为：Classification 和 Localization两个问题。但是，这两个任务本质对特征的需求是矛盾的，Classification需要特征对多种Transformation具有不变性，而 Localization需要对 Transformation比较敏感。但是，普通的 Segmentation Model大多针对 Localization Issue设计，正如图1(b)所示，而这不利于 Classification。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557316452935-9e7bed38-544e-47f7-8d1a-f4312248dfa4.png#align=left&display=inline&height=465&name=image.png&originHeight=465&originWidth=646&size=80055&status=done&width=646)

所以，为了兼顾这两个 Task，本文提出了两个 Principle：

1. 从 Localization 来看，我们需要全卷积网络，而且不能有全连接或者全局池化等操作丢失位置信息。
1. 从 Classification 来看，我们需要让 Per-pixel Classifier 或者 Feature Map 上每个点的连接更稠密一些，也就需要更大的 Kernel Size，如图1c所示。

根据这两条 Principle，本文提出了Global Convolutional Network（GCN）。如图2所示，这个方法整体结构正是背景介绍中提到的U-shape结构，**其核心模块主要包括：GCN 和 BR**。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557307660728-5d030965-2067-46f6-9fce-53e2d7ebc20a.png#align=left&display=inline&height=777&name=image.png&originHeight=777&originWidth=1314&size=233984&status=done&width=1314)

此处主要介绍GCN设计。正如图3(b)所示，它采用了较大 Kernel Size的卷积核，来同时解决上述的两个 Issue；然后根据卷积分解，利用1 x k + k x 1 和k x 1 + 1 x k 的卷积来替代原来的k x k 大核卷积。相对于原本的大核卷积，该设计能明显降低参数量和计算量。图3可视化了 Large Kernel Conv 和 普通 Conv网络有效感受野的对比。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557316552742-ac8b6f13-c182-4b42-b828-4a06c553de30.png#align=left&display=inline&height=462&name=image.png&originHeight=462&originWidth=1308&size=390619&status=done&width=1308)

> **Feature maps from modern networks such as GoolgeNet or ResNet usually have very large receptive field because of the deep architecture. However, studies [Object detectors emerge in deep scene cnns] show that network tends to gather information mainly from a much smaller region in the receptive field**, which is called valid receptive field (VRF) in this paper. 

可以看出，通过提出的全局卷积网络可以得到更大的VRF。

## 实验细节

- We evaluate our approach on the standard benchmark PASCAL VOC 2012 and Cityscapes.
    - PASCAL VOC 2012 has 1464 images for training, 1449 images for validation and 1456 images for testing, which belongs to 20 object classes along with one background class.
    - We **also use the Semantic Boundaries Dataset as auxiliary dataset**, resulting in 10,582 images for training.
- We choose the state-of-the-art network ResNet 152 (pretrained on ImageNet) as our base model for fine tuning.
- During the training time, we use standard SGD with batch size 1, momentum 0.99 and weight decay 0.0005 .
- Data augmentations like mean subtraction and horizontal flip are also applied in training.
- The performance is measured by standard mean intersection-over-union (IoU).

### 验证Large Kernel Conv的有效性

做后续的实验的时候：we use PASCAL VOC 2012 validation set for the evaluation. For all succeeding experiments, we **pad each input image into 512x512 so that the top-most feature map is 16x16**. 

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557317140025-7d81b75f-e2ce-4695-a534-bbd567dbb7b5.png#align=left&display=inline&height=386&name=image.png&originHeight=386&originWidth=626&size=58699&status=done&width=626)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557317551887-f7ac431d-c1ec-41b4-b491-0ec3bd7f20f2.png#align=left&display=inline&height=178&name=image.png&originHeight=178&originWidth=669&size=40503&status=done&width=669)

表格比较了图4中的A随着k的变化而得到的性能的变化，而B则是基线。

因为特征图最小为16x16，所以卷积核到15的时候基本上已近变成了真正的全局卷积。从上面比较了可以看出，随着k的增大，性能是在提升的。

### GCN的核增大性能提升，是否在于参数的增加？

为了进一步验证是否是因为k的增加导致参数的增加，继而导致的性能的提升，这里进一步测试，设计了几个如图4中C的结构，改变其A和C的k，统计如下：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557317884637-5a6dffb7-19e6-4623-a68e-113e253a2f06.png#align=left&display=inline&height=312&name=image.png&originHeight=312&originWidth=649&size=77770&status=done&width=649)

可以看出，随着参数的增加，GCN性能一如既往的替身，但是一般的卷积在k>5的时候反而会下降。然而，在训练中发现一般的大核卷积实际上使得网络难以收敛，而GCN结构将不会面临这个问题。因此，实际原因还需要进一步研究。

### 是否可以通过堆叠多个Small Kernel Size的Conv来替代GCN？


GCN使用 Large Kernel Size增大了感受野，是否可以通过堆叠多个 Small Kernel Size的 Conv来替代？文章为此设计了图4D（n个小卷积核（3x3）堆叠来代替一个kxk的核）的结构，与A结构进行实验，对比两者的结果。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557318182302-705be1fe-9684-421a-982c-698a50d82949.png#align=left&display=inline&height=251&name=image.png&originHeight=251&originWidth=646&size=58143&status=done&width=646)

> we do not apply nonlinearity within convolutional stacks so as to keep consistent with GCN structure.

可以看到 GCN 依然优于普通 Conv 的堆叠，尤其是在较大 Kernel Size 的情况下。

对于大的内核大小（例如k = 7），3x3卷积堆叠将带来比GCN更多的参数，这可能对结果产生副作用。因此，尝试减少卷积堆叠中间特征映射的数量，并进行进一步的比较。结果列于表4中。显然，其性能受到参数少的影响。总之，与一般的卷积堆叠相比，GCN是一种更好的结构。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557318796479-039981da-1ca0-4138-8f36-a65c4c81bcd1.png#align=left&display=inline&height=234&name=image.png&originHeight=234&originWidth=667&size=57929&status=done&width=667)

> 这是一个很有价值的实验，可以启发去思考关于网络感受野的问题。以往认为，通过堆叠多个小核 Conv 可以达到和大核 Conv 一样的感受野，同时计算量还更少。最常见的应用比如 VGG-Net。但是，实际上并非如此。随着网络深度的提升，理论上网络的感受野大多可以直接覆盖全图，但是实际有效感受野却远小于此。
> 网友的理解是对同一个 Feature Map 进行卷积，边缘区域进行计算的次数会小于中心区域，所以随着 Conv 的不断堆叠，实际上会导致边缘感受野的衰减，即有效感受野会远小于理论感受野。

### GCN对于分割结果有怎样的贡献

GCN通过引入与特征图的密集连接来提高分割模型的分类能力，这有助于处理大的变换。基于此，可以推断出位于大型物体中心的像素可能会从GCN中受益更多，因为它非常接近“纯”分类问题。然而，对于物体的边界像素，性能主要受定位能力的影响。
为了验证推断，将分割得分图分为两部分：

1. 边界区域，其像素位于靠近物体的边界（距离<=7）
1. 内部区域作为其他像素。

在两个区域评估分割模型（GCN有着k=15）。结果如表5所示，主要发现GCN模型有助于提高内部区域的准确性，而边界区域的影响很小，这有力地支持了前面的论证。此外，在表5中，还评估了前面提到的边界细化（BF）块。与GCN结构相反，BF主要提高了边界区域的精度，这也证实了其有效性。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557321409955-9f5039db-cb44-4651-8544-6cdf33450d93.png#align=left&display=inline&height=236&name=image.png&originHeight=236&originWidth=646&size=56243&status=done&width=646)

### 在预训练模型上的嵌入

上面的模型是从ResNet-152网络中精确调整的。由于大内核在分割任务中起着至关重要的作用，因此在预训练模型上应用GCN的思想也是很自然的。因此，提出了一种新的ResNet-GCN结构，如图5所示。删除了ResNet使用的原始瓶颈结构中的前两层，并用GCN模块替换它们。为了与原始数据保持一致，还在每个卷积层之后应用BN和ReLU。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557321729426-a9658a7a-6a73-4ff6-aa40-a3f43bf0a6e6.png#align=left&display=inline&height=469&name=image.png&originHeight=469&originWidth=627&size=55561&status=done&width=627)

将ResNet-GCN结构与原始ResNet模型进行比较。为了公平比较，仔细选择ResNet-GCN的大小，以便两个网络具有相似的计算成本和参数数量。附录中提供了更多详细信息。先在ImageNet 2015上预先训练ResNet-GCN并对PASCAL VOC 2012分割数据集进行微调。结果如表6所示。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557321766273-37563e2c-45f1-4fc1-98a7-c7f26dfb10c9.png#align=left&display=inline&height=275&name=image.png&originHeight=275&originWidth=632&size=67078&status=done&width=632)

注意这里使用ResNet50的模型来做比较，因为大型ResNet152的训练是非常费时的。从结果我们可以看出，基于GCN的ResNet作为ImageNet分类模型比原始ResNet略差。然而，在对细分数据集进行微调后，ResNet-GCN模型显着优于原始ResNet 5.5％。随着GCN和边界细化的应用，基于GCN的预训练模型的增益变得很小但仍然占优势。可以有把握地得出结论，无论在预训练模型还是特定于分段的结构中，GCN主要有助于提高分割性能。

### Pascal VOC 2012

- employ the Mi-crosoft COCO dataset to pre-train our model.
    - COCO has 80 classes and here we only retain the images including the same 20 classes in PASCAL VOC 2012.
- The training phase is split into three stages:
    - InStage-1, we mix up all the images from COCO, SBD and standard PASCAL VOC 2012, resulting in 109,892 images for training.
    - During the Stage-2, we use the SBD and standard PASCAL VOC 2012 images.
    - For Stage-3, we only use the standard PASCAL VOC 2012 dataset.
    - The input image is padded to 640x640 in Stage-1 and 512x512 for Stage-2 and Stage-3.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557322846933-3147a873-739f-42fb-858b-bf7211bb460e.png#align=left&display=inline&height=296&name=image.png&originHeight=296&originWidth=641&size=61924&status=done&width=641)

> Our GCN + BR model clearly prevails, meanwhile the post-processing multi-scale and denseCRF also bring benefits.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557322959315-ee57d759-4805-4ccb-a3c5-768b002dfdb8.png#align=left&display=inline&height=498&name=image.png&originHeight=498&originWidth=606&size=107656&status=done&width=606)

### Cityscapes

- Cityscapes is a dataset collected for semantic seg-mentation on urban street scenes. It contains 24998 images from 50 cities with different conditions, which belongs to 30 classes without background class.
- For some reasons, only 19 out of 30 classes are evaluated on leaderboard. The images are split into two set according to their labeling quality.
    - **5,000 of them are fine annotated while the other 19,998 are coarse annotated.**
    - The 5,000 fine annotated images are further grouped into **2975 training images,** 500 validation images and 1525 testing images. 
- The images in Cityscapes have a fixed size of 1024x2048, which is too large to our network architecture. Therefore we **randomly crop the images into 800x800 during training phase**.
-  We also increase k of GCN from 15 to 25 **as the final feature map is 25x25**.
- The training phase is split into two stages:
    - In Stage-1, we mix up the coarse annotated images and the training set, resulting in 22,973 images.
    - For Stage-2, we only finetune the network on training set.
- During the evaluation phase, we split the images into four 1024x1024 crops and fuse their score maps. 

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557323806423-14eb6ba2-66ca-448a-9d63-e169b498ff47.png#align=left&display=inline&height=242&name=image.png&originHeight=242&originWidth=648&size=43158&status=done&width=648)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1557323833173-11deead4-fc91-467d-ab65-59eecabeb5a9.png#align=left&display=inline&height=372&name=image.png&originHeight=372&originWidth=572&size=60401&status=done&width=572)

## 总结

根据对分类和分割的分析，发现大核对于缓解分类和定位之间的矛盾至关重要。在阐述大尺寸内核的原理的同时，提出了全局卷积网络。消融实验表明，提出的结构在有效的感受野和参数数量之间达到了良好的平衡，同时获得了良好的性能。为了进一步细化对象边界，提出了一种新颖的边界重细化模块。

定性地，GCN主要改善内部区域，而边界细化则提高边界附近的性能。

最佳模型在两个公共基准测试中实现了最先进的技术：PASCAL VOC 2012（82.2％）和Cityscapes（76.9％）。
