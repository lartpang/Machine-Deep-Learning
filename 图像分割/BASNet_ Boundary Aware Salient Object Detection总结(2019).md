# BASNet: Boundary Aware Salient Object Detection

* [BASNet: Boundary Aware Salient Object Detection](#basnet-boundary-aware-salient-object-detection)
  * [网络结构](#网络结构)
  * [主要亮点](#主要亮点)
  * [损失函数](#损失函数)
  * [实验细节](#实验细节)
  * [相关链接](#相关链接)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1558924702986-22bd62e8-5e90-4e49-b609-1dda87ec1ce0.png#align=left&display=inline&height=132&name=image.png&originHeight=132&originWidth=679&size=21657&status=done&width=679)

## 网络结构

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1558751606548-c163fa5c-7faf-48ab-b081-d02bcfc52935.png#align=left&display=inline&height=507&name=image.png&originHeight=507&originWidth=1222&size=287094&status=done&width=1222)

## 主要亮点

- 深监督编解码器(sup1~sup8，可见前图）
- 额外的残差细化模块
- 使用了一种混合了BCE、SSIM（结构相似性）、IOU三种损失的混合损失
    - to supervise the training process of accurate salient object prediction on three levels: **pixel-level, patch-level and map-level**

## 损失函数

损失函数由三部分组成。一个是普通的交叉熵，代表着pixel-level的监督。

第二个则是使用了[结构相似性指标](https://zh.wikipedia.org/wiki/%E7%B5%90%E6%A7%8B%E7%9B%B8%E4%BC%BC%E6%80%A7)，这个指标利用的是区域数据进行的计算，最后计算一个均值。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1558753517845-790f0cea-4f82-4206-a5ec-aa8ee3aa8403.png#align=left&display=inline&height=60&name=image.png&originHeight=60&originWidth=409&size=7598&status=done&width=409)

> 结构相似性计算，mu表示区域的均值，sigmax和sigmay表示区域的标准差，而sigmaxy表示的是两个区域数据的协方差。在计算两张影像的结构相似性指标时，会开一个局部性的视窗，一般为N×N的小区块，计算出视窗内信号的结构相似性指标，每次以像素为单位移动视窗，直到整张影像每个位置的局部结构相似性指标都计算完毕。将全部的局部结构相似性指标平均起来即为两张影像的结构相似性指标。
>
> 结构相似性指标（英文：structural similarity index，SSIM index）是一种用以衡量两张数位影像相似程度的指标。当两张影像其中一张为无失真影像，另一张为失真后的影像，二者的结构相似性可以看成是失真影像的影像品质衡量指标。相较于传统所使用的影像品质衡量指标，像是峰值信噪比（英文：PSNR），结构相似性在影像品质的衡量上更能符合人眼对影像品质的判断。

文章中使用的结构相似性损失表达如下：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1558753753779-0b538276-9931-484b-820c-280342c92afd.png#align=left&display=inline&height=65&name=image.png&originHeight=65&originWidth=438&size=11532&status=done&width=438)

相似度越高，损失越低。文中设定：

- 每个区块大小为NxN
- C1=0.01^2
- C2=0.03^2

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1558754373623-34fcca18-e479-4b72-8835-920ddd0b1541.png#align=left&display=inline&height=595&name=image.png&originHeight=595&originWidth=570&size=104199&status=done&width=570)

关于结构相似性损失，文中有这么一段描述：
> SSIM丢失是一种 patch-level measure，它考虑每个像素的局部邻域。
> 它为边界分配较高的权重，即边界附近的损失较高，即使边界上的预测概率和前景的其余部分相同。在训练开始时，沿边界的损失是最大的（见图5的第二行）。它有助于优化专注于边界。随着训练的进行，前景的SSIM损失减少，背景损失成为主导词。然而，背景损失对训练没有贡献，直到背景像素的预测变得非常接近真值，其中损失快速从1下降到0。这是有帮助的，**因为预测通常在训练过程的后期接近于零，其中BCE损失变得平坦**。SSIM损失确保仍有足够的梯度来推动学习过程。
> 由于概率被推到零，因此背景预测看起来更清晰。

使用的IOU损失如下：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1558753987241-107e5068-cbba-4f4d-8cc8-57838e2d1e34.png#align=left&display=inline&height=105&name=image.png&originHeight=105&originWidth=438&size=12448&status=done&width=438)

> S中的数据是显著性概率，而G中是只有0/1。

## 实验细节

- We train our network using the DUTS-TR dataset, which has 10553 images.
    - Before training, the dataset is augmented by horizontal flipping to 21106 images.
    - During training, each image is first resized to 256x256 and randomly cropped to 224x224.
- Part of the encoder parameters are initialized from the **ResNet-34** model. Other convolutional layers are initialized by Xavier.
- We utilize the Adam optimizer to train our network and its hyper parameters are set to the default values, where the **initial learning rate lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight decay=0**.
- We train the network until the loss converges **without using validation set**.
- The training loss converges after 400k iterations with a batch size of 8 and the whole training process takes about **125 hours**.
- During testing, the input image is resized to 256x256 and fed into the network to obtain its saliency map.
    - Then, the saliency map (256x256) is resized back to the original size of the input image.
    - Both the resizing processes use bilinear interpolation. 

![](https://cdn.nlark.com/yuque/0/2019/png/192314/1558755208807-19f46e9b-4ad0-49e8-a603-38202315c82c.png#align=left&display=inline&height=414&originHeight=414&originWidth=595&status=done&width=595)

> 这里的relaxF是文章中用来评估边界的一个指标。
> 首先使用0.5阈值将预测结果二值化，通过使用膨胀技术处理该结果后得到的结果与原始二值图异或操作，获得对应的边界图，类似可以获得真值的边界图。从而计算Fmeasure。注意，这里的计算Fmeasure的时候，使用的p=relaxPrecision和r=relaxRecall有些特殊处理：**relaxPrecision定义为预测边界像素中，位于真值边界像素的ρ个像素范围内的像素比例；relaxRecall定义为真值边界像素中，位于预测边界像素的ρ个像素范围内的像素比例。**
> 实验中设定ρ=3。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1558755862895-6da57d63-2fcf-4b39-9b28-0c4e99a5a71c.png#align=left&display=inline&height=353&name=image.png&originHeight=353&originWidth=579&size=85669&status=done&width=579)
可见这里的结构相似性损失效果不错。
![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1558756081009-6d5477d9-30c4-4255-8470-b7601abfe18b.png#align=left&display=inline&height=495&name=image.png&originHeight=495&originWidth=1199&size=255934&status=done&width=1199)

## 相关链接

- 论文：[https://webdocs.cs.ualberta.ca/~xuebin/BASNet.pdf](https://webdocs.cs.ualberta.ca/~xuebin/BASNet.pdf)
- 论文补充材料：[https://pdfs.semanticscholar.org/cc79/4b54461daee43eee1add6d099733033a5d42.pdf](https://pdfs.semanticscholar.org/cc79/4b54461daee43eee1add6d099733033a5d42.pdf)
- 代码：[https://github.com/NathanUA/BASNet](https://github.com/NathanUA/BASNet)
