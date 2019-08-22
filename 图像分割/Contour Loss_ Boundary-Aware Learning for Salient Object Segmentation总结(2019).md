# Contour Loss: Boundary-Aware Learning for Salient Object Segmentation

- [Contour Loss: Boundary-Aware Learning for Salient Object Segmentation](#contour-loss-boundary-aware-learning-for-salient-object-segmentation)
  - [网络结构](#网络结构)
  - [Contour Loss](#contour-loss)
  - [Hierarchical Global Attention Module](#hierarchical-global-attention-module)
  - [实验细节](#实验细节)
  - [参考链接](#参考链接)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565358482108-6db04317-7aa7-47df-82b8-1b844682c931.png#align=left&display=inline&height=232&name=image.png&originHeight=209&originWidth=776&size=26875&status=done&width=862.2222450633114)

arxiv2019年8月9日21:48:09 目前还是放在arxiv上的, 具体链接在最后一节中.

这篇文章主要内容:

1. 提出一种轮廓损失, 利用目标轮廓来引导模型获得更具区分能力的特征, 保留目标边界, 同时也能在一定程度上增强局部的显著性预测.
1. 提出了一种层次全局显著性模块, 来促使模块逐阶段获取全局内容, 捕获全局显著性.

## 网络结构

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565359588865-275d37e8-3ed5-45b6-aa10-26f979acf966.png#align=left&display=inline&height=533&name=image.png&originHeight=480&originWidth=817&size=219022&status=done&width=907.7778018256771)

* 是一个FPN-like的结构
* Ei表示第i个编码器模块的输出特征, 所有的这些特征集合表示为FE
* 每个编码器都使用了一个残差结构来集成多尺度特征, 解码器将Ei转化为Resi, 所有这些特征集合表示为FR

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565360239282-2e09a0d9-a289-455b-b2cc-de1682b78a0b.png#align=left&display=inline&height=67&name=image.png&originHeight=60&originWidth=384&size=8152&status=done&width=426.66667796947365)

* 这里的$\delta$表示卷积层, $\theta$表示卷积层参数
* $\bigoplus$表示concat, $\{\star \}^{up \times 2}$表示上采样两倍
* 为了实现深监督, 这里对于每个Resi, 直接上采样到224x224, 获得Ui, 再通过使用sigmoid激活的卷积层来获得显著性预测Pi

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565360658214-ae5a49ab-c4ae-48be-a048-6e1247fda254.png#align=left&display=inline&height=60&name=image.png&originHeight=54&originWidth=390&size=6476&status=done&width=433.3333448127467)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565360710442-6f9c329d-7011-473c-8f10-933c76d8ec39.png#align=left&display=inline&height=59&name=image.png&originHeight=53&originWidth=399&size=5212&status=done&width=443.33334507765625)

* 训练时使用五个层级的损失, 但是对于**各级使用了不同的权重**

## Contour Loss

由于对于显著性目标检测(这里与"分割"无异)的每个样本的密集预测来说, 实际上在边界附近的像素可以看作是一些难样本, 参考Focal Loss的设计, 在交叉熵上使用空间加权, 来对显著性目标边界的像素的结果设置更高的权重. 空间权重可以表示为下式对应的集合.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565361243662-b17ccf54-59b4-4215-bbd4-28a8fa98a435.png#align=left&display=inline&height=37&name=image.png&originHeight=33&originWidth=385&size=4472&status=done&width=427.77778911001917)

* $(\star; S)^+$表示膨胀操作, $(\star; S)^-$表示腐蚀操作, 都是用的5x5的核S实施的
* K是一个超参数, 这里论文设定为5
* Gauss是为了赋予接近边界但是并没有位于边界上的像素一些关注, 这里对于Gauss的范围设置为5x5
* 这里的$\mathbb{1}$表示像素224x224这样的整个图上的像素, 是否是远离边界, 是的话就是1, 反之为0
* Compared with some boundary operators, such as Laplace operator, **the above approach can generate thicker object contours for considerable error rates**.
* 整体损失函数设置如下:

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565361805197-de995e68-1894-4d20-916c-c3b743e88f75.png#align=left&display=inline&height=73&name=image.png&originHeight=66&originWidth=402&size=6084&status=done&width=446.66667849929274)

* 这里论文里的描述应该有误, M是权重, Y是真值, Y*是预测
* 实际中, 对于式子(3)中的loss, 对应的就是(5)中对应层级的LC

## Hierarchical Global Attention Module

认为现有的显著性检测方法大多是基于softmax函数的:which enormously emphasizes several important pixels and endows the others with a very small value. Therefore these attention modules cannot attend to global contexts in high-resolution, which easily lead to overfitting in training.

因此, 这里使用了一个新的基于全局对比度的方法来利用全局上下文信息. 这里使用了特征图的均值来作为一个标准: Since a region is conspicuous infeature maps, each pixel in the region is also significant with a relatively large value, for example, over the mean. In other words, the inconsequential features often have a relatively small value in feature maps, which are often smaller than the mean. 于是使用特征图减去均值, 正值表示显著性区域, 负值表示非显著性区域. 于是可以得到如下的分层全局注意力:

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565362844790-557c2429-c219-49ca-86d5-a147fe42e021.png#align=left&display=inline&height=52&name=image.png&originHeight=47&originWidth=383&size=6392&status=done&width=425.5555668289282)

* 这里的FIn表示输入的特征, Aver和Var表示对应于FIn的均值和方差值
* $\lambda$表示一个正则项, 这里设定为0.1
* $\epsilon$是一个小数, 防止除零
* Compared with softmax results, the pixel-wise disparity of our attention maps is more reasonable, in other words, our attention method can retain conspicuous regions from feature maps in high-resolution

这里通过提出一个hierarchical global attention module (HGAM) 来捕获multi-scale global context.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565363657640-a6462709-997d-490a-8a3c-e06defe69b8c.png#align=left&display=inline&height=431&name=image.png&originHeight=388&originWidth=405&size=67806&status=done&width=450.0000119209293)

* 这里作为输入的有三部分: 来自本级的上采样特征Ui; 来自本级的编码器特征Ei, 以及来自上一级的HGAM信息Houti+1
* 为了提取全局上下文新西, 这里对于Ui使用了最大池化和平均池化处理, 获得H1和H2, 这里来自于[Cbam: Con-volutional block attention module]
* 对于Ei和Houti+1调整通道和分辨率分别获得H3和H4

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565364138804-a75f7e79-5188-4c83-ad89-d500b6a81d6b.png#align=left&display=inline&height=60&name=image.png&originHeight=54&originWidth=383&size=8955&status=done&width=425.5555668289282)

* HAtten可以通过(6)获得
* Houti用于生成下一个Houti+1, 而HAtten用来引导残差结构:

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565364259661-173fea05-3d02-446a-9a20-b1fa5d755e3a.png#align=left&display=inline&height=37&name=image.png&originHeight=33&originWidth=393&size=3468&status=done&width=436.6666782343832)

* 这里的$\bigodot$表示像素级乘法
* **所以最终的预测输出实际上是ResG1生成的P**

总体的训练损失改进为:

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565364436603-293c2fff-db56-4901-963c-377f6e123bf0.png#align=left&display=inline&height=83&name=image.png&originHeight=75&originWidth=382&size=5991&status=done&width=424.4444556883827)

## 实验细节

* Our experiments are based on the Pytorch framework and run on a PC machine with a single NVIDIA TITAN X GPU (with 12G memory).
* For training, we adopt DUTS-TR as training set and utilize data augmentation, which **resamples each image to 256×256 before random flipping, and randomly crops the 224×224 region**.
* We employ **stochastic gradient descent (SGD) as the optimizer with a momentum (0.9) and a weight decay (1e-4)**.
* We also set **basic learning rate to 1e-3** and finetune the **VGG-16 backbone** with a **0.05 times smaller learning rate**.
* Since the saliency maps of hierarchical predictions are coarse to fine from P5 to P1, we set the incremental weights with these predictions. Therefore WL5, ..., WL1 are set to 0.3, 0.4, 0.6, 0.8, 1 respectively in both Eq 3 and 9.
* The minibatch size of our network is set to 10. **The maximum iteration is set to 150 epochs with the learning rate decay by a factor of 0.05 for each 10 epochs**.
* As it costs less than 500s for one epoch including training and evaluation, the total training time is below 21 hours.
* For testing, follow the training settings, we also **resize the feeding images to 224×224, and only utilize the final output P**. Since the testing time for each image is 0.038s, our model achieves 26 fps speed with 224×224 resolution.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565362232418-d8e073ad-887c-4d55-b553-fef6552445bb.png#align=left&display=inline&height=551&name=image.png&originHeight=496&originWidth=856&size=137314&status=done&width=951.1111363069517)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565364995817-c12cc391-9433-4de2-b9d3-4f67e58bf9a8.png#align=left&display=inline&height=441&name=image.png&originHeight=397&originWidth=500&size=137015&status=done&width=555.5555702727522)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565365010107-0bd353c2-b8f3-4755-8e02-c5300e9ace4b.png#align=left&display=inline&height=338&name=image.png&originHeight=304&originWidth=495&size=103756&status=done&width=550.0000145700246)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1565365025407-9db3290c-c04b-477a-a751-0d0cb86bc801.png#align=left&display=inline&height=332&name=image.png&originHeight=299&originWidth=488&size=84323&status=done&width=542.2222365862061)

## 参考链接

* [https://arxiv.org/pdf/1908.01975.pdf](https://arxiv.org/pdf/1908.01975.pdf)
