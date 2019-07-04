# SE2Net: Siamese Edge-Enhancement Network for Salient Object Detection

- [SE2Net: Siamese Edge-Enhancement Network for Salient Object Detection](#se2net-siamese-edge-enhancement-network-for-salient-object-detection)
  - [网络结构](#%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84)
  - [主要贡献](#%E4%B8%BB%E8%A6%81%E8%B4%A1%E7%8C%AE)
  - [迭代过程](#%E8%BF%AD%E4%BB%A3%E8%BF%87%E7%A8%8B)
  - [边缘引导](#%E8%BE%B9%E7%BC%98%E5%BC%95%E5%AF%BC)
  - [实验细节](#%E5%AE%9E%E9%AA%8C%E7%BB%86%E8%8A%82)
    - [消融实验](#%E6%B6%88%E8%9E%8D%E5%AE%9E%E9%AA%8C)
    - [和其他模型比较](#%E5%92%8C%E5%85%B6%E4%BB%96%E6%A8%A1%E5%9E%8B%E6%AF%94%E8%BE%83)
  - [一点感想](#%E4%B8%80%E7%82%B9%E6%84%9F%E6%83%B3)
  - [参考链接](#%E5%8F%82%E8%80%83%E9%93%BE%E6%8E%A5)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562139082006-1dab8d65-8446-418c-9574-5c6e3434bbbf.png#align=left&display=inline&height=242&name=image.png&originHeight=242&originWidth=1189&size=55582&status=done&width=1189)

## 网络结构

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562139265497-f771a666-d8cd-4191-908a-36b13ae28569.png#align=left&display=inline&height=536&name=image.png&originHeight=536&originWidth=1229&size=122797&status=done&width=1229)

- 从图中可以看出来，使用了多个阶段进行迭代计算，这里是仅为结构相同的双分支“siamese”网络，权重参数并不共享。
- 从backbone中前三个阶段的预测结果送入边缘分支，后两个阶段的预测结果送入区域分支。一路低级特征信息，一路高级特征信息。
- 每个阶段输出一个由边缘分支和区域分支**分别生成的特征二者融合得到的特征，获得一个在边缘有着更强相应的区域特征图**。最终的输出使用最后一个阶段的预测结果。
- 第2个阶段之后的特征输入都是包含之前阶段的输出以及对应的原始低级或者高级特征信息。
- 每个阶段的输出都会计算损失，最终的损失包含了所有位置的损失。这里使用的损失不是常用的交叉熵函数，而是自己构造的L2损失。
- 最终还使用了提出的边缘引导推理算法来对于测试时预测的结果进一步的细化。

## 主要贡献

- 双分支孪生多阶段迭代网络
- 边缘与区域双重监督，信息互补
- 使用自定义L2损失同时兼顾了点与区域的信息
- 提出一种边缘引导推理算法来优化预测结果

## 迭代过程

In the training process,

- we first pass each image through a backbone network, i.e., VGG16, ResNet50 or ResNext101, to generate a set of feature maps.
- As a result, five scales of feature maps, namely 1, 1/2, 1/4, 1/8 and 1/16 of the input size, are computed to generate the low-level and high-level features.
- In particular, **the first three scales of feature maps are concatenated to generate the low-level features Li**, and **the last two scales of feature maps are concatenated to generate the high-level features Hi**. 
- 迭代过程如式子1和2，注意第一阶段（t=1）输入与后面不同，这里表达的是后几个阶段的过程，具体参数结构如下表：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562154990778-cd740f86-92ae-4539-a6a3-7b5b54124eb6.png#align=left&display=inline&height=92&name=image.png&originHeight=92&originWidth=511&size=10839&status=done&width=511)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562155397541-ce0bc9d3-1c1a-4a91-b4f8-ca0c220ffd26.png#align=left&display=inline&height=592&name=image.png&originHeight=592&originWidth=530&size=75417&status=done&width=530)

- Given a new testing image, our siamese multi-stage net-work can predict a set of salient maps of edges and regions. In general, the quality of salient maps is consistently improved over the stages, therefore one can directly take predictions at the last stage as the final results. What’s more, we **further design a simple yet effective fusion network to fuse the predictions from all stages.** 最终的预测结果可以表示如下，下面通过两个融合分支分别融合不同阶段的边缘信息或区域信息，得到最终的预测结果，示意图如图3：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562154952417-6cf6b169-055c-4451-b1a3-8741ce1b56a3.png#align=left&display=inline&height=100&name=image.png&originHeight=100&originWidth=571&size=9392&status=done&width=571)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562155304656-1dee044f-0d5a-44b7-bcd8-4bfcec3e2204.png#align=left&display=inline&height=248&name=image.png&originHeight=248&originWidth=519&size=54753&status=done&width=519)

损失计算

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562155439998-9892ff12-4cab-4fc9-9a40-c0ed89df6341.png#align=left&display=inline&height=175&name=image.png&originHeight=175&originWidth=501&size=19999&status=done&width=501)

这里引入了一个加权L2损失。式子中标有 `*` 的表示对应于边缘和区域真值图上的值。这里的N表示x位置周围的邻域，其邻域的有效范围半径为rou，是一个自定义参量。可见式子7。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562156249459-d6343c76-292d-4b0d-8651-bbfb8c78216e.png#align=left&display=inline&height=118&name=image.png&originHeight=118&originWidth=519&size=12192&status=done&width=519)

这里的K是一个截断高斯核函数，有着手工设定的方差sigma。相当于是利用距离进行了一个加权。

The advantage of our weighted L2 loss function over the standard one is that, **it considers the regression problem in a local neighborhood, therefore the learned maps are robust to the salient object annotations**. Finally, we **extend our weighted L2 loss function into all the training samples and all the network stages**, then the overall objective function can be formulated as follows: 

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562156588475-9fdc0fd3-36ad-4d94-ad66-66ba2686b273.png#align=left&display=inline&height=67&name=image.png&originHeight=67&originWidth=507&size=7628&status=done&width=507)

## 边缘引导

Although the DNN based methods can usually obtain the high-quality masks of salient objects, the resulting salient maps may be not very smooth or precise at the output layers. **Therefore, post-processing algorithms are usually needed to further boost the final results.**

Novel to most of the salient object detection methods, since our network can jointly predict the salient maps of edges and regions, **we developed a novel edge-guided inference algorithm to filter small bumpy regions along the predicted edges.**

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562157465907-51b4ba9a-5ce4-4cf1-81c8-6086030b38e9.png#align=left&display=inline&height=271&name=image.png&originHeight=271&originWidth=520&size=79809&status=done&width=520)

具体流程如下：

- 首先生成一系列5×5大小的矩形框B，其中各个框的中心都位于边预测图上的边上，而且所有的K个框，除了第一个和第K个有重叠外，其余框之间相接但是不相交。
- 各个框根据边预测图，被内部的边分成了两部分，如第k个框划分结果为：B={B，B}。在分出来的两个区域中，根据区域预测图中，对应的显著性区域和非显著性区域的占比对该划分区域的预测结果进行重新赋值。主要基于观察：对于一块区域而言，内部的显著性区域大于非显著性区域的话，该区域就很有可能为显著性区域，反之亦然。主要的处理思路如式子10，这里![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562159493934-d20e5744-acf6-4a5c-9ed2-117e3e72262f.png#align=left&display=inline&height=42&name=image.png&originHeight=42&originWidth=118&size=3680&status=done&width=118)，计算了该划分区域B中的显著性成分与非显著性成分之间的比值。从而依据此来对该区域进行判定。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562159366411-31b4d08b-5fab-4b9c-99d7-c12585d0baff.png#align=left&display=inline&height=97&name=image.png&originHeight=97&originWidth=521&size=8018&status=done&width=521)Because we further use the edge information to help refine the region masks, we **name it as the edge-guided inference algorithm.**

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562159641365-36f24089-89d2-4e2a-9612-387a218a4e5f.png#align=left&display=inline&height=498&name=image.png&originHeight=498&originWidth=531&size=67758&status=done&width=531)

这里没有细讲如何更新。个人感觉就是使用更新后的B来对R内的显著性区域进行调整，使其更贴合边界。而且上述伪代码中的第8行中的B的上标应该是k。

## 实验细节

-  We used the DUTS dataset to train our SE2Net network.
  - The DUTS dataset is a latest released challenging dataset that contains 10,553 training images and 5,019 testing images in very complex scenarios.
  - As indicated in [10], a good salient object detection model should work well over almost all datasets, therefore we also evaluated our model on the other five datasets, i.e., the ECSSD, SOD, DUT-OMRON, THUR 15K and HKU-IS, which contains 1,000, 300, 5,168, 6,232, and 4,447 natural images, respectively.
  - In each image, there are different numbers of salient objects with diverse locations. For fair comparison, we follow the same data partition as in [Salient Object Detection via High-to-Low Hierarchical Context Aggregation].
  - Our SE2Net requires the annotations of both edges and regions, while the existing datasets can only provide the ground truth of regions. We generate the ground truth of edges in a very simple and cheap two-step approach:
    - Generate the edge annotations from the ground truth of regions by **using the Canny operator**; 
    - **Dilate the width of each edge annotation to five pixels**.
  - The **batch size is set to be 10**, the learning rate is initialized to be 0.01 and **decreased by a factor of 0.1 at every two epochs**.
  - In the training process, we first randomly crop 300×300 from input images, then follow a **random horizontal flipping** for data augmentation.
  - There are two hyper parameters in our weighted L2 loss function, and we set **ρ=3** and **sigma=0.01** in all the experiments.

### 消融实验

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562160695865-660572f1-8b63-44a3-8658-9172853f1b9d.png#align=left&display=inline&height=272&name=image.png&originHeight=272&originWidth=1081&size=70876&status=done&width=1081)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562160747012-35ae7bff-cd3e-4055-a651-e779b003ddd4.png#align=left&display=inline&height=365&name=image.png&originHeight=365&originWidth=1084&size=94249&status=done&width=1084)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562160714250-d6eb98df-110e-42cb-a277-61eba7701cd1.png#align=left&display=inline&height=361&name=image.png&originHeight=361&originWidth=508&size=42050&status=done&width=508)

从这里可以看出来，边缘分支的作用明显。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562160733611-ef5ff107-6e97-41b4-b5e8-0cef367358ee.png#align=left&display=inline&height=363&name=image.png&originHeight=363&originWidth=500&size=43760&status=done&width=500)

### 和其他模型比较

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562160953312-5e1dd063-1954-47b1-aa15-70860f326b4b.png#align=left&display=inline&height=490&name=image.png&originHeight=490&originWidth=1068&size=130835&status=done&width=1068)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562161011414-c3d27148-b06e-401d-b5bb-cb96b1f4482e.png#align=left&display=inline&height=154&name=image.png&originHeight=154&originWidth=539&size=32934&status=done&width=539)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562161080737-12ddca65-668a-4db3-b2fe-3e4b0e399a12.png#align=left&display=inline&height=205&name=image.png&originHeight=205&originWidth=540&size=24415&status=done&width=540)

## 一点感想

- 结构上：
    - 和R3Net、PFA的结构是一致的，都是利用早期几层生成低级特征，后期几层生成高级特征。这里倒是结构处理的方式与R3Net更像，但是R3Net使用了一种残差式的迭代模块，而这里使用的是两个相同结构的分支，二值之间进行交互融合迭代。虽说是迭代，不过代码实现应该各个阶段是独立的代码。
    - 另外，这也为除了FCN/UNet结构之外提供了一种构造网络的思路。
- 损失函数上：
    - 这里没有使用常用的交叉熵函数，可见交叉熵函数不是必须，按需求尝试。
    - 这里设计损失函数不只考虑了点之间的关系，还涉及到了与邻域的计算：The advantage of our weighted L2 loss function over the standard one is that, **it considers the regression problem in a local neighborhood, therefore the learned maps are robust to the salient object annotaion. 个人理解是这里不仅仅学习点之间的关联，而且在一定程度上学习了一个分布的关系。**
- 对于边缘信息的利用：
    - 这里使用边缘信息的方式很特别，使用其与趋于信息不断的融合，利用融合后的信息再分别生成边缘与区域特征，这一点想法上有些类似[MLMSNet](https://www.yuque.com/lart/papers/ph0c35)，都是反复的利用边缘与区域信息互相促进，但是后者是级联型结构，这里是并行式结构。
    - 边缘信息是否可以直接融合到预测上，直接增强预测的边缘的信息？

## 参考链接

- 论文：[https://arxiv.org/pdf/1904.00048.pdf](https://arxiv.org/pdf/1904.00048.pdf)
