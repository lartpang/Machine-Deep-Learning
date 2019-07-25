# Knowledge Adaptation for Efficient Semantic Segmentation

- [Knowledge Adaptation for Efficient Semantic Segmentation](#knowledge-adaptation-for-efficient-semantic-segmentation)
  - [主要工作](#主要工作)
  - [网络结构](#网络结构)
  - [Auto-Encoder](#auto-encoder)
  - [Feature Adapter](#feature-adapter)
  - [Affinity Distillation Module](#affinity-distillation-module)
  - [Train Process](#train-process)
  - [实验细节](#实验细节)
  - [一些思考](#一些思考)
  - [参考链接](#参考链接)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563886634045-41dfc16d-7f3c-4cca-9cb5-68754a856f4a.png#align=left&display=inline&height=168&name=image.png&originHeight=168&originWidth=960&size=31827&status=done&width=960)

CVPR2019使用蒸馏做语义分割的一篇文章。

## 主要工作

针对语义分割任务对传统蒸馏方法进行了一些改进

- 使用自编码器来使得教师网络的知识更易学习
- 使用关联适配模块来使得学生模型在学习教师知识的时候可以看到更多，提取更多的远程关联信息
- 添加了了特征适配器来实现教师与学生特征之间的适配，促进更好的学习
- 针对不同的结构设计了不同的损失来进行监督

对于目前在语义分割的工作中，模型一般会使用较大的分辨率的特征进行计算，导致网络参数量和计算复杂度太大，限制了实际的应用与发展，所以逐渐有工作开始探究如何处理这样的问题。一个思路是降低特征的分辨率，但是这会导致大量细节信息的缺失，不利于预测的准确性。如何解决这个困境，在准确性和效率之间找到更好的平衡，已经讨论了很长时间。被Hinton提出的知识蒸馏方法，因为其简单而有效的表现，收到了很多关注。

> The knowledge in KD is defined as soft label output from a large teacher network, which contains more useful information, **such as intraclass similarity**, than one-hot encoding. The student network is supervised by both soft labels and hard one-hot labels simultaneously, reconciled by a hyper-parameter to adjust the loss weight.
>
> Following KD, many methods are proposed to regulate the intermediate features. However, these methods are **mainly designed for the image-level classification task without considering the spatial context structures**. Moreover, in the semantic segmentation task, the feature maps from the teacher and student usually **have inconsistent context and mismatched features**. Thus these methods are improper to be used for semantic segmentation directly. 


与本文有关的主要工作文章分析了三篇：

1. Distilling the Knowledge in a Neural Network
   1. 开创性工作，主要针对图像分类任务
   2. 有一定的局限：
      1. FitNet的作者试图迫使学生在决策空间中模拟教师网络的输出分布，其中有用的上下文信息是级联的
      2. 对于图像级分类任务，需要的知识对于两个模型来说是相似的，但是语义分割的决策空间可能不同，因为两个模型有着不同的捕获远程和上下文依赖关系的能力，这依赖于网络的结构
      3. 温度超参数对于任务而言是敏感的，并且很难调整，尤其是对于大的基准数据集。
2. FitNets: Hints for Thin Deep Nets
   1. 对于学习中间表示，而直接使用对齐特征图的方法，这对忽略两个模型之间的内在差异（如空间分辨率、通道数和网络架构）来说，可能不是一个好的选择
   2. 同时，两种模型的抽象能力存在显著差异，可能会使这种情况更加严重
3. Paying More At-tention to Attention: Improving the Performance of Convo-lutional Neural Networks via Attention Transfer
   1. 旨在模拟学生和老师模型之间的注意力图，这基于假设：特征图沿通道维度的加和在图像分类任务中可能表示注意力分布
   2. 然而，这种假设可能不适合像素级的分割任务，因为不同的通道表示不同类的激活，简单地对通道进行求和就会得到混淆了的注意力图

## 网络结构

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563886701137-98a0dc2c-5e97-4e87-a4b6-2f454ecedea1.png#align=left&display=inline&height=592&name=image.png&originHeight=592&originWidth=1065&size=182534&status=done&width=1065)

从图中可以看出文章的几个关键结构：

1. Auto-Encoder
1. Feature Adapter
1. Affinity Distillation Module

关于文中反复提到的“知识”的表达，主要定为两个方面：

1. The first part is designed for translating the knowledge from the teacher network to a compressed space that is more informative. The translator is achieved by training an autoencoder to compress the knowledge to a compact format that is easier to be learned by the student network, otherwise much **harder due to the inherent structure differences**.
1. The second part is designed to **capture long-range dependencies from the teacher network**, which is difficult to be learned for small models due to the limited receptive field and abstracting capability. 小网络自身很难提取远程依赖关系，进而通过老师网络来进行学习。

## Auto-Encoder

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563890025897-a9fee55f-9400-45c0-8cae-920a98189401.png#align=left&display=inline&height=222&name=image.png&originHeight=222&originWidth=171&size=14426&status=done&width=171)

用来重构教师模型的特征，以获得更为有用和重要的信息。通过训练一个自编码器，来挖掘显式的结构信息，并将老师的知识“简化”，转换成对于学生网络更容易理解复制的知识。与低、中阶特征相比，高阶特征更适合我们的实际情况。低、中阶特征由于固有的网络差异，一般在在不同的模型中普遍存在或者而难以传输。

> Directly transferring the outputs from teacher overlooks the **inherent differences of network architecture between two models**. Compact representation, on the other hand, can help the student focus on the most critical part by removing redundancy knowledge and noisy information. 

在提出的模型中，自编码器将来自教师模型的最后的卷积层特征作为输入，其中使用了三个非一跨步的卷积层以及对称的转置卷积层。
这里对于自编码器训练的重构损失可以表达为：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563889400504-55e2aea8-9e17-4d91-9304-df3537372d6e.png#align=left&display=inline&height=33&name=image.png&originHeight=33&originWidth=515&size=4797&status=done&width=515)

- E和D表示的就是对应的编码器和解码器
- alpha对于所有试验设定为10e-7
- ![](https://cdn.nlark.com/yuque/__latex/664fd38f237431420dc024098bf29753.svg#card=math&code=%5CPhi_t&height=24&width=19)表示教师网络的最后的特征图，![](https://cdn.nlark.com/yuque/__latex/83517399526725bd8cc255e06a572888.svg#card=math&code=%5CPhi_s&height=24&width=20)表示学生网络的最后的特征图


在训练auto-encoder的过程中，一个常见的问题是模型可能只学习一个恒等函数，这意味着**提取的结构知识更有可能与输入特性共享相同的模式**。这里对损失添加了一个l1范数来进行约束，因为l1范数可以产生一个稀疏表达。

> 这里的策略类似于_Babajide O. Ayinde and Jacek M. Zurada. Deep Learning of Constrained Autoencoders for Enhanced Understanding of Data. IEEE Trans. Neural Netw. & Learn. Syst., 2018._ 中用来正则化权重和重表示（re-represented）空间的策略。

## Feature Adapter

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563890043867-ad206c1a-3467-4bde-9655-2cbacd8974e3.png#align=left&display=inline&height=56&name=image.png&originHeight=56&originWidth=82&size=2377&status=done&width=82)

为了解决特征不匹配问题和降低教师模型与学生模型之间存在的固有网络差异造成的影响，通过添加一个卷积层来作为特征适配器（feature adapter）。这里在整体训练的时候会设置一个损失函数，进而对知识迁移过程进约束：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563890169277-5cd8417f-f889-41bf-a53a-da599cbf8d0d.png#align=left&display=inline&height=72&name=image.png&originHeight=72&originWidth=509&size=10749&status=done&width=509)

- 这里的E表示自编码器
- I表示对于所有位置上对于student-teacher的成对信息的索引（也就是“所有位置集合“）
- ![](https://cdn.nlark.com/yuque/__latex/2df74df310cb7f4878ec1bf0678dff59.svg#card=math&code=C_f&height=24&width=21)是对于学生网络特征设置的适配器，这里使用了一个![](https://cdn.nlark.com/yuque/__latex/0d7bd8d05c82067f4a72d909599a3e89.svg#card=math&code=3%20%5Ctimes%203&height=24&width=38)核大小，步长为1，padding为1，包含BN和ReLU的卷积层
- 特征在匹配之前进行归一化
- 这里的p和q是不同的范数类型以归一化知识，来提升稳定性

## Affinity Distillation Module

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563932142463-0e510a46-c618-470c-ac5a-7180eb5d6cc7.png#align=left&display=inline&height=98&name=image.png&originHeight=98&originWidth=78&size=3832&status=done&width=78)

由于小网络缺乏抽象能力，在学习知识的过程中能力有限，这也导致其在捕获远程依赖关系的能力上有限。而捕获远程依赖关系对于语义分割任务而言是重要且有益的。文章因此针对小网络在学习大网络的知识的过程中，设置了一个新提出的affinity distillation module，借此来提取来自大的教师模型特征中的远程非局部依赖关系。

在学习中，有时通过提供额外的差异信息或相关联的信息来学习新知识会更有效。这里在网络中定义了关联信息，通过直接计算任意特征位置之间的交互，而不考虑其空间距离。这样，对于有些特征信息相近的像素位置可以得到较高的响应结果，而对于差异较大的结果会得到较低的响应结果。（这里相比**余弦距离（向量空间中两个向量夹角的余弦值）**多了一些数值大小关系上的考虑）

这里实际上设计计算关联矩阵的时候，使用的还是余弦距离（注意这里表示的是A矩阵中对应于位置(i,j)处的值的计算，也就是得到的是一个1x1的值）：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563893679266-0d604266-a740-4b88-a907-59d75669efd4.png#align=left&display=inline&height=60&name=image.png&originHeight=60&originWidth=513&size=6108&status=done&width=513)

---

余弦距离，可以看出就是向量点积或內积除以各自的L2范数，也就是各自的长度、模：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563893692841-f0b3dd89-b2e3-4011-8e05-7e58c6e1f86c.png#align=left&display=inline&height=62&name=image.png&originHeight=62&originWidth=420&size=3775&status=done&width=420)

---

- 这里的![](https://cdn.nlark.com/yuque/__latex/571ca3d7c7a5d375a429ff5a90bc5099.svg#card=math&code=%5Ccdot&height=24&width=5)运算应该表示的是点积，这样才能由两个1xc的矢量计算得到一个的1x1的值
- 这里的A整体表示的是使用**空间归一化以后的关联矩阵**，是一个mxm的矩阵，表示两个hxw的特征图上任意点之间的关联
- 这里的![](https://cdn.nlark.com/yuque/__latex/2f51310acab41649af988ccebfe4186d.svg#card=math&code=%5CPhi&height=24&width=12)表示最后一层特征图，大小为hxwxc，对应前面的m=hxw，所以实际上是将两个特征图向量化了
- 教师和学生都有自己的一个关联矩阵

这里使用l2损失来约束教师与学生之间的关联矩阵，定义如下：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563894336223-8dd87bde-38e1-4d4a-ac2c-22c810ae08f9.png#align=left&display=inline&height=64&name=image.png&originHeight=64&originWidth=512&size=6564&status=done&width=512)

- 这里的E(Phit)表示转译后的教师知识
- Ca是对于学生关联的适配器
- i索引着特征图的位置
- 也就是两个关联矩阵
- 这里计算的就是MAE：[https://www.jiqizhixin.com/articles/2018-06-21-3](https://www.jiqizhixin.com/articles/2018-06-21-3)

## Train Process

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563894529063-225661fe-640a-4647-896e-de47c58c0c68.png#align=left&display=inline&height=239&name=image.png&originHeight=239&originWidth=523&size=46548&status=done&width=523)
有几个关键点：

1. 教师模型事先是训练好的（**是从哪些数据集上训练的呢？和学生模型训练使用的是一样的数据么？**）
1. 学生网络使用了三个损失来进行监督，使用权重系数beta和gamma（**试验中被设置为50和1**）加权获得最终的整体：
   1. 使用真值标签监督的交叉熵损失Lce
   2. 适配损失Ladapt
   3. 关联转换损失Laff
2. WE、WD、WS表示编码器、解码器和学生模型的参数
3. 要注意整体结构图，虽然训练了一个自编码器，但是实际在训练学生模型的时候，使用的仅是其编码器部分，所以第一阶段主要寻找一个最优的编码器

## 实验细节

- Training teacher network
    - To demonstrate the effectiveness of our method, we select two totally different teacher models, **ResNet-50 and Xception-41**. Both atrous convolution and atrous spatial pyramid pooling (ASPP) are utilized to obtain a series of feature maps with large size. 
    - We use mini-batch **stochastic gradient descent (SGD) with batch size 16 (at least 12), momentum 0.9, and weight decay 4×10e−5 in training**. Similar to [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation], we apply the **poly learning rate strategy with power 0.9**. The initial learning rate is 0.007.
    - General data augmentation methods are also used in network training, such as randomly flipping the images and randomly performing scale jitter.
        - For the Pascal VOC dataset, the training process can be split into two steps.First, we train 300K iterations on the COCO dataset, then 30K iterations on the _trainaug_ dataset.
        - For theCityscapes dataset, we do not pre-train our model on the COCO dataset for fair comparison. We train 90K iterations on the _train-fine_ dataset, which is fine tuned on _trainval_ and _train-coarse_ to evaluate on _test_ dataset.
        - For the Pascal Context dataset, the COCO dataset is not used for pre-training. 30k iterations are trained on the _train_ set and evaluated on the _val_ set.
- Training auto encoder.
    - We finished the auto-encoder training within one epoch with a learning rate of 0.1.
    - **Large weight decay of 10e−4 is used to attribute low energy to a smaller portion of the input points.**
- Training the whole system.
    - Most of the training parameters are similar to the process of training the teacher network, except that our **student network does not involve the ASPP and the decoder, which are exactly the same with [MobileNetV2: Inverted Residuals and Linear Bottlenecks]**.
    - With the help of atrous convolution, low resolution feature maps are generated.
    - During the training process, **the parameters of the teacher net WT and the parameters for auto-encoder WE are fixed without updating**. 

对于关联矩阵A中的特定位置进行可视化：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563933969014-a20bf136-a0d9-4593-9b05-2cf0d7ae8189.png#align=left&display=inline&height=634&name=image.png&originHeight=634&originWidth=525&size=359072&status=done&width=525)

可以看出来，通过添加关联蒸馏模块，使得对于全局的响应更为明确和清晰。上图中绘制的彩图，是全图关于原图红色加号位置得到的响应情况。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563934239550-96f68ef2-971f-4083-8b63-aa5978885467.png#align=left&display=inline&height=351&name=image.png&originHeight=351&originWidth=523&size=59942&status=done&width=523)

因为如果两个模型具有不同的输出特征，关联矩阵会不匹配。为了显示单个关联模块的效果，**我们将特征映射的大小调整为相同的尺寸（resize）**。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563934767983-45c64e42-e743-405d-9303-6422341360de.png#align=left&display=inline&height=526&name=image.png&originHeight=526&originWidth=517&size=66895&status=done&width=517)

在图1中可以找到不同输出步幅设置的比较。其中**16s模型甚至比仅使用8％FLOPS的4s输出的基线模型更好，而不引入额外的参数**。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563934380168-55322a43-8278-4673-a995-1882e9c0bf7a.png#align=left&display=inline&height=343&name=image.png&originHeight=343&originWidth=525&size=57691&status=done&width=525)

这是其他的一些验证试验。In this experiment, we make comparisons with other knowl-edge distillation methods: KD [11] and FitNet [20],

1. KD is designed for image-level classification. The knowledge defined in KD is the soft label output by a teacher network. The soften degree is controlled by a hyper-parameter temperature t, which has a significant influence on the distillation and learning processes. **We set t to 2, 4, 6. To make fair comparisons, we bilinearly upsample the logits map to the size of the teacher network**.
1. FitNet, different from KD, tries to match the intermediate representation between two models. **But this requires similar network design**. In our experiments, we **directly upsample the feature map of the last layer and add a ℓ2 loss**. The loss curve is shown in Figure 5.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563934915858-601c4387-e73a-494f-bcaf-c470c4ebc988.png#align=left&display=inline&height=455&name=image.png&originHeight=455&originWidth=513&size=64320&status=done&width=513)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563934871756-1d09a804-d208-4ec5-bf50-dce2adfacc2a.png#align=left&display=inline&height=315&name=image.png&originHeight=315&originWidth=523&size=39788&status=done&width=523)

Our method achieves better performances than KD, with **all the hyper-parameters fixed across all experiments and datasets**. Our method also outperforms FitNet by 1.2 points, **indicating that the knowl-edge defined by our method alleviates the inherent difference of two networks**.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563934939666-3db1cc7c-b574-4852-be50-7ea007eba42e.png#align=left&display=inline&height=286&name=image.png&originHeight=286&originWidth=532&size=37722&status=done&width=532)

Compared with the traditional methods, the qualitative segmentation results in Figure 4 visually demonstrate **the effectiveness of our distillation method for objects that require more context information, which is captured by our proposed affinity transfer module.** 

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563935553828-5d601e99-3f05-40fc-8957-4a008a223609.png#align=left&display=inline&height=532&name=image.png&originHeight=532&originWidth=1059&size=389699&status=done&width=1059)

On the other hand, the knowledge translator and adapter reduce the loss of the detailed information and produce more consistent and detail-preserving predictions, as shown in Figure 6. 

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563934990672-4c5fda2a-694a-4884-b01d-1440e2ecb7cf.png#align=left&display=inline&height=261&name=image.png&originHeight=261&originWidth=529&size=37212&status=done&width=529)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563935014075-cfcf206b-6021-43d8-8521-11db19a3d403.png#align=left&display=inline&height=431&name=image.png&originHeight=431&originWidth=523&size=63178&status=done&width=523)

## 一些思考

* 文章中关于模型如何生成最终的预测没有讲太多，有可能是直接上采样？
* 这篇文章比较实验的设计
    * 主要针对的是一些轻量模型
    * 自身的消融实验
    * 主要比较了运算量和参数以及相关的指标

## 参考链接

- 论文链接：[http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Knowledge_Adaptation_for_Efficient_Semantic_Segmentation_CVPR_2019_paper.pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Knowledge_Adaptation_for_Efficient_Semantic_Segmentation_CVPR_2019_paper.pdf)
- 基于知识蒸馏的模型压缩和加速：[https://www.cnblogs.com/Libo-Master/p/9669334.html](https://www.cnblogs.com/Libo-Master/p/9669334.html)
