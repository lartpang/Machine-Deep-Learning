# Detect Globally, Refine Locally: A Novel Approach to Saliency Detection

- [Detect Globally, Refine Locally: A Novel Approach to Saliency Detection](#Detect-Globally-Refine-Locally-A-Novel-Approach-to-Saliency-Detection)
  - [Recurrent Localization Network (RLN)](#Recurrent-Localization-Network-RLN)
    - [Contextual Weighting Module (CWM)](#Contextual-Weighting-Module-CWM)
    - [Recurrent Module (RM)](#Recurrent-Module-RM)
  - [Boundary Refinement Network (BRN)](#Boundary-Refinement-Network-BRN)
  - [实验细节](#%E5%AE%9E%E9%AA%8C%E7%BB%86%E8%8A%82)
  - [相关链接](#%E7%9B%B8%E5%85%B3%E9%93%BE%E6%8E%A5)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562495326959-bca89ade-7ffd-492c-bc8a-e3fbc03dc6fb.png#align=left&display=inline&height=346&name=image.png&originHeight=346&originWidth=1440&size=83109&status=done&width=1440)

Effective integration of contextual information is crucial for salient object detection. To achieve this, most existing methods based on ’skip’ architecture mainly focus on how to integrate hierarchical features of Convolutional Neural Networks (CNNs). **They simply apply concatenation or element-wise operation to incorporate high-level semantic cues and low-level detailed information. However, this can degrade the quality of predictions because cluttered and noisy information can also be passed through.** 它们只是应用连接或元素操作来合并高级语义提示和低级细节信息。然而，这会降低预测的质量，因为也可以传递杂乱和嘈杂的信息。

To address this problem：

1. we **proposes a global Recurrent Localization Network (RLN)** which exploits contextual information by the weighted response map in order to localize salient objects more accurately. Particularly, **a recurrent module is employed to progressively refine the inner structure of the CNN over multiple time steps**.
1. Moreover, to effectively recover object boundaries, we **propose a local Boundary Refinement Network (BRN) to adaptively learn the local contextual information for each spatial position.** The learned propagation coefficients can be used to optimally capture relations between each pixel and its neighbors. 

## Recurrent Localization Network (RLN)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562504923677-94ad1d96-aa7f-45d0-bd3c-b92098cb50f2.png#align=left&display=inline&height=385&name=image.png&originHeight=385&originWidth=684&size=92936&status=done&width=684)

因此，从全局的角度来看，我们提出了一种新颖的循环定位网络（RLN），它由两个模块组成：一个类似于Inception结构的上下文加权模块（CWM）和一个循环模块（RM）。

### Contextual Weighting Module (CWM)

CWM旨在预测一个空间响应图来自适应加权每个位置的特征图，这可以定位给定输入的最能被注意到的部分。CWM被放置在每个卷积块的侧输出之上，这将输出特征图作为输入，并且学习一个针对每个像素位置的权重，这基于多尺度的上下文信息。这些权重然后被用到每个特征图上，来产生一个加权的空间表示。**CWM用于过滤分散注意力的和杂乱的背景，使显著性目标更加突出**。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562505495180-0893e056-78c5-4692-a61d-751a60c35e14.png#align=left&display=inline&height=374&name=image.png&originHeight=374&originWidth=443&size=37919&status=done&width=443)

首先连接特征图fk之后的一个下采样层，这是由第k个残差块生成的。然后一个卷积层，有着mxm大小的核滑动，来处理局部的特征。因此上下文信息可以被包含在隐藏层上下文滤波器中。

为了获得一个多尺度上下文信息，使用一个inception模块，使用三个上下文滤波器，分别尺寸为3/5/7，每个滤波器产生一个WxHxC的激活图，紧跟一个L2归一化层。然后拼接所有的激活层来形成特征图f。

为了计算上下文加权响应图Mk，使用一个单通道输出的卷积层用在fkcat后面。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562507627747-24a422ba-e803-4ab1-b4de-296ccdf24641.png#align=left&display=inline&height=36&name=image.png&originHeight=36&originWidth=639&size=4881&status=done&width=639)

得到的WxH的响应图的每个位置都指示着每个空间位置的重要性。之后会使用一个softmax操作得到最终的加权响应图。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562507613258-951453d3-3ac1-49fa-9bd2-830c373ef3cc.png#align=left&display=inline&height=71&name=image.png&originHeight=71&originWidth=634&size=12713&status=done&width=634)

这里的phi表示归一化响应值，k是残差块的索引。最终加权图被上采样得到![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562507978441-5d3036cc-4af1-4499-89c2-db6907c9c2c8.png#align=left&display=inline&height=40&name=image.png&originHeight=40&originWidth=51&size=1420&status=done&width=51)，并被应用到![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562508019300-1c400e3e-e2f4-4bb0-8315-fa1c7c8e1415.png#align=left&display=inline&height=32&name=image.png&originHeight=32&originWidth=28&size=1146&status=done&width=28)上。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562507822717-70cdd0ef-679b-459a-8b3f-2fd1fb6e1ecd.png#align=left&display=inline&height=42&name=image.png&originHeight=42&originWidth=632&size=6024&status=done&width=632)

这里的c表示第c个特征通道。这里是元素乘法。该加权图被应用于所有通道。

---

受启发于论文Learned Contextual Feature Reweighting for Image Geo-Localization的结构：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562505348214-900eb15c-33fa-468f-863d-16fa39a3dca7.png#align=left&display=inline&height=796&name=image.png&originHeight=796&originWidth=513&size=95324&status=done&width=513)

### Recurrent Module (RM)

也提出了一种循环结构，以便在“时间”内逐步细化预测的显着图。它建立循环连接以将某些块的输出传输到其输入，**以便在不同层的训练过程中利用上下文线索**。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562505021741-002a44fb-3ff7-473a-bdbe-36b81f429be5.png#align=left&display=inline&height=426&name=image.png&originHeight=426&originWidth=650&size=82474&status=done&width=650)

Contextual information has been proved effective in saliency detection. **Larger context usually captures global spatial relations among objects while smaller context focuses on the local appearance, both contributing to the saliency detection**.

In this paper, we **propose a novel recurrent module which offers the advantage that increasing time steps enable the whole network to integrate contextual knowledge in a larger neighborhood as time evolves and serves as a refinement mechanism by combining the semantic cues and detailed information** in the inner blocks of Resnet-50.

将ResNet50的每一个block作为基础循环单元，随时间推移，共享相同的权重层参数。当前block由当前前馈输入和上一时间步的该block的输出经过处理后得到的预测加和作为输入。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562508976683-dfea6432-4eb7-4627-bd10-13b0a5496712.png#align=left&display=inline&height=110&name=image.png&originHeight=110&originWidth=683&size=15946&status=done&width=683)

这里的 `*` 表示卷积操作，fk表示一个包含BN和ReLU以及卷积操作的组合。fu表示表示上采样操作。wf表示前馈网络的权重，作用于加和后的特征，而wr表示一个反馈权重，作用于上一时间步的输出特征。wrk横跨不同时间步的相同block，独立进行学习的，来学习特定的转换过程，以结合来自时间步t-1的上下文信息。

所提出的循环结构有几个优点。

1. 首先，通过采用在不同的时间步的相同块的循环连接，循环结构能够利用卷积单元融合（原文是吸收）上下文和结构信息。
1. 其次，通过在每一层共享多次权重，新架构可以增加传统CNN的深度而不会显着增加参数的总数。

## Boundary Refinement Network (BRN)

其次，从局部的角度来看，我们采用边界细化网络（BRN）来恢复边界细节。**BRN将初始RGB图像和显着图作为输入**。显着图用作先验图，其可以帮助学习过程生成更准确的预测。BRN可以预测每个像素的n×n传播系数图，其指示中心点与其n×n个邻居之间的关系。对于每个像素，相应的系数是position-aware的并且可以自适应地学习n×n个邻居的局部上下文信息。

![](https://cdn.nlark.com/yuque/0/2019/png/192314/1562509795882-3d1f4b08-b247-4621-9733-d3d9d1e4c3c2.png#align=left&display=inline&height=423&originHeight=423&originWidth=634&status=done&width=634)

RLN可以通过滤除噪声部分来聚合有用的特征，并通过整合相关信息逐步改进预测。但是，仍然缺少沿着显着物体边界的一些细节结构。为了恢复连续的细节以获得空间精度，采用局部边界细化网络（BRN）来自适应地校正预测。

RLN生成的显著性图和原始的RGB图像被拼接送入BRN，对于每个位置希望通过BRN学习得到一个nxn的传播系数映射，使用其可以将局部上下文信息集成到中心像素上。

对于每个位置i：

1. BRN将会针对各个位置输出一个nxn的传播系数向量
1. 将该向量调整成一个nxn的方形
1. 位置i的细化图将由传播图和显着图在i位置的邻域计算的乘积加和来生成

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562510696066-ad206c77-8b2b-4244-9554-4fd0fb6e2fa3.png#align=left&display=inline&height=76&name=image.png&originHeight=76&originWidth=627&size=8426&status=done&width=627)

1. vdi是**对于原始位置i的第d个邻居计算得到的系数向量**
1. nxn表示局部邻域大小，该卷积块输出的通道数量
1. sdi和s'i分别表示位置i在细化操作前和操作后的预测矢量（原文是这么说，但是感觉这样说有些模糊，sdi应该指的是细化操作前显著性图上指定位置i的nxn邻域内的显著性数据，而s'i则是在说位置i处计算后的最终结果这是汇总了nxn邻域的信息后的结果）

BRN中的每个位置都是位置自适应的，具有不同的传播系数，可以在没有明确监督的情况下通过反向传播自动学习。

As shown in Table 2, BRN is composed of **7 convolutional layers, each with the kernel size of 3 × 3. The ReLU nonlinearity operations are performed between two convolutional layers. We do not utilize pooling layers and large strides in convolutional layers in order to keep the same resolution between input and output feature maps.**

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562511036364-cc2f5773-0042-4cf0-97f7-078058c89819.png#align=left&display=inline&height=336&name=image.png&originHeight=336&originWidth=639&size=47389&status=done&width=639)

The propagation matrices can **model spatial relations among neighbors to help refine the predicted map generated by the RLN**. Compared to the initial saliency map, the refined one should not change too much in terms of the visual appearance. 

To achieve this, we adopt the following initialization in BRN:

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562511157243-87f0510f-9daa-4ba8-a502-505e289209a6.png#align=left&display=inline&height=111&name=image.png&originHeight=111&originWidth=613&size=13704&status=done&width=613)

1. **l∈{1, 2, ..., L} denotes the l-th convolutional layer of BRN.** （BRN共有L个卷积层）
1. **kl** is the **convolutional kernel initialized by the Gaussican distribution （均值µ=0, 标准差σ=0.1）**
1. **z** is the position in each kernel
1. **c** represents the index of the channel.
1. We set all bias parameters in the l-th layer (l<L) to 0.
1. For the L-th layer, biases are set to 0 except that the value **at the center position of n×n neighbors is set to 1**.

Following this initialization, saliency prediction of a certain pixel will be primarily influenced by the central coefficient of the propagation map and also be be affected by the other coefficients.

这里实际上描述的是BRN的L个卷积层的卷积核参数如何初始化的问题。对于卷积核权重而言，使用特定高斯分布初始化，对于卷积层使用的偏置参数而言，只对第L层(最后的那个nxn个卷积核的层)的n×n个卷积核中的中心位置((nxn+1)/2)上的卷积核偏置值设定为1，其余设定为0。因为这个位置实际对应的正是后续计算中的当前位置。

---

这里的BRN参考自[Global-residual and Local-boundary Refinement Networks for Rectifying Scene Parsing Predictions](https://www.ijcai.org/proceedings/2017/0479.pdf)中的LRN：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562510067291-7e6cb2c1-dd49-410b-bf50-182e3ad63e42.png#align=left&display=inline&height=308&name=image.png&originHeight=308&originWidth=645&size=72076&status=done&width=645)

## 实验细节

- We have implemented our network on a single Nvidia GTX 1080 GPU.
- Pre-trained ResNet-50 is used to initialize the convolutional layers in the RLN network (i.e. the conv1 to conv5 block). Other convolutional parameters are randomly assigned.
- We **train our model on the training set of DUTS** and test on its testing set and other datasets.
- All **training and test images are resized to 384×384** as the input to the RLN and 480× 480 to the BRN. We **do not use validation set and train the model until its training loss converges**.
- We **use the SGD method** to train our network.
- A **fixed learning rate is set to 1e-10** for training the RLN and **1e-8** for the BRN with the weight decay 0.0005.
- We use the softmax entropy loss to train both networks.
- For the recurrent structure, the **time step t is set to 2** and we employ **three top supervisions between the ground truth and prediction maps**.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562642427678-65ddacd3-beb8-43b8-900d-9342554b6a46.png#align=left&display=inline&height=495&name=image.png&originHeight=495&originWidth=1286&size=149550&status=done&width=1286)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562642996484-826c7409-475c-4f00-b15b-2ee2c3492870.png#align=left&display=inline&height=259&name=image.png&originHeight=259&originWidth=850&size=108636&status=done&width=850)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562642858492-1bbc19ab-0568-4573-af0a-6c6e4ab63ebf.png#align=left&display=inline&height=177&name=image.png&originHeight=177&originWidth=887&size=40831&status=done&width=887)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562642874572-d9c59dcd-6e0e-44dd-a32d-8bb3f181d6a7.png#align=left&display=inline&height=734&name=image.png&originHeight=734&originWidth=939&size=240815&status=done&width=939)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562645334005-d4cc0d7c-b4ba-426e-8efe-d86befb2d684.png#align=left&display=inline&height=539&name=image.png&originHeight=539&originWidth=960&size=312916&status=done&width=960)

## 相关链接

* 论文：https://www.crcv.ucf.edu/papers/cvpr2018/camera_ready.pdf
