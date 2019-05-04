# Superpixel Sampling Networks

* [Superpixel Sampling Networks](#superpixel-sampling-networks)
  * [工作介绍](#工作介绍)
  * [相关工作](#相关工作)
    * [超像素算法](#超像素算法)
    * [深度聚类](#深度聚类)
  * [基础知识](#基础知识)
  * [SSN(Superpixel Sampling Networks)](#ssnsuperpixel-sampling-networks)
    * [网络结构](#网络结构)
    * [可微分的SLIC](#可微分的slic)
    * [Superpixel Sampling Network](#superpixel-sampling-network)
    * [像素和超像素表示之间的映射](#像素和超像素表示之间的映射)
  * [学习任务特定的超像素](#学习任务特定的超像素)
      * [任务特定的重建损失](#任务特定的重建损失)
      * [紧凑性损失](#紧凑性损失)
  * [实验细节](#实验细节)
    * [超像素](#超像素)
      * [评估指标](#评估指标)
      * [消融实验](#消融实验)
      * [比较](#比较)
    * [语义分割](#语义分割)
      * [Cityscapes](#cityscapes)
      * [Pascal VOC](#pascal-voc)
      * [额外实验](#额外实验)
    * [光流](#光流)
  * [总结](#总结)
  * [相关连接](#相关连接)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556789607583-bc44a804-4fa0-4fbf-869b-9e6593d02b15.png#align=left&display=inline&height=231&name=image.png&originHeight=231&originWidth=728&size=38077&status=done&width=728)

本文的思想很简单，传统的超像素算法是一种有效的低/中级的图像数据的表达，极大地降低了后续图像任务的基元(image primitives)数量。但是现存的超像素算法由于使用的是最邻近算法，一般都是不可微的，这就导致很难将它们集成到端到端的深度网络中，所以文章就改进提出了一种可微分的超像素算法，也就是文章提出的超像素采样网络（Superpixel Sampling Network），这可以**学习任务特定的有着灵活的损失函数的超像素**，并且具有快速运行时间(runtime)。

## 工作介绍

超像素是通过基于低级图像属性对图像像素进行分组而形成的图像的过分割。它们提供了图像内容感知上有意义的细分，从而减少了后续图像处理的图像基元的数量。由于它们的代表性和计算效率，超像素已经成为一种既定的低/中级图像表示，并广泛应用于计算机视觉算法，如物体检测，语义分割，显着性估计，光流估计，深度估计，跟踪等等。超像素尤其广泛用于传统的能量最小化框架，其中少量图像基元极大地降低了优化复杂度。

近年来，广泛的计算机视觉问题都开始采用深度学习。除了一些方法（例如，[_Superpixel convolutional networks using bilateral inceptions，SuperCNN: A superpixelwise convolutional neural network for salient object detection，Recursive context propagation network for semantic scene labeling_]），超像素几乎不与现代深度网络结合使用。这有两个主要原因。

1. 形成大多数深层结构基础的标准卷积运算通常在规则网格上定义，并且当在不规则超像素晶格上操作时变得低效。
1. 现有的超像素算法是不可微分的，因此在深度网络中使用超像素使得在端到端的可训练网络架构中引入了不可微分的模块。

这项工作通过提出一种新的深度可扩展的超像素分割算法来缓解第二个问题。

首先重新讨论广泛使用的简单线性迭代聚类（SLIC）超像素算法，并通过放松SLIC中存在的最近邻居约束将其转换为可微分算法。这种新的可微分算法允许端到端训练，使得能够利用强大的深度网络来学习超像素，而不是使用传统的手工设计特征。这种深度网络与可微分SLIC的结合形成了**超像素采样网络（SSN）** 的端到端可训练超像素算法。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556790286923-54e76592-2811-4f69-b8f4-72357c7d7b43.png#align=left&display=inline&height=341&name=image.png&originHeight=341&originWidth=757&size=187732&status=done&width=757)

图1显示了所提出的SSN的概述。

1. 给定的输入图像首先通过深度网络产生更为有效的特征。
1. 然后将这些深度特征传递到可微分SLIC上
1. SLIC执行迭代聚类
1. 从而产生所需的超像素

整个网络是端到端的可训练的。**SSN的可微分性允许使用灵活的损失函数来学习特定于任务的超像素**。图1显示了一些SSN生成的超像素样本。

与现有的超像素算法相比，所提出的SSN具有以下有利特性：

1. 可端到端训练
1. 灵活且任务特定： SSN允许学习灵活的损失函数，从而学习特定于任务的超像素
1. 当前最佳的超像素算法
1. 有利的运行时间：SSN在运行时方面也优于当前优异的超像素算法，使其适合在大型数据集上学习，并且对实际应用也有效

## 相关工作

### 超像素算法

传统的超像素算法可以被大致分为基于图的和基于聚类的方法。

- 基于图的方法将超像素分割问题表达为图划分问题。图的节点为像素，图的边表示邻接像素之间的连接强度。通常，通过解决离散优化问题来进行图划分。一些广泛使用的算法包括下面几种，但由于离散优化涉及离散变量，优化目标通常是不可微分的，因此难以在基于图的方法中利用深度网络。
  - 归一化切割[Normalized-Cuts]
  - Felzenszwalb和Huttenlocher（FH）
  - 熵率超像素（ERS）
- 基于聚类的方法利用传统的聚类技术，例如用于超像素分割的k均值。广泛使用的算法包括下面几种，这些方法主要进行k均值聚类，但其特征表示不同。**虽然这些聚类算法需要迭代更新，但在SNIC方法[2]中提出了一种用于超像素分割的非迭代聚类方案**。
  - SLIC：SLIC将每个像素表示为**5维特征（位置和Lab颜色 XY Lab特征）**
  - LSC：LSC方法**将这些5维特征投影到10维空间并在投影空间中执行聚类**
  - Manifold-SLIC：使用**二维流形特征空间**进行超像素聚类

本文提出的方法也是基于聚类的方法。但是，与现有技术不同，这里利用深度网络通过端到端的训练框架来学习超像素聚类的功能。

正如最近的一篇调查论文[36]所详述的，其他技术被用于超像素分割，包括分水岭变换，几何流，图形切割，均值漂移和爬山算法（watershed transform, geometric flows, graph-cuts, mean-shift, and hill-climbing）。然而，这些方法都依赖于手工设计的特征，将深度网络融入这些技术并非易事。最新的SEAL技术[_Learning superpixels with segmentation-aware affinity loss_]提出了一种不可微的超像素算法通过绕过梯度来为超像素分割学习深度特征的方法。与我们的SSN框架不同，SEAL不是端到端可微分的。

### 深度聚类

受到监督任务深度学习成功的启发，有几种方法研究了深度网络在无监督数据聚类中的应用。

- 最近，Greff等人提出神经期望最大化框架，他们使用深度网络模拟聚类标签的后验分布，并展开EM程序中的迭代步骤以进行端到端训练
- 在另一项工作中，梯形网络用于建模用于聚类的分层潜变量模型
- Hershey等人提出了一种基于深度学习的聚类框架，用于分离和分割音频信号
- Xie等人提出了一个深度嵌入式聚类框架，用于同时学习特征表示和聚类分配
- 在最近的一份调查报告中，Aljalbout等人给出了基于深度学习的聚类方法的分类

在本文提出了一种基于深度学习的聚类算法。与以前的工作不同，本文的算法是针对超像素分割任务而定制的，其中使用特定于图像的约束。此外，该框架可以轻松地结合其他视觉目标函数来学习任务特定的超像素表示。

## 基础知识

SSN方法核心是一种可微分的聚类方法，受SLIC超像素算法启发。这里简单介绍下SLIC算法。SLIC算法是最简单最广泛使用的超像素算法之一，它很容易实现，有着较高的运行时（runtime），并且可以生成较为紧凑和均匀的超像素。尽管已经有数种SLIC算法的变体，在SLIC的原始形式中，SLIC是一种K均值聚类算法，针对图像像素的五维特征空间（XY LAB）进行聚类。

对于超像素计算的任务而言，其最终目的是要对每个像素分配一个超像素标号，假设要分成m个超像素，且给定一个图像I，它是一个nx5的集合，有n个像素，每个像素是5维（XY LAB）向量，SLIC算法的主要操作流程如下所述。

采样初始的m个聚类中心（超像素中心）S0，其是mx5的集合。该采样通常在像素网格上均匀地进行，并且基于图像梯度进行一些局部扰动。给定这些初始的超像素中心后，SLIC算法可以开始反复迭代下面的两步：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556795622890-ae856da4-5114-4427-8e69-90982f684e1b.png#align=left&display=inline&height=66&name=image.png&originHeight=66&originWidth=737&size=7777&status=done&width=737)![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556795809659-bcf6c9a1-d9cc-4a64-9ca5-a07574ca83ae.png#align=left&display=inline&height=69&name=image.png&originHeight=69&originWidth=710&size=6302&status=done&width=710)

1. **像素-超像素关联**上一次迭代确定的超像素中心Hp=Si**，如上1式，这里的D计算的是欧式距离的平方![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556795673616-53d1562a-145f-47ae-833c-98ee70140ff0.png#align=left&display=inline&height=28&name=image.png&originHeight=28&originWidth=185&size=2773&status=done&width=185)。实际上就是一个聚成簇的过程。
1. **超像素中心更新**：对每个超像素簇中所有的像素的特征进行平均，获得簇中心，进而得到这次迭代后的聚类中心。如上式2，表示的就是这个过程。其中Zi表示第i个簇中包含的像素数量。

这两个步骤构成了SLIC算法的核心，并且一直重复**直到收敛**或者**达到固定次数的迭代**。因为在所有像素和超像素之间计算公式1中的距离D是耗时的，该计算通常被约束到**每个超像素中心周围的固定邻域**。最后，根据应用，有一个可选步骤，强制每个超像素簇中的像素之间的空间连接。[_SLIC superpixels compared to state-of-the-art superpixel methods_]

## SSN(Superpixel Sampling Networks)

### 网络结构

![](https://cdn.nlark.com/yuque/0/2019/png/192314/1556797849298-bd4d761d-542d-41b3-8e95-f16502b666d2.png#align=left&display=inline&height=321&originHeight=327&originWidth=760&status=done&width=746)

途中的箭头是双线性插值上采样，多个输入会被拼接起来送入卷积。卷积使用的都是3x3卷积，每一层输出为64通道，除了最后一层的输出为k-5，因为要和原图像的XYLab特征向量进行拼接，一起生成一个k维的特征张量集合。
k微特征被送入两个可微的SLIC模块，迭代更新关联与聚类中心v步，整个网络端到端训练。

### 可微分的SLIC

首先分析为什么SLIC不可以微分。仔细观察SLIC中的所有计算，_像素-超像素关联_的计算产生不可微分性，其涉及不可微分的最近邻操作。这个最近邻计算也构成了SLIC超像素聚类的核心，因此无法避免这种操作。

可微分的SLIC的关键在于转换这种不可微分的最近邻操作为可微分的计算。由于前面的SLIC算法中的这种**硬性关联H**存在不可微的特性，那么就将其**软化**，这里有点类似于阶跃函数和Sigmoid函数的关系。后者也可以看做是前者的一个**软化**。

这里提出一种计算soft-associations Q(nxm)的方法。对于迭代过程中第t步的像素p和超像素i，这里替换最近邻操作用以下的关于距离的钟型函数权重的形式来表达：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556796448973-df72aef2-a3ed-483d-bcaf-f0be065469ba.png#align=left&display=inline&height=63&name=image.png&originHeight=63&originWidth=758&size=6565&status=done&width=758)

因为这里对于超像素与像素的关联实际上就是一中距离上的关系，原本是直接限定了最近的，相当于直接截断了整图像素与超像素中心的距离关联。这里不用截断操作。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556796457584-d4e03e09-5568-4f08-9553-c4d148874b04.png#align=left&display=inline&height=75&name=image.png&originHeight=75&originWidth=756&size=6663&status=done&width=756)

式子4通过1加权的形式计算了新的超像素中心。这里的![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556796851192-39f5b9d6-de0c-4bc9-a0b1-70bc74791655.png#align=left&display=inline&height=31&name=image.png&originHeight=31&originWidth=129&size=2564&status=done&width=129)是一个归一化约束，实际上就是对于Q的列归一化，表示为![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556797010790-c1a813b1-628e-479a-9ad6-a048cb8a2421.png#align=left&display=inline&height=32&name=image.png&originHeight=32&originWidth=26&size=944&status=done&width=26)，于是式子4可以写作![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556796991992-c1efc780-e1db-42d4-8c3d-3ea958b78299.png#align=left&display=inline&height=36&name=image.png&originHeight=36&originWidth=101&size=2006&status=done&width=101)（(mxn)x(nx5)）。因为对于计算所有的像素和超像素之间的距离仍然是一件计算昂贵的事情，所以这里进行了约束，只计算像素和9个周围的像素，如下图中的红色和绿色框所示。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556797330615-1211cc64-64dd-4b92-afa9-4c9a010401cf.png#align=left&display=inline&height=227&name=image.png&originHeight=227&originWidth=752&size=178563&status=done&width=752)

对于绿色框中的像素，在计算关联的时候只考虑红色框里周围的超像素。这将Q从nxm变为了nx9，一定程度上降低了计算和存储的消耗。在Q计算中的近似有些相似于SLIC中的最近邻搜索。

现在，每个SLIC迭代中的计算都是完全可微分的，将这种修改后的算法称为可微分SLIC（_ differentiable SLIC_）。根据经验，观察到用可微分SLIC中的软关联替换SLIC中的硬关联不会导致任何性能下降。由于这种新的超像素算法是可微分的，因此可以轻松地集成到任何深度网络架构中。

### Superpixel Sampling Network

可以利用深度特征提取器而不是手工设计的超像素特征Ip，并且端到端的对整个网络进行训练。换句话说，将上面的式3和4中的图像特征Ip替换为深度网络得到的k维像素特征Fp（nxk）。将深度网络与可微分SLIC的耦合称为超像素采样网络（SSN）。

![](https://cdn.nlark.com/yuque/0/2019/png/192314/1556797613632-14c6b706-8d99-4ba6-847f-6503af1eb548.png#align=left&display=inline&height=251&originHeight=423&originWidth=1258&status=done&width=746)
算法1概述了SSN中的所有计算步骤。

1. 从使用CNN的深度图像特征提取开始（第1行）
1. 使用初始**常规超像素网格**中的平均像素特征来初始化超像素中心（第2行）（图2）
1. 对于v次迭代，使用上述计算（第3-6行）迭代地更新**像素-超像素关联**超像素中心**。
  1. 虽然可以直接使用软关联Q来执行多个下游任务，但根据应用需求，可以选择将软关联转换为硬关联（第7行）。
  1. 此外，与原始SLIC算法一样，可以选择强制跨每个超像素集群内的像素进行空间连接。这是通过**将小于特定阈值的超像素与周围的超像素合并**，然后为每个空间连接的组件分配唯一的簇ID来实现的。
  1. 请注意，这**两个可选步骤（第7,8行）不可微分**。

### 像素和超像素表示之间的映射

对于使用超像素的一些下游应用程序，像素表示被映射到超像素表示，以及反过来。

1. 利用提供硬聚类的传统超像素算法，这种**从像素到超像素表示的映射**是通过在每个聚类内部进行平均来完成的（方程2）。
1. **从超像素到像素表示**的逆映射是通过将相同的超像素特征分配给属于该超像素的所有像素来完成的。我们可以使用与SSN超像素相同的像素-超像素映射，这使用了从SSN获得的硬聚类（算法1中的第7行）。然而，由于这种硬关联的计算是_不可微分_的，因此在集成到端到端可训练系统时可能不希望使用硬聚类。

值得注意的是，由SSN生成的软像素-超像素关联也可以容易地用于像素和超像素表示之间的映射。

1. 式4描述了从像素到超像素表示的映射。![](https://cdn.nlark.com/yuque/0/2019/png/192314/1556796991992-c1efc780-e1db-42d4-8c3d-3ea958b78299.png#align=left&display=inline&height=32&originHeight=36&originWidth=101&status=done&width=90)
1. 从超像素到像素的逆映射通过乘以行归一化的Q来计算，表示为：![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556798639260-67c29278-fcf6-4df0-ad2e-9c84757d938c.png#align=left&display=inline&height=26&name=image.png&originHeight=26&originWidth=78&size=1710&status=done&width=78)。

因此像素-超像素特征映射关系可以使用简单的矩阵乘法来进行描述，并且这是可微的。

## 学习任务特定的超像素

端到端可训练SSN的主要优点之一是损失函数的灵活性，可以使用它来学习任务特定的超像素表示。

与任何CNN一样，可以将SSN与任何特定于任务的损失函数相结合，从而学习针对下游计算机视觉任务进行优化的超像素。

在这里，专注于优化超像素的表现效率，即学习可以有效地表示场景特征的超像素，例如语义标签，光流，深度等。例如，如果想要学习超像素用于下游语义分割任务，期望产生遵循语义边界的超像素。为了优化表示效率，作者发现任务特定的重建损失和紧凑性损失的组合表现良好。

#### 任务特定的重建损失

用超像素表示我们**想要有效表示的像素属性为R（n×l）**。例如，R可以是语义标签（独热编码后的）或光流图。重要的是要注意，我们**在测试时是不能无法获取R，即SSN仅使用图像数据来预测超像素**。只用R来训练，以便SSN可以学习预测适合的表示R的超像素。可以使用列标准化关联矩阵Q，![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556799594509-ffeea6dd-4870-4524-93bd-661305292d00.png#align=left&display=inline&height=33&name=image.png&originHeight=33&originWidth=95&size=2031&status=done&width=95)将像素属性映射到超像素上。然后使用行标准化关联矩阵Q，![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556799643130-b8758fb6-1a7d-4792-a22d-ff1a6638d754.png#align=left&display=inline&height=28&name=image.png&originHeight=28&originWidth=87&size=1839&status=done&width=87)将得到的超像素表示映射回像素表示R*（n×1）。可以得到重建损失为：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556799684769-e930dd88-c7e8-4d49-a816-a8356a1d2aca.png#align=left&display=inline&height=50&name=image.png&originHeight=50&originWidth=745&size=7459&status=done&width=745)

而这里的L表示的是任务特定的损失函数，针对分割任务使用交叉熵函数来作为L，使用L1范数损失来应对光流任务。这里的Q表示的是可微分SLIC的最终迭代之后关联矩阵Qv。为了方便，忽略了v。

#### 紧凑性损失

除了上面的损失，也使用了一个紧凑性损失来鼓励超像素实现空间上的紧凑性。也就是在超像素簇内部有着更低的空间方差。

1. 使用Ixy表示位置像素特征。
1. 首先将这些位置特征使用Q映射到超像素表示。
1. 然后使用硬关联H替代软关联Q来逆映射到像素表示。
  1. 这里通过将相同超像素位置的特征赋给属于该超像素的像素。

紧凑损失可以使用如下的L2损失表示：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556810533319-fc55769b-884e-49de-bf99-8aca825ce5d3.png#align=left&display=inline&height=37&name=image.png&originHeight=33&originWidth=587&size=4839&status=done&width=652.2222395002111)

这个损失鼓励超像素有着更低的空间方差。

SSN的灵活性允许使用许多其他的损失函数。这份工作中使用了前面的重建损失与这里的紧凑性损失的组合损失：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556811522204-922c34a0-ff1d-42df-abd5-88b560d1b21d.png#align=left&display=inline&height=26&name=image.png&originHeight=23&originWidth=166&size=3033&status=done&width=184.44444933055374)

其中的系数lambda为1e-5。

## 实验细节

1. 使用放缩过的XYLab特征作为SSN的输入，其中位置和色彩特征尺寸表示为ypos和ycolor。
   1. ycolor的值独立于超像素的数量，被设置为0.26，颜色值被放缩到0~255。
   2. ypos的值依赖于超像素数量，![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556811799721-ae9a432f-76d0-4a24-9419-d24d907190f7.png#align=left&display=inline&height=24&name=image.png&originHeight=22&originWidth=230&size=4632&status=done&width=255.555562325466)，这里的m和n分别表示超像素和像素沿着图像的宽和高的数量。实际中设置eta为2.5。
2. 训练中使用201x201大小的图像patch和100个超像素。
3. 数据增强使用了左右反转，对于小的BSDS500数据集，使用了额外的图像patch随机放缩的增强方式。
4. 所有实验都是用**Adam优化器，batch为8，学习率为0.0001**。
5. 除非特别提及，训练模型**500K次迭代**，并基于验证集准确率来选择最终的训练模型。
6. 消融研究中，训练改变了参数的模型200K次迭代。
7. 需要注意的是，使用一个训练好的SSN模型，通过缩放上面描述的输入位置特征来估计不同数量的超像素。
8.  在训练的时候，可微分的SLIC使用五次迭代，也就是v=5，而测试的时候，v=10。因为观察到，随着迭代次数的提升，性能增益不大（only marginal performance gains with more iterations）。

### 超像素

[BSDS500数据集](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)：BSDS500 consists of 200 train, 100 validation, and 200 test images. Each image is annotated with ground-truth (GT) segments from multiple annotators. We **treat each annotation as as a separate sample resulting** in 1633 training/validation pairs and 1063 testing pairs.

为了学习附着到真值分割的超像素，在重建损失（式子5）中使用真值分割标签，也就是将真值分割标签表达为one-hot编码向量，并且使用其作为像素属性R，用在重建损失中。在式子5中使用交叉熵损失作为L，注意这里与真值标签具有意义的语义分割任务不同，这个数据集里的真值标签并不带有什么语义信息。这对网络的学习设置没有任何问题，因为SSN和重建损失都与像素属性R的含义无关（agnostic）。重建损失使用给定输入信号R及其重建版本R*生成损失值，并不考虑是否在图像中保留了R的含义。

#### 评估指标

超像素在各种视觉任务中都很有用，并且存在用于评估超像素的若干度量。

在这项工作中，将**可实现的分割准确度**（Achievable Segmentation Accuracy，ASA）视为主要指标，同时还报告边界指标，如边界召回（BR）和边界精度（BP）指标。

- ASA得分表示通过在超像素上执行的任何分割步骤可实现的准确度的上限。
- 另一方面，边界精度和召回率测量超像素边界与GT边界的对齐程度。

在补充材料中更详细地解释了这些指标。这些得分越高，分割结果越好。通过改变生成的超像素的平均数量来报告平均ASA和边界度量。对边界精度和召回的公平评估期望超像素在空间上连接。因此，为了进行无偏比较，遵循计算硬聚类的可选后处理，并在SSN超像素上实施空间连通性（算法1中的第7-8行）。

#### 消融实验

- 参考图3所示的主模型，在深网络中有7个卷积层，作为SSNdeep。
- 作为基线模型，评估使用可微分SLIC生成的超像素，该像素采用像素XYLab特征作为输入。这与标准SLIC算法类似，将其称为SSNpix，并且没有可训练的参数。
- 作为另一个基线模型，替换深度网络用一个简单的卷积层来学习输入XYLab特征的线性变换，这个表示为SSNlinear。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556850879478-35a726b0-24b9-461a-908d-ee29eb91145d.png#align=left&display=inline&height=433&name=image.png&originHeight=433&originWidth=1194&size=149637&status=done&width=1194)

图4中可以看出来，SSNlinear就已经实现了远高于基线的ASA和BR得分，这显示了损失函数和通过超像素算法反向传播损失信号的重要性。SSNdeep进一步提升了ASA和BR得分。可以看到，k越高，往往也就有着更高的得分，v也是这样。

**出于计算的考虑，这里之后的SSNdeep都指代k=20和v=10。**

#### 比较

![](https://cdn.nlark.com/yuque/0/2019/png/192314/1556850895423-11927450-e8d6-4286-a1a6-dc885a46ef14.png#align=left&display=inline&height=256&originHeight=401&originWidth=1168&status=done&width=746)

图上可以看到，SSNpix效果接近于SLIC算法，这也反映出来当放松最近邻约束的时候SLIC的性能并不会损失。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556851551423-2cf5edac-13a6-4a8d-ab48-e55089d0516f.png#align=left&display=inline&height=510&name=image.png&originHeight=510&originWidth=1314&size=950581&status=done&width=1314)

请注意，SSNdeep超像素平滑地跟随对象边界，并且也更集中在对象边界附近。

### 语义分割

使用的数据集：Cityscapes和Pascal VOC。

- Cityscpes：We train SSN with the 2975 train images and evaluate on the 500 validation images. For the ease of experimentation, we experiment with half-resolution(512 × 1024) images.
- Pascal VOC：We train SSN with 1464 train images and validate on 1449 validation images.

#### Cityscapes

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556852211383-a18eab7c-9b77-48c5-b666-89eb2a23fe2d.png#align=left&display=inline&height=351&name=image.png&originHeight=351&originWidth=1042&size=117787&status=done&width=1042)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556851987035-d8ea2795-cce2-4983-976c-54083f5383b7.png#align=left&display=inline&height=366&name=image.png&originHeight=366&originWidth=398&size=52082&status=done&width=398)

我们使用NVIDIA Tesla V100 GPU计算GPU运行时。SSNpix和SSNdeep之间的运行时比较**表明SSN计算时间的很大一部分是由于可微分SLIC**。运行时表明SSN比几个超像素算法的实现快得多。

这里与上一部分的差异主要在于语意标签的使用和重建损失。鼓励SSN学习附着于语义分割的超像素。

#### Pascal VOC

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556852295601-2f8e35fe-c4a2-41f0-bc7b-2bba1dc78c51.png#align=left&display=inline&height=383&name=image.png&originHeight=383&originWidth=1033&size=133840&status=done&width=1033)

图8中a的曲线显示了不同技术的ASA得分，这里没有分析在这个数据集上的边界得分，因为真值语义边界被用忽略标签给扩展了。这里同时评估了使用BSDS训练的模型，也就是图中的SSNdeep-BSDS模型，可以看出，相较于当前数据集训练的模型，只有少量的得分损失。这也体现出了SSN在不同数据集上的泛化能力和鲁棒性。

图7中有些图片示例。

#### 额外实验

进行了一个额外的实验，将SSN插入到[_Superpixel convolutional networks using bilateral inceptions_]的下游语义分割网络中，[_Superpixel convolutional networks using bilateral inceptions_]中的网络具有双边inception层(bilateral inception layer)，**利用超像素进行远程数据自适应信息传播**，跨过中间CNN表示。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556852870175-543abd73-bda0-4394-8628-b7df13e276a2.png#align=left&display=inline&height=242&name=image.png&originHeight=242&originWidth=331&size=34893&status=done&width=331)

表2显示了在测试数据上评估的该联合模型的IoU得分。与这篇论文中使用的原始SLIC超像素相比，IoU的改进表明**SSN还可以为使用超像素的下游任务网络带来性能改进**。

### 光流

使用的数据集是MPI-Sintel：The MPI-Sintel dataset consists of 23 video sequences, which we split intodisjoint sets of 18 (836 frames) training and 5 (205 frames) validation sequences.

To this end, we experiment on the MPI-Sintel dataset and use SSN to predict superpixels** given a pair of input frames**.

为了**证明SSN对回归任务的适用性**，进行了概念验证实验，学习了遵循光流边界的超像素。使用真值光流作为像素属性R，用在重建损失中，使用L1损失作为L，鼓励SSN来生成有效表示光流的超像素。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1556853064754-48efb025-4b4f-4a11-8800-1cfe41d4a6fb.png#align=left&display=inline&height=259&name=image.png&originHeight=259&originWidth=1047&size=179564&status=done&width=1047)

这里使用相同的方法来计算ASA得分。对于每个超像素内部的像素，指定平均真值光流来产生一个分割光流，图9中显示了一些分割光流的结果。之后计算真值光流和分割光流的欧氏距离，这称为终点误差（end-point error EPE）。该值越低，超像素越能更好的表示光流。

图9中的结果表明，SSNdeep超像素相比其他超像素，对于真值光流的变化更好地对齐。图8b显示了现有超像素技术的平均EPE值，可以看出SSNdeep相较于现存的超像素技术表现出色。这显示了**SSN在学习任务特定的超像素中的有用性**。

## 总结

提出了一种新颖的超像素采样网络（SSN），它利用通过端到端训练学到的深层特征来估计任务特定的超像素。这是第一个端到端可训练的深度超像素预测技术。

- 实验的几个基准测试表明，SSN始终如一地在表现出色，同时也更快。
- 将SSN集成到语义分割网络中还可以提高性能，显示SSN在下游计算机视觉任务中的实用性。
- SSN快速，易于实施，可以轻松集成到其他深层网络中，具有良好的实证性能。
- SSN解决了将超像素纳入深度网络的主要障碍之一，这是现有超像素算法的不可微分性质。
- 在深度网络中使用超像素可以具有几个优点。
    - 超像素可以降低计算复杂度，尤其是在处理高分辨率图像时
    - 超像素也可用于加强区域常量假设（enforce piece-wise constant assumptions）
    - 也有助于远程信息传播

相信这项工作开辟了利用深层网络中的超像素的新途径，并激发了使用超像素的新深度学习技术。

## 相关连接

- 论文：[http://openaccess.thecvf.com/content_ECCV_2018/papers/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.pdf)
- 代码：[https://varunjampani.github.io/ssn/](https://varunjampani.github.io/ssn/)
