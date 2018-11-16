# Inception V2(V3)(2015)

## 背景

2015年12月，发布了一个新版本的GoogLeNet(Rethinking the Inception Architecture for Computer Vision)模块和相应的架构，并且更好地解释了原来的GoogLeNet架构，GoogLeNet原始思想：

* 通过构建平衡深度和宽度的网络，最大化网络的信息流。在进入pooling层之前增加feature maps
* 当网络层数深度增加时，特征的数量或层的宽度也相对应地增加
* 在每一层使用宽度增加以增加下一层之前的特征的组合
* 只使用3x3卷积,甚至非对称卷积

因此最后的模型就变成这样了：

![img](https://chenzomi12.github.io/2016/12/13/CNN-Architectures/inceptionv3.jpg)

网络架构最后还是跟GoogleNet一样使用pooling层+softmax层作为最后的分类器。

在Rethinking论文中其实提出的是V2 和 V3，V2其实是楼主说说的分解5\*5到3\*3或者1\\*3,3\\*1等。而V3只在V2的基础上添加了BN

## 概述

从2012年Krizhevsky等人[9]赢得了ImageNet竞赛[16]起，他们的网络“AlexNet”已经成功了应用到了许多计算机视觉任务中，例如目标检测[5]，分割[12]，行人姿势评估[22]，视频分类[8]，目标跟踪[23]和超分辨率[3]。

一个有趣的发现是在分类性能上的收益趋向于转换成各种应用领域上的显著质量收益。这意味着深度卷积架构上的架构改进可以用来改善大多数越来越多地依赖于高质量、可学习视觉特征的其它计算机视觉任务的性能。网络质量的改善也导致了卷积网络在新领域的应用，在AlexNet特征不能与手工精心设计的解决方案竞争的情况下，例如，检测时的候选区域生成(proposal generation in detection)[4]。

Inception的计算成本也远低于VGGNet或其更高性能的后继者[6]。这使得可以在大数据场景中[17]，[13]，在大量数据需要以合理成本处理的情况下或在内存或计算能力固有地受限情况下，利用Inception网络变得可行，例如在移动视觉设定中。

从2012年Krizhevsky等人[9]赢得了ImageNet竞赛[16]起，他们的网络“AlexNet”已经成功了应用到了许多计算机视觉任务中，例如目标检测[5]，分割[12]，行人姿势评估[22]，视频分类[8]，目标跟踪[23]和超分辨率[3]。

一个有趣的发现是在分类性能上的收益趋向于转换成各种应用领域上的显著质量收益。这意味着深度卷积架构上的架构改进可以用来改善大多数越来越多地依赖于高质量、可学习视觉特征的其它计算机视觉任务的性能。网络质量的改善也导致了卷积网络在新领域的应用，在AlexNet特征不能与手工精心设计的解决方案竞争的情况下，例如，检测时的候选区域生成(proposal generation in detection)[4]。

Inception的计算成本也远低于VGGNet或其更高性能的后继者[6]。这使得可以在大数据场景中[17]，[13]，在大量数据需要以合理成本处理的情况下或在内存或计算能力固有地受限情况下，利用Inception网络变得可行，例如在移动视觉设定中。

## 构思

### 通用设计原则

这里我们将介绍一些具有卷积网络的、具有各种架构选择的、基于大规模实验的设计原则。在这一点上，以下原则的效用是推测性的，另外将来的实验证据将对于评估其准确性和有效领域是必要的。然而，严重偏移这些原则往往会导致网络质量的恶化，修正检测到的这些偏差状况通常会导致改进的架构。

1. Avoid representational bottlenecks, especially early in the network. 
    Feed-forward networks can be represented by an acyclic graph from the 
    input layer(s) to the classifier or regressor. This defines a clear 
    direction for the information flow. For any cut separating the inputs 
    from the outputs, one can access the amount of information passing 
    though the cut. One should avoid bottlenecks with extreme compression. 
    In general the representation size should gently decrease from the 
    inputs to the outputs before reaching the final representation used for 
    the task at hand. Theoretically, information content can not be assessed
    merely by the dimensionality of the representation as it discards 
    important factors like correlation structure; the dimensionality merely 
    provides a rough estimate of information content.

    避免表征瓶颈，尤其是在网络的前面。前馈网络可以由从输入层到分类器或回归器的非循环图表示。这为信息流定义了一个明确的方向。对于分离输入输出的任何切口，可以访问通过切口的信息量。应该避免极端压缩的瓶颈。一般来说，在达到用于着手任务的最终表示之前，表示大小(即feature map的大小)应该从输入到输出缓慢减小。**理论上，信息内容不能仅通过表示的维度来评估，因为它丢弃了诸如相关结构的重要因素；维度仅提供信息内容的粗略估计。**

2. Higher dimensional representations are easier to process locally within a
    network. Increasing the activations per tile in a convolutional network
    allows for more disentangled features. The resulting networks will 
    train faster.

    更高维度的表示在网络中更容易局部处理。在卷积网络中增加每个图块的激活允许更多解耦的特征。所产生的网络将训练更快。

    > :question:
    >
    > 为什么会更快?

3. Spatial aggregation can be done over lower dimensional embeddings 
    without much or any loss in representational power. For example, before 
    performing a more spread out (e.g. 3 × 3) convolution, one can reduce 
    the dimension of the input representation before the spatial aggregation
    without expecting serious adverse effects. We hypothesize that the 
    reason for that is the strong correlation between adjacent unit results 
    in much less loss of information during dimension reduction, if the 
    outputs are used in a spatial aggregation context. Given that these 
    signals should be easily compressible, the dimension reduction even 
    promotes faster learning.

    空间聚合(即大于1*1的卷积运算)可以在较低维度嵌入上完成，而不会在表示能力上造成许多或任何损失。例如，**在执行更多展开（例如3×3）卷积之前，可以在空间聚合之前减小输入表示的维度，没有预期的严重不利影响**。我们假设，如果在空间聚合上下文中使用输出，则相邻单元之间的强相关性会导致维度缩减期间的信息损失少得多。鉴于这些信号应该易于压缩，因此尺寸减小甚至会促进更快的学习。

4. Balance the width and depth of the network. Optimal performance of the 
    network can be reached by balancing the number of filters per stage and 
    the depth of the network. Increasing both the width and the depth of the
    network can contribute to higher quality networks. However, the optimal
    improvement for a constant amount of computation can be reached if both
    are increased in parallel. The computational budget should therefore be
    distributed in a balanced way between the depth and width of the 
    network.

    平衡网络的宽度和深度。通过平衡每个阶段的滤波器数量和网络的深度可以达到网络的最佳性能。增加网络的宽度和深度可以有助于更高质量的网络。然而，如果**两者并行增加，则可以达到恒定计算量的最佳改进**。因此，计算预算应该在网络的深度和宽度之间以平衡方式进行分配。

### 大滤波器尺寸分解卷积

GoogLeNet网络[20]的大部分初始收益来源于大量地使用降维。这可以被视为以计算有效的方式分解卷积的特例。考虑例如1×1卷积层之后接一个3×3卷积层的情况。在视觉网络中，**预期相近激活的输出是高度相关的**。因此，我们可以预期，它们的激活可以在聚合之前被减少，并且这应该会导致类似的富有表现力的局部表示。

由于Inception网络是全卷积的，每个权重对应每个激活的一次乘法。因此，任何计算成本的降低会导致参数数量减少。这意味着，通过适当的分解，我们可以**得到更多的解耦参数**，从而加快训练。

> :question:
>
> 这里怎么就加快了训练?

此外，我们可以**使用计算和内存节省来增加我们网络的滤波器组的大小**，同时保持我们在单个计算机上训练每个模型副本的能力。

### 分解滤波器的动机

具有较大空间滤波器（例如5×5或7×7）的卷积在计算方面往往不成比例地昂贵。

Of course, a 5 × 5 filter can capture dependencies between signals 
between activations of units further away in the earlier layers, so a 
reduction of the geometric size of the filters comes at a large cost of 
expressiveness.

使用两个3x3的代替一个5x5的,可以明显减小计算量.

不过，这个设置提出了两个一般性的问题：这种替换是否会导致任何表征力的丧失？如果我们的主要目标是对计算的线性部分进行分解，是不是建议在第一层保持线性激活？

实验测试反映出来,对于表现力的丧失的影响不大, 也并不是一定要在开始使用线性激活.

大于3×3的卷积滤波器可能不是通常有用的，因为它们总是可以简化为3×3卷积层序列。

那考虑,是否应该分解为更小的,滤波器?

实际表明,与其使用更小的,到不如尝试使用一些非对称的卷积结构.即nx1.

### 对辅助分类器的补充

最初的动机是将有用的梯度推向较低层，使其立即有用，并通过抵抗非常深的网络中的消失梯度问题来提高训练过程中的收敛。Lee等人[11]也认为辅助分类器促进了更稳定的学习和更好的收敛。有趣的是，我们发现辅助分类器**在训练早期并没有导致改善收敛**：在两个模型达到高精度之前，有无侧边网络的训练进度看起来几乎相同。接近**训练结束**，辅助分支网络开始超越没有任何分支的网络的准确性，达到了更高的稳定水平。

另外，[20]在网络的不同阶段使用了两个侧分支。移除更下面的辅助分支对网络的最终质量没有任何不利影响。再加上前一段的观察结果，这意味着[20]最初的假设，这些分支有助于演变低级特征很可能是不适当的。相反，我们认为**辅助分类器起着正则化项**的作用。

这是支持了这样的事实, 如果侧分支是批标准化的[7]或具有丢弃层，则网络的主分类器性能更好。这也为推测批标准化表现为一个正则化项给出了一个弱支持证据。

### 有效的网格尺寸的减小

传统上，卷积网络使用一些池化操作来缩减特征图的网格大小。为了避免表示瓶颈，在应用最大池化或平均池化之前，需要扩展网络滤波器的激活维度。

考虑一种情况,开始有一个带有k个滤波器的d×d网格，如果我们想要达到一个带有2k个滤波器的d/2×d/2网格，我们首先需要用2k个滤波器计算步长为1的卷积，然后应用一个额外的池化步骤。这意味着总体计算成本由在较大的网格上使用$2d^2k^2$次运算的昂贵卷积支配。一种可能性是转换为带有卷积的池化，因此导致$2(d/2)^2k^2$次运算，将计算成本降低为原来的四分之一。然而，由于表示的整体维度下降到$(d/2)^2k$，会导致表示能力较弱的网络（参见图9），这会产生一个表示瓶颈。

我们建议另一种变体，其甚至进一步降低了计算成本，同时消除了表示瓶颈（见图10），而不是这样做。我们可以使用两个平行的步长为2的块：P和C。P是一个池化层（平均池化或最大池化）的激活，两者都是步长为2，其滤波器组连接如图10所示。

![Figure 9](http://upload-images.jianshu.io/upload_images/3232548-d1264260b6b37706.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

减少网格尺寸的两种替代方式。左边的解决方案违反了第2节中不引入表示瓶颈的原则1。右边的版本计算量昂贵3倍。

![Figure 10](http://upload-images.jianshu.io/upload_images/3232548-06802c0d2cbade36.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

图10。缩减网格尺寸的同时扩展滤波器组的Inception模块。它不仅廉价并且避免了原则1中提出的表示瓶颈。右侧的图表示相同的解决方案，但是从网格大小而不是运算的角度来看。

> https://cs.stackexchange.com/questions/90257/what-exactly-representational-bottleneck-in-inceptionv3-means
>
> 为什么认为前面的两个不合适,后面这两个比较合适?
>
> 答者的回答主要是因为在提前使用pooling中,会导致信息造成不必要的流失,会直接剪裁信息.而先卷积,可以使得先对输入进行扩展到更为稀疏的空间表示,是的在被剪裁之前可以做出更为有意义的推断.之后再进行pooling剪裁,可以进而选择更好的更为抽象的(语义)信息.
>
> pooling的处理相当于压缩了通道,处于前侧的顺序,正好形成了一个表示特征的瓶颈,对后面紧接着的提取表示造成了限制与束缚.
>
> 而下面的两个,好处则认为是平衡了表示与计算的问题.
>
> 但是为什么就平衡了呢?
>
> :question:

## 新意

### 非对称卷积

![Figure 6](http://upload-images.jianshu.io/upload_images/3232548-8951f5af66981a64.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

例如使用3×1卷积后接一个1×3卷积，相当于以与3×3卷积相同的感受野滑动两层网络。

![Figure 3](http://upload-images.jianshu.io/upload_images/3232548-eb478b39a22e8161.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果输入和输出滤波器的数量相等，那么对于相同数量的输出滤波器，两层解决方案节省了33％的计算量。相比之下，将3×3卷积分解为两个2×2卷积表示仅节省了11％的计算量。

> 输入滤波器指的是最靠近输入的那一层的滤波器么?
>
> 输出滤波器指的是最靠近输出的那一层的滤波器么?

> 对比:
>
> 一个5x5的图用一个3x3的模板去卷积,padding=SAME,stride=1,那卷积模板需要横向位移4次(即计算5次),纵向位移4次(同上),即横向纵向一共计算5x5=25次,每个模板是3x3也就是要计算9个数,一共就是25x9=225次
>
> 然后换成1x3和3x1两个模板去卷积,1x3需要横向计算5次,纵向计算5次,每次3个数,一共就是5x5x3=25x3=75,同理3x1模板也需要计算75次,即一共150次
>
> 225和150差33%
>
> 在inception论文的原文中仅仅使用在padding=same的情况下.
>
> **输入输出大小相同, 所以3x1是对1x3的结果(5x5)进行卷积.输出还是要求5x5, 所以这里的分裂卷积层是级联的形式.**
>
> ------
>
> 比如5x5的图用1x3和3x1来卷积，在VALID的情况下，第一次卷积（1x3）横向运算3次，纵向运算5次一共15次，每次运算量3，第一次卷积运算总次数为3x15=45次，并且获得卷积结果图大小为5x3，**然后进入第二次卷积**（3x1），则横向运算3次，纵向运算也是3次，一共运算9次，每次运算量3，则第二次运算量为9x3=27次，一共45+27=72
>
> 如果我用3x3来直接卷积，则横向，纵向位移都是3次，一共位移9次，每次运算量=9，则一共计算了9x9=81次
>
> 好了，如果我用5x1 和1x5的卷积核，那就会和楼主的计算类似，计算量反而增加了。
>
> ------
>
> 最终结论如下：
>
> 33%的开支节省仅仅适用于padding=SAME，因为这个情况下，位移导致的计算量不会因为卷积核以及图像之间的大小比例产生变化.
>
> 而当padding=VALID的情况下，计算量是会随卷积核与图像之间的大小的差值变化而变化的，因此这个差值越大（卷积核大小/图像大小），那么节省的开支就会越大，并且应该有一个阈值，这个阈值我没去算，当高出这个阈值后，非对称卷积就是有利的。
>
> https://www.zhihu.com/question/270055683/answer/351817237

**在理论上，我们可以进一步论证，可以通过1×n卷积和后面接一个n×1卷积替换任何n×n卷积，并且随着n增长，计算成本节省显著增加**（见图6）。

实际上，我们发现，采用这种分解在前面的层次上不能很好地工作，但是对于中等网格尺寸（在m×m特征图上，其中m范围在12到20之间），其给出了非常好的结果。在这个水平上，通过使用1×7卷积，然后是7×1卷积可以获得非常好的结果。

### 标签平滑正则化(LDR)

我们提出了一种通过估计训练期间标签丢弃的边缘化效应来对分类器层进行正则化的机制。

对于每个训练样本$x$，我们的模型计算每个标签的概率**(预测值)$k\in \lbrace 1\ldots K \rbrace: p(k|x) = \frac{\exp(z_k)}{\sum_{i=1}^K \exp(z_i)}$**。这里，$z_i$是对数单位或未归一化的对数概率。考虑这个训练样本在标签上的**实际分布$q(k|x)$**，因此归一化后$\sum_k q(k|x) = 1$。为了简洁，我们省略$p$和$q$对样本$x$的依赖。我们将样本损失定义为交叉熵：$\ell = -\sum_{k=1}^K \log(p(k)) q(k)$。**最小化交叉熵等价于最大化标签对数似然期望**，其中标签是根据它的实际分布$q(k)$选择的。

> 对数似然部分应该就是指前面的p(k)部分.
>
> 因为交叉熵的计算前面带一个符号,最小化这个负值期望,也就是最大化公式里的p(k)的期望,因为q属于实际分布,算是已知的内容.

交叉熵损失对于$z_k$是可微的，因此可以用来进行深度模型的梯度训练。其梯度有一个更简单的形式：$\frac{\partial\ell}{\partial z_k} = p(k) - q(k)$，它的范围在-1到1之间。

考虑单个真实标签y的例子，对于所有$k\neq y$，有$q(y)=1，q(k)=0$。在这种情况下，最小化交叉熵等价于最大化正确标签的对数似然。对于一个特定的样本x，其标签为y，对于$q(k)= \delta_{k,y}$，最大化其对数概率，$\delta_{k,y}$为狄拉克δ函数，当且仅当$k=y$时，δ函数值为1，否则为0。对于有限的$z_k$，不能取得最大值，但对于所有$k\neq y$，如果$z_y\gg z_k$——也就是说，**如果对应实际标签的逻辑单元远多于其它的逻辑单元，那么对数概率会接近最大值**。

然而这可能会引起两个问题。

* 它可能导致过拟合：如果模型学习到对于每一个训练样本，分配所有概率到实际标签上，那么它不能保证泛化能力。
* 它鼓励最大的逻辑单元与所有其它逻辑单元之间的差距变大，与有界限的梯度$\frac{\partial\ell}{\partial z_k}$相结合，这会降低模型的适应能力。直观上讲这会发生，因为模型变得对它的预测过于自信。

我们提出了一个鼓励模型不那么自信的机制。如果目标是最大化训练标签的对数似然，这可能不是想要的，但它确实使模型正则化并使其更具适应性。这个方法很简单。考虑**标签上的一个分布$u(k)$**和**平滑参数$\epsilon$**，与**训练样本x相互独立**。

对于一个真实标签为y的训练样本，我们用$q’(k|x) = (1-\epsilon) \delta_{k,y} + \epsilon u(k)$代替标签分布$q(k|x)=\delta_{k,y}$，其由最初的实际分布$q(k|x)$和固定分布$u(k)$混合得到，它们的权重分别为$1-\epsilon$和$\epsilon$。

按如下方式, 可以看作获得标签$k$的分布：

* 将其设置为真实标签$k=y$；
* 用<u>分布$u(k)$中的采样</u>和<u>概率$\epsilon$</u>替代k。
* 我们建议使用标签上的先验分布作为$u(k)$。在我们的实验中，我们使用了均匀分布$u(k) = 1/K$，以便使得$q’(k) = (1-\epsilon) \delta_{k,y} + \frac{\epsilon}{K}$.

我们将真实标签分布中的这种变化称为*标签平滑正则化*，或LSR。

> 真实标签的分布变的更为平滑, 大小的程度不完全由标签的真实与否来独立决定,现在要考虑到相对的大小.

注意，LSR实现了期望的目标，**阻止了最大的逻辑单元变得比其它的逻辑单元更大**。

实际上，如果发生这种情况，则一个$q(k)$将接近1，而所有其它的将会接近0。**这会导致$q’(k)$有一个大的交叉熵**，因为不同于$q(k)=\delta_{k,y}$，所有的$q’(k)$都有一个正的下界。

LSR的另一种解释可以通过考虑交叉熵来获得：$H(q’,p) = -\sum_{k=1}^K \log p(k) q’(k) = (1-\epsilon)H(q, p) + \epsilon H(u, p)$

因此，LSR等价于用一对这样的损失$H(q,p)$和$H(u,p)$来替换单个交叉熵损失$H(q,p)$。

第二个损失**惩罚预测的标签分布p与先验u之间的偏差**，其中相对权重为$\frac{\epsilon}{1-\epsilon}$。注意，由于$H(u,p) = D_{KL}(u|p) + H(u)$和$H(u)$是固定的，因此这个**偏差可以等价地被KL散度捕获**。

**当u是均匀分布时，$H(u,p)$是度量预测分布p与均匀分布不同的程度**，也可以通过负熵$-H(p)$来度量（但不等价）；我们还没有实验过这种方法。

> 这里考虑负熵度量有何意义?
>
> ![1540900985258](assets/1540900985258.png)
>
> 当且仅当各个x的概率相同,即处于均匀分布的时候,该熵可以获得最大值.
>
> 从$H(p)$的性质的角度来看, 当取反的时候, 正好可以认为在均匀分布的时候,是最小的, 也就正好可以度量与均匀分布的一个差异程度. 

## 训练

1. TensorFlow[1]分布式机器学习系统

2. 随机梯度方法训练

3. 使用了50个副本，每个副本在一个NVidia Kepler GPU上运行

4. 批处理大小为32

5. 100个epoch

6. 之前的实验使用动量方法[19]，衰减值为0.9，而我们最好的模型是用RMSProp [21]实现的，衰减值为0.9，\epsilon=1.0。

7. 使用0.045的学习率，每两个epoch以0.94的指数速率衰减

8. 阈值为2.0的梯度裁剪[14]被发现对于稳定训练是有用的

    > 梯度截断可以一定程度上缓解梯度爆炸问题（梯度截断，即在执行梯度下降步骤之前将梯度设置为阈值

9. 使用随时间计算的运行参数的平均值来执行模型评估。

## 测试

### 低分辨率输入

普遍的看法是，**使用更高分辨率感受野的模型倾向于导致显著改进的识别性能**。

然而，区分第一层感受野分辨率增加的效果和较大的模型容量、计算量的效果是很重要的。

如果我们只是改变输入的分辨率而不进一步调整模型，那么我们最终将使用计算上更简陋的模型来解决更困难的任务。当然，由于减少了计算量，这些解决方案很自然就出来了。

为了做出准确的评估，模型需要分析模糊的提示，以便能够“幻化”细节。这在计算上是昂贵的。

因此问题依然存在：如果计算量保持不变，更高的输入分辨率会有多少帮助。

确保不断努力的一个简单方法是**在较低分辨率输入的情况下减少前两层的步长，或者简单地移除网络的第一个池化层。**

> 尽可能保留原始的信息, 不做过早的使用表示瓶颈.

![Table 2](http://upload-images.jianshu.io/upload_images/3232548-87fd6728c8b67d1b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

1. 步长为2，大小为$299\times 299$的感受野和最大池化。
2. 步长为1，大小为$151\times 151$的感受野和最大池化。
3. 步长为1，大小为$79\times 79$的感受野和第一层之后**没有**池化。

当感受野尺寸变化时，识别性能的比较，在每种情况下，网络都进行了训练，直到收敛，并在ImageNet ILSVRC 2012分类基准数据集的验证集上衡量其质量。

但计算代价是不变的。

表明，可以通过感受野分辨率为$79\times 79$的感受野取得高质量的结果。

这可能证明在检测相对较小物体的系统中是有用的。

### 对V3进行测试

![Table 5](http://upload-images.jianshu.io/upload_images/3232548-208753b7acf5eb03.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里的V3指的是有BN-auxiliary(是指辅助分类器的全连接层也批标准化的版本),同时还有滤波器分解, 标签平滑.

我们已经研究了怎样在神经网络中**分解卷积**和**积极降维**来得出计算成本相对较低的网络，同时保持高质量。

较低的参数数量、额外的正则化、批标准化的辅助分类器和标签平滑的组合允许在相对适中大小的训练集上训练高质量的网络。