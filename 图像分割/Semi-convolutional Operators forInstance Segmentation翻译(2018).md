# Semi-convolutional Operators forInstance Segmentation

![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1Njk4NTE2MDMzOTctY2FmZjA4YWItMjQ3Ni00NmQzLWI1MGMtZjJjNjc2ODhkM2JlLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=348&name=image.png&originHeight=348&originWidth=857&search=&size=58836&status=done&width=857)

这篇文章看的我是真恶心，最后还是没有坚持下去，还是直接翻译一波吧。

## 摘要

目标检测和实例分割由基于区域的方法（如Mask RCNN）主导。然而，人们越来越感兴趣的是**将这些问题减少（reduce）到像素标记任务**，因为后者可以更高效，可以无缝地集成在图像到图像网络体系结构中，如许多其他任务中使用的那样，并且**可以更精确地用于没有被边界框很好地近似的对象**。在本文中我们理论和经验性的展示了该构建密集像素嵌入，这**可以分离使用卷积操作算子（operators）不能很容易实现的目标示例**。与此同时，我们展示了简单的修改，我们称之为半卷积，这在这个任务中，更有机会成功。我们使用后者（半卷积操作）展示了和霍夫投票以及被卷积网络在空间上操作（steer）的双边核（双边滤波）的变体之间的联系。我们证明了这些操作算子也可用于改善例如Mask RCNN这样的方法，证明了相较于单独的Mask RCNN而言，对于复杂的生物的形状和PASCAL VOC类别，可以实现更好的分割

目前主流的物体检测方法，如R-CNN系列，YOLO，SSD等，主要是一种建议加验证（propose&verify，P&V）的思想：首先动态地或者在固定的提案中挑选出一些候选区域，然后通过一个卷积神经网络（CNN）决定哪个区域最有可能包含感兴趣的实例。这种思想在结合传统的CNNs网络可以获得不错的效果。

## 引言

用于检测图像中目标的最先进的方法，例如R-CNN系列，YOLO和SSD，可以被视为相同范式的变体：**提出一定数量的候选图像区域（要么动态地要么来自固定池（a fixed pool）），然后使用卷积神经网络（CNN）来确定这些区域中的哪一个紧密地包围着感兴趣对象的实例**。这种策略（我们称之为**提议和验证（P&V））** 的一个重要优势是，它与标准CNN一起工作得特别好。然而，P&V也有几个显着的缺点：

1. 首先是矩形提案只能**近似**对象的实际形状
1. 尤其在分割目标时，需要两步（two-step）的方法，如在Mask R-CNN中，首先使用矩形等简单形状检测对象实例，然后才能将检测细化为像素精确的分割。

一种可以克服这种限制的P&V的替代方案是用相应目标出现的标识符直接标记各个像素。这种方法，我们称之为实例着色（instance coloring，IC），可以通过预测单个标签映射来有效地表示任意数量的任意形状的目标。因此，IC原则上比P&V有效得多。IC的另一个吸引力在于它可以表示为图像到图像的回归问题，类似于其他图像理解任务，例如去噪、深度和法线估计（denoising, depth and normal estimation）以及语义分割。因此，该策略可以允许更容易地构建统一的体系结构，同时处理实例分割和其他任务。

尽管理论上IC有很多好处，但是，P&V方法目前在总体精确度方面占主导地位。本文的目的是探讨造成这种差距的一些原因，并提出一些替代的解决方案（workarounds）。部分问题可能与密集标签的性质（nature）有关。给目标着色的最明显的方法是给它们编号，并用它们相应的编号来“画”它们。然而，后者是全局操作，因为它需要知道图像中的所有目标。**CNN是局部的并且具有平移不变性（local and translation invariant），因此可能不适合直接枚举**。因此，几位作者探索了更适合卷积网络的替代着色方案。一种流行的方法将任意颜色（通常以实向量的形式表示）指定给每个目标的出现（object occurrence），唯一的要求是不同的颜色应该用于不同的目标。所得到的颜色亲和性（affinities）然后可经由非卷积算法很容易地枚举目标。

在本文中，我们认为即使是后一种技术也不足以使IC服从（amenable to）CNN的计算。原因是，由于CNN是平移不变的，它们仍然会为目标的相同副本分配相同的颜色，使得副本无法通过卷积着色来区分，由于实际中大多数CNN的感受野大小几乎与整个图像一样大，因此这个说法始终是有限度的（holds in the limit）；然而，它表明网络的卷积结构至少并不能自然地适合IC。

![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMjI5ODY2MDktNTE0MTFiM2UtOTk0Ny00ZjJlLWJmMWMtNWQ3MGI3OTk1MDVhLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=377&name=image.png&originHeight=377&originWidth=818&search=&size=269607&status=done&width=818)

为了克服这个问题，我们建议用于IC的架构不应该是平移不变的；虽然这可能看起来与卷积网络有很大的不同，但我们也**表明对标准CNN的一个小的修改可以克服这个问题**。我们通过定义半卷积操作符（semi-convolutional operators），**将从卷积网络提取的信息与关于像素全局位置的信息混合**。我们训练后者，使得运算符的响应对于属于同一目标实例的所有像素是相似的，使得这种嵌入自然地适合于IC。我们展示了，如果混合函数是相加的（additive），那么得到的算子与Hough投票和相关的检测方法有一些相似之处。在扩展嵌入表示以**合并捕获外观线索的标准卷积响应**之后，我们使用它来诱导像素亲和性，并展示后者如何被解释为双边核的引导版本（a steered version of a bilateral kernel）。最后，我们展示了如何将这种亲和性整合到诸如Mask RCNN这样的结构中。

我们用几个实验来评估我们的方法。我们从研究我们方法在简单合成数据上的极限属性开始。然后，我们展示了我们的半卷积特征提取器可以成功地与最先进的方法相结合，以解决**包含重叠和关节生物体的生物图像**的解析问题。最后，我们将后者应用于一个标准实例分割基准Pascal VOC。在所有这些情况下都表明，使用半卷积特征可以提高最先进的实例分割方法（如Mask RCNN）的性能。

## 相关工作

The past years have seen large improvements in object detection, thanks to powerful baselines such as Faster-RCNN [47], SSD [39] or other similar approaches [11, 45, 35], all from the propose & verify strategy. Following the success of object detection and semantic segmentation, the challenging task of instance-level segmentation has received increasing attention. Several very different families of approaches have been proposed.

### Proposal-based instance segmentation

1. While earlier methods relied on bottom-up segmentations [18, 9], the vast majority of recent instance-level approaches combine segment proposals together with powerful object classifiers.
    1. In general, they implement a multi-stage pipeline that first generates region proposals or class agnostic boxes, and then classifies them [30, 20, 7, 43, 10, 44, 33].
    1. For instance DeepMask [43] and follow-up approaches [44, 8] learn to propose segment candidates that are then classified.
    1. The MNC approach [10], based on Faster-RCNN [47], repeats this process twice [10] while [33] does it multiple times.
    1. [22] extends [10] to model the shape of objects.
    1. The fully convolutional instance segmentation method of [32] also combines segmentation proposal and object detection using a position sensitive score map.
2. Some methods start with semantic segmentation first, and then cut the regions obtained for each category into multiple instances [26, 4, 38], possibly involving higher-order CRFs [3].
2. Among the most successful methods to date, Mask-RCNN [23] extends FasterR-CNN [47] with a small fully convolutional network branch [41] producing segmentation masks for each region of interest predicted by the detection branch. Despite its outstanding results, Mask-RCNN does not come without shortcomings: **it relies on a small and predefined set of region proposals and non-maximum suppression, making it less robust to strong occlusions, crowded scenes, or objects with fundamentally non-rectangular shapes**.

### Instance-sensitive embeddings

1. Some works have explored the use of pixel-level embeddings in the context of clustering tasks, employing them as a soft, differentiable proxy for cluster assignments [54, 21, 15, 12, 42, 28]. This is reminiscent of unsupervised image segmentation approaches [49, 16]. It has been used for body joints [42], semantic segmentation [1, 21, 6] and optical flow [1], and, more relevant to our work, to instance segmentation [15, 12, 6, 28].
1. The goal of this type of approaches is to **bring points that belong to the same instance close to each other in an embedding space**, so that the decision for two pixels to belong to the same instance can be directly measured by a simple distance function. **Such an embedding requires a high degree of invariance to the interior appearance of objects**.
1. Among the most recent methods,
    1. [15] combines the embedding with a greedy mechanism to select seed pixels, that are used as starting points to construct instance segments.
    1. [6] connects embeddings, low rank matrices and densely connected random fields.
    1. [28] embeds the pixels and then groups them into instanceswith a variant of mean-shift that is implemented as a recurrent neural network.
    1. All these approaches are based on convolutions, that are **local and translation invariant by construction**, and consequently are inherently ill-suited to distinguish several identical instances of the same object.
    1. A recent work [25] employs position sensitive convolutional embeddings that regress the location of the centroid of each pixel's instance.
    1. We mainly differ by allowing embeddings to regress an unconstrained representative point of each instance.
4. Among other approaches using a clustering component,
    1. [50] leverages a coverage loss and  [56, 51, 52] make use of depth information.
    1. In particular, [52] trains a network to predict each pixel direction towards its instance center along with monocular depth and semantic labeling. Then template matching and proposal fusion techniques are applied.

### Other instance segmentation approaches

1. Several methods [43, 44, 34, 24] move away from box proposals and use Faster-RCNN [47] to produce "centerness" scores on each pixel instead. They **directly predict the mask of each object in a second stage**. An issue with such approaches is that objects do not necessarily fit in the receptive fields.
1. Recurrent approaches sequentially generate a list of individual segments.
    1. For instance, [2] uses an LSTM for detection with a permutation invariant loss while uses an LSTM to produce binary segmentation masks for each instance.
    1. [46] extends [48] by refining segmentations in each window using a box network.
    1. These approaches are slow and do not scale to large and crowded images.
3. Some approaches use watershed algorithms.
    1. [4] predicts pixel-level energy values and then partition the image with a watershed algorithm.
    1. [26] combines a watershed algorithm with an instance aware boundary map. Such methods create disconnected regions, especially in the presence of occlusion.

## 方法

### Semi-convolutional networks for instance coloring

基本符号：

1. $\mathbf{x} \in \mathcal{X} = \mathbb{R}^{H \times W \times 3}$表示一个三通道的图像
1. $u \in \Omega = \{1, \cdots, H\} \times \{1, \cdots, W\}$表示图像中的一个像素
1. $\mathcal{S}_x=\{ S_1, \cdots, S_{K_\mathbf{x}} \} \subset 2^\Omega$表示图像像素针对各个实例类别的分割区域
1. $S_0=\Omega-\cup_kS_k$表示背景分割区域

The regions as well as their number are a function of the image and the goal is to predict both.

这里主要研究将实例分割视为对单个像素分类的问题，其中这里的“类”是指一个实例（instance）。也就是寻找一种函数映射关系：$\Phi:\mathcal{X} \rightarrow \mathcal{L}^{\Omega}$，关联每个像素u到一个特定的标签$\Phi_u(\mathbf{x}) \in \mathcal{L}$，使得作为一个整体而言，标签编码了对应的分割$\mathcal{S}_x$。直观地说，这可以通过用不同的颜色绘制不同的区域（即像素标签）来实现，从而使对象在后处理中很容易恢复。我们称这个过程为实例着色（IC）。

一个受欢迎的IC方法是使用实向量$\mathcal{L}=\mathbb{R}^d$作为颜色的表示，然后要求不同区域的颜色充分分离。正式地，这里应该有一个边界参数$M > 0$，所以有如下公式：
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1Njk5MjY4OTIyOTktMjFmOGI5NzQtNGJlZi00MGJlLWI1ZDUtZTQ3NjE0NmEwYmNkLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=65&name=image.png&originHeight=65&originWidth=696&search=&size=12492&status=done&width=696)
如果是这种情况，那么聚类颜色通常会重建区域。


不幸的是，对于一个卷积操作符$\Phi$，很难满足约束(1)或类似的。虽然这在小节 _the convolutional coloring dilemma_ 中得到了正式证明，但就目前而言，直觉已经足够：如果图像包含相同对象的副本，则平移不变的卷积网络将会为每个副本分配相同的颜色。


如果卷积运算是不合适的，那么我们必须抛弃卷积运算。虽然这听起来很复杂，但我们建议对卷积进行非常简单的修改就可以满足需求了（may suffice）。尤其是，如果$\Phi_u(\mathbf{x})$是卷积操作在像素u位置的输出，那么我们可以构造一个非卷积响应，通过混合它和像素位置信息。数学上可以定义半卷积操作为：
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1Njk5ODMzOTExNzAtZThkODQ5NmYtZDgyYS00ZGZjLWIxNWYtZmFkNDIwYjUyNmEyLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=26&name=image.png&originHeight=26&originWidth=689&search=&size=3864&status=done&width=689)
 这里的$f:\mathcal{L} \times \Omega \rightarrow \mathcal{L}'$是一个合适的混合函数。作为这种算子的主要例子，我们考虑一种特别简单的混合函数，即加法。于是等式2可以特殊化为：
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1Njk5ODM5NzM3NzUtZWUxMWY4OTQtNmVmZC00NWJhLThiYzktN2YwZGE0ZmIxZTU4LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=27&name=image.png&originHeight=27&originWidth=691&search=&size=4787&status=done&width=691)
（也就是卷积的输出和对应的位置信息的加和，这里约束了输出的卷积特征是一个二元矢量）虽然这种选择是有限制的，但它的好处是有一个非常简单的解释。假设结果嵌入可以完美地分离实例，在这种情况下，会有$\Psi_u(\mathbf{x})=\Psi_v(\mathbf{x}) \Leftrightarrow \exists k:(u,v)\in S_k$（也就是说，对应同一实例内部的u和v的位置上计算的混合表示应该是一致的，也就是说即使添加了位置关系，也要保证混合后的结果是一样的，这里没有提位于不同实例的情况），然后对于所有的位于区域$S_k$中的像素，可以写成如下形式：
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1Njk5ODQ2ODAwOTctNTNmY2QwYzAtMGNiMi00MzgwLTgwMzgtNjhlMmJkYjZmZDI5LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=22&name=image.png&originHeight=22&originWidth=691&search=&size=4310&status=done&width=691)
 这里的$c_k \in \mathbb{R}^2$是一个实例特定的点（instance-specific point）。换句话说，我们看到对实例分割学习这个semi-convolutional嵌入的影响是去预测一个位移场（displacement field）$\Phi(\mathbf{x})$，这可以将目标实例的所有像素映射到一个实例特定（instance-specific）的质心$c_k$。位移场的图解可以在图2中找到。

![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1Njk5ODU0MjY5OTgtNTY3ZjMzNjYtM2I5NC00ZmY2LThkM2YtN2YwOWI5MzdkYjc0LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=347&name=image.png&originHeight=347&originWidth=698&search=&size=380794&status=done&width=698)


#### Relation to Hough voting and implicit shape models

等式3和等式4让人联想到计算机视觉中著名的检测方法：Hough voting和implicit shape model（ISM）。这两种方法将图像patches映射到对于可能的目标出现的参数$\theta$的投票上。简单的情况下$\theta \in \mathbb{R}^2$可以使目标的质心，并且在投票的时候可能和等式4有着相似的形式。

这确立了基于投票的目标检测方法和基于着色的实例分割方法之间清晰的关联。与此同时也有一些显著的差异。

1. 首先，**这里的目标是对像素分组，而不是重构一个目标实例的参数（诸如其质心和尺寸）**。等式3可以这样解释，但更一般的形式的等式2却不能。
1. 第二，在方法如Hough或ISM中，质心被定义为目标的实际中心。而这里的质心$c_k$并没有明确的含义，但可以被自动推断为一个有用但任意的参考点。
1. 第三，在传统的投票方案中，投票集成提取自单个（individual）的patches的局部信息。这里$\Phi_u(x)$的感受野大小可能是足以包含整个对象，或着更多。等式2和3的目标并不是去汇集局部的信息，而是要去解决一个代表性的问题（representational issue）。

### Learning additive semi-convolutional features

学习等式2的semi-convolutional特征可以从许多不同的方面来制定。这里我们使用一个启发自semantic instance segmentation with a discriminative loss function的简单直接的形式，并且为每个图像$\mathbf{x}$和实例$S \in \mathcal{S}$构建一个损失函数，通过考虑对于分割$S$中的像素$u$的嵌入表示和该分割中的所有嵌入的均值（质心）之间的距离来表示：
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1Njk5ODg0MTE0NTYtYTdlNWNhZWYtMGIwMi00ZWZhLWJkMmYtMWZlMGY4OTUyNmNhLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=63&name=image.png&originHeight=63&originWidth=687&search=&size=7642&status=done&width=687)
（可以看出了，这里只是计算了L2范数，是开了根号的那个）注意，虽然这个量**类似于对每个分割的嵌入值的方差**，但它并不像，因为没有平方（更像是标准差）；这也被发现是更加鲁棒的。


注意，这个损失函数相较于余量条件（margin condition，式子1）和在[12]中提出的损失更加简单（这更类似于1）。特别地，提出的这个损失仅包括“吸引（attractive）”力，这鼓励对于每个segment的嵌入一致于特定的均值，**但不明确鼓励不同的segments要被分配为不同的嵌入值**。虽然这也可以一并设置，但是我们发现，通过最小化等式5，也是可以令人满意的学习到不错的加式semi-convolutional嵌入。


### Coloring instances using individuals’ traits

实际上，很少有图像包含某个目标的精确复制品。相反，更典型的情况是，不同的事件（occurrences）具有一些不同的个体特征（individual traits）。例如，不同的人通常有不同的穿着方式，包括不同的颜色。在实例分割中，可以使用这样的线索来区分不同的实例。更多的是，这些线索可以通过传统的卷积操作来提取。

为了将这些线索结合到我们的加性半卷积表达，我们仍然考虑式子：$\Psi_u(x) = \hat{u} + \Phi_u(x)$。然而，我们将$\Phi_u(\mathbf{x}) \in \mathbb{R}^d$放松到两维以上，$d>2$。更进一步，我们定义$\hat{u}$作为$u$的像素坐标，具体表示如下：
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMDMwOTQ0NDktNTE5ZGJjZTQtOTU3My00NWUyLWJlY2UtNTBmY2RmYTQwYmU2LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=34&name=image.png&originHeight=34&originWidth=689&search=&size=4205&status=done&width=689)
前两项表示像素$u$的x和y的坐标。在这种方法中，后面的$d-2$维的嵌入表示用作传统的卷积特征，可以正常地提取实例特定的性状（traits）。

### Steered bilateral kernels

像素嵌入向量$Ψ_u(\mathbf{x})$最终必须解码为一组图像区域。另外，有几种可能的策略，可从简单的K均值聚类开始。这一小节，我们考虑将嵌入转换到一个两个像素之间的关联矩阵（affinity matrix），因为后者可以被用在许多算法中。

为了定义像素$u，v$之间的affinity，首先考虑高斯核：
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMDM1MDkyNzUtNmUzOGZjMTMtNjkzYS00MDZhLTgyZGQtZDRlNjE4NWEwM2NhLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=48&name=image.png&originHeight=48&originWidth=688&search=&size=7184&status=done&width=688)
如果式子6中增强的嵌入表示使用在定义$\Psi_u(\mathbf{x}) = \hat{u} + \Phi_u(\mathbf{x})$中，则可以讲$\Phi_u(\mathbf{x})$分割为一个几何（geometric）部分$\Phi_u^g(\mathbf{x}) \in \mathbb{R}^2$和一个外观（appearance）部分$\Phi_u^a(\mathbf{x}) \in \mathbb{R}^{d-2}$，扩展这个核可以表示为如下部分：
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMDM3NzU4MjQtMTNjNWYzMDktODg0Zi00YmUzLWFiOTYtNTBmMDM5OTg1NzQ0LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=75&name=image.png&originHeight=75&originWidth=692&search=&size=10170&status=done&width=692)
让人感兴趣的是将其与双边核定义进行比较：
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMDM5MzE2MjUtNGUxMjlkNWMtYjdkNy00ZGU0LTk3MTctOGIwZDdiMzFiYzY0LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=55&name=image.png&originHeight=55&originWidth=690&search=&size=8970&status=done&width=690)
> 双边核中，通常使用$\Phi^a_u(x)=x_u \in \mathbb{R}^3$作为RGB三元组来作为外观特征。

双边核在许多应用中非常流行，包括图像滤波和[均值漂移聚类](https://www.chardlau.com/mean-shift/)（mean shift clustering）。双边核的思想是考虑像素如果在空间和外观上都很接近，就认为它们是相似的。在这里，我们展示了核（式子8），而核（式子7）可以被解释为这个核的一般化形式，在这个核中，空间位置被网络引导steered（扭曲distorted），从而将属于同一个目标实例的像素移到更近的地方。


在这些核的实际实现中，例如为了平衡空间和外观的成分（component），向量在比较之前应该重新放缩。在我们的例子中，由于嵌入是端到端训练的，所以网络可以学习自动处理这种平衡，但实际上式子4隐含地定义了核空间成分的放缩。因此，我们修改了式子7，主要在两个方面：

1. 引入可学的标量参数$σ$
1. 考虑拉普拉斯而不是高斯核

![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMDUwMTY5NTEtMTM2MWE2YzYtMTY3ZS00N2JjLTk5YjQtNTIyNTQzZmE0NzZjLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=55&name=image.png&originHeight=55&originWidth=691&search=&size=7477&status=done&width=691)
这个核对于异常点更加鲁棒（因为它使用了欧氏距离而不是它的平方），并且仍然是正定的。

下一节将会展示这个核怎样可以用来处理实例着色问题。

### Semi-convolutional Mask-RCNN

我们提出的半卷积框架非常通用，可以与许多现有方法相结合。在这里，我们将描述如何将其与当前实例分割领域的最新技术——Mask RCNN（MRCNN）框架相结合。

MRCNN是基于RCNN的propose&verify策略，并首先生成一组矩形区域$\mathcal{R}$，其中每个矩形$R∈ \mathcal{R}$紧紧包围一个实例候选区域。然后，一个全卷积网络（FCN）在每个区域内产生前景/背景分割。在实践中，它使用前景得分logit值$s(u_i) \in \mathbb{R}$为$R$中的每个像素$u_i$都标记了标签。然而，对于交接的目标或遮挡的场景，这不是最佳策略，因为标准FCN很难执行单个的前景/背景预测。因此，我们利用我们的像素级平移敏感的嵌入来提高预测$s(u_i)$的质量。

#### Extending MRCNN

我们的方法基于两个直觉：

1. 首先，一些点比其他点更容易被识别为前景
1. 第二，一旦确定了一个种子点，它与其他像素的亲和力（affinity）可以用来分割前景区域

在实践中，我们首先使用MRCNN前景置信度得分图$\mathbf{s}=[s(u_1),\cdots, s(u_{|R|})]$在每个区域$R$中确定一个种子像素$u_s$。我们选择最自信的种子点记作$u_s=argmax_{1 \le i \le |R|}s(u_i)$，在为种子提取嵌入$\Psi_{u_s}$以及针对区域中的像素$u_i$的$\Psi_{u_i}$之后，评估steered bilateral kernel$K_σ(u_s,u)$，然后定义更新分数$\hat{s}(u_i)=s(u_i)+logK_{\sigma}(u_s,u_i)$。为了改进数值稳定性，分数和核函数的组合是在对数空间中进行的。最后每个前景的概率按照[Mask r-cnn]中的方法使用$sigmoid(\hat{s}(u_i))$获取。

整个体系结构——区域选择机制、前景预测和像素级嵌入——都是端到端训练。为了可微性，需要做以下修改：我们需要用得分上的柔性最大值$\mathbf{p}_s=softmax(\mathbf{s})$来替换取最大操作，并且我们获得种子嵌入$Ψ_{u_s}$作为在概率密度$p_s$下，嵌入$\Psi_u$的期望。网络优化器最小化，MRCNN的损失，以及图像级嵌入损失$\mathcal{L}(Ψ|\mathbf{x},\mathcal{S})$，并且进一步附加一个二次二值交叉熵损失，这类似于MRCNN的掩膜预测器，最小化输出$K_{\sigma}(u_s,u_i)$和真值实例掩膜之间的二值交叉熵损失。

semi-convolutional特征$Ψ_u$的预测器，被实现为一个浅层子网络的输出，在所有的FPN层之间共享。这个子网络由256-channel 1×1卷积滤波器（跟着ReLU和最后的一个3×3卷积滤波器），最终会产生一个$D=8$维的嵌入$Ψ_u$。由于RPN分量对底层FPN表示中的扰动过于敏感，我们将浅层子网生成和被共享FPN张量接收的梯度缩小了10倍。

### The convolutional coloring dilemma

在本节中，我们将证明卷积运算在解决实例分割问题方面的一些性质。为了做到这一点，我们需要从形式化问题开始。

考虑一个图像信号$\mathbf{x}:\Omega \rightarrow\mathbb{R}$，这里的域$\Omega$是$\mathbb{Z}^m$或者$\mathbb{R}^m$。再分各种，给定一族这样的信号$\mathbf{x} \in \mathcal{X}$，其中的每一个都是关联到一个特定关于域$\Omega$的分区$\mathcal{S}_\mathbf{x}=\{ S_1, \cdots, S_{K_\mathbf{x}} \}$。问题的最终目标是去构造一个分割算法$\mathcal{A} : \mathbf{x} \mapsto \mathcal{S}_{\mathbf{x}}$来计算这个函数。我们看一个特殊的情况，通过假定每个点$u \in \Omega$赋一个标签$\Phi_u(\mathbf{x}) \in \mathcal{L}$来预处理信号。进一假定这个标记操作$\Phi$是一个**局部的并且平移不变**的操作，因此可以使用卷积神经网络来实现。
> 这里提到的“局部的”和“平移不变”的表述：
> 如果$\Phi_{u}(\mathbf{x}(\cdot-\tau))=\Phi_{u-\tau}(\mathbf{x})$对于所有的变换$\tau \in \Omega$都成立，则说$\Phi$是平移不变的。
> 如果存在一个大于0的常量$M$，使得$x_u=x_{u'}$对于所有的$|u-u'|<M$都成立，可以推出$\Phi_{u}(\mathbf{x})=\Phi_{u}^{\prime}(\mathbf{x})$，则说$\Phi$是局部的。


有两类算法可以用来以这种方式分割信号。

#### Propose & verify

第一类算法提交所有可能的区域$S_r⊂Ω$（使用索引变量$r$），标记函数$Φ_r(\mathbf{x})∈\{0,1\}$验证哪些属于分割$\mathcal{S}_\mathbf{x}$（即$Φ_r(\mathbf{x})=1⇔S_r∈S_\mathbf{x}$）。因为在实践中不可能测试Ω的所有可能的子集，这样的算法必须专注于一组较小的提案区域。一个典型的选择是考虑所的有平移正方形（或矩形）$S_u=[-H, H]^m+u$。因为索引变量$u \in \Omega$现在是一个转换，操作符$\Phi_u(\mathbf{x})$有着前面讨论的形式，尽管并不必须是局部的或者平移不变的。

#### Instance coloring

第二类算法直接使用对应区域的索引来对像素着色，例如$Φ_u(\mathbf{x}) = k⇔u∈S_k$。不同于$P\&V$方法，这可以有效的表示任意的形状。然而这里的映射$Φ$需要隐式地决定为每一个区域分配哪一个数字，这是一个全局操作。数位作者试图使其更适合于卷积网络。一种流行的方法[15,12]是任意给像素着色（例如使用向量嵌入），以便在相同区域内为像素分配相似的颜色，在不同区域之间使用不同的颜色，公式1中已经详细说明。

#### Convolution coloring dilemma

这里我们展示前面讨论的一些变体，IC方法不能与卷积操作符一起用，即使在和$P\&V$一起工作的情况下。
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMjAxMDIyOTItNWE5ODFkOTQtZTA3MS00MWU5LWJjMzMtYWFkMzdjODlkYmFjLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=92&name=image.png&originHeight=92&originWidth=352&search=&size=6294&status=done&width=352)
我们通过考虑一个简单的1D的例子。让$\mathbf{x}$表示周期为2的信号（$x_{u+2}=x_u$），对于$u \in [-1,1]$时，信号通过$x_{u}=\min (1-u, 1+u)$（可见上图给出了部分长度的信号表示）给出。假设关联到$\mathbf{x}$的分割是$\mathcal{S}=\{[-1,1]+2k,k\in \mathbb{Z}\}$。付过我们假设对于一个基于着色的算法的必要条件是，至少一些区域会被赋予不同的颜色，我们看到这不能通过卷积操作来实现。事实上，因为$\mathbf{x}$的周期性，任何平移不变的函数都将最终给像素$2k, k \in \mathbb{Z}$赋予相同的颜色。

另一方面，这个问题可以通过使用提案集合$\{[-1,1]+u, u \in \Omega\}$来使用$P\&V$方法以及局部的且平移不变的验证函数$\Phi_{u}(\mathbf{x})=\left[x_{u}=1\right]$（这可以检测每个区域的中心位置）来解决。

后者是卷积着色难题（dilemma）的一个极端例子：即，局部和平移不变运算符会自然地将相同的颜色分配给对象的相同副本，即使它们出现在不同的位置（可以比较有趣的同时期的工作[37]，他们探索了相关卷积的难题）。

#### Solving the dilemma

解决着色难题可以通过使用不具有平移不变性的操作来实现。在上面的反例（counterexample）中，这可以通过使用semi-convolutional函数$\Phi_{u}(x)=u+\left(1-x_{u}\right) \dot{x}_{u}$来实现。这可以很容易的展示出，$\Phi_{u}(x)=2 k$通过移动每个点$u$到最近的区域的中心，来为每个像素$u \in S_k=[-1, 1]+2k$（这里可以看前图中对应的位置，当$k=0$的时候，$u$属于$[-1,1]$的范围内，此时对应的分割为$S_0$；当$k=1$的时候，$u$属于范围$[2*1-1, 2*1+1]=[1, 3]$内，此时对应的分割为$S_1$）着色，使用对应区域的索引的两倍。这是可行的，因为这样的位移可以通过局部观察基于信号的形状来进行计算。

## 实验

We first conduct experiments on synthetic data in order to clearly demonstrate
inherent limitations of convolutional operators for the task of instance segmentation. In the ensuing parts we demonstrate benefits of the semi-convolutional operators on a challenging scenario with a high number of overlapping articulated instances and finally we compare to the competition on a standard instance segmentation benchmark.

### Synthetic experiments

We suggested that convolution operators are unsuitable for instance segmentation via coloring, but that semi-convolutional ones can do. These experiments illustrate this point by learning a deep neural network to segment a synthetic image xS where object instances correspond to identical dots arranged in a regular grid (fig. 3 (a)).

1. We use a network consisting of a pretrained ResNet50 model truncated after the Res2c layer, **followed by a set of 1×1 filters that, for each pixel u, produce 8-dimensional pixel embeddings $Φ_u(x_S)$ or $\Psi_u(x_S)$**.
1. **We optimize the network by minimizing the loss from eq. (5) with stochastic gradient descent.**
1. Then, the embeddings corresponding to the foreground regions are extracted and clustered with the k-means algorithm into K clusters, where K is the true number of dots present in the synthetic image.

![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMjE1NTk5MDMtY2Q5NzJiMmEtYjRmZi00NWU2LThjMzQtYzA1YzlmYmIxMGEwLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=473&name=image.png&originHeight=473&originWidth=802&search=&size=160729&status=done&width=802)

Fig. 3 visualizes the results.

- Clustering the features consisting of the position invariant convolutional embedding $Φ_u(x_S)$ results in nearly random clusters (fig. 3 (c)).
- On the contrary, the semi-convolutional embedding $Ψ_u(x_S) = Φ_u(x_S)+u$ allows to separate the different instances almost perfectly when compared to the ground truth segmentation masks (fig. 3 (d)).

### Parsing biological images

The second set of experiments considers the parsing of biological images. **Organisms to be segmented present non-rigid pose variations, and frequently form clusters of overlapping instances, making the parsing of such images challenging.** Yet, this scenario is of crucial importance for many biological studies.

#### Dataset and evaluation

![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMjI2MDY0MzYtODJmOTQxM2YtY2NhNC00ZjY1LWJkYzEtMGVlYzA4ODhkZDNjLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=276&name=image.png&originHeight=276&originWidth=805&search=&size=60562&status=done&width=805)

We evaluate our approach on the C. Elegans dataset (illustrated fig. 4), a subset of the Broad Biomedical Benchmark collection [40].

- The dataset consists of 100 bright-field microscopy images.
- Following standard practice [53,55], we operate on the binary segmentation of the microscopy images.


**However, since there is no publicly defined evaluation protocol for this dataset, a fair numerical comparison with previously published experiments is infeasible**. We therefore compare our method against a very strong baseline (MRCNN) and adopt the methodology introduced by [55] in which the dataset is divided into 50 training and 50 test images. We evaluate the segmentation using average precision (AP) computed using the standard COCO evaluation criteria [36]. We compare our method against the MRCNN FPN-101 model from [23] which attains results on par with state of the art on the challenging COCO instance segmentation task.

#### Results

![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMjI2MTc3NTgtYjM1MmM3NTQtZjI5My00NjUxLTk3YzYtMDU4NTJkZDU2MDNjLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=159&name=image.png&originHeight=159&originWidth=804&search=&size=31164&status=done&width=804)

The results are given in table 1. We observe that the semi-convolutional embedding $Ψ_u$ brings improvements in all considered instance segmentation metrics. The improvement is more significant at higher IoU thresholds which underlines the importance of utilizing position sensitive embedding in order to precisely delineate an instance within an MRCNN crop.

### Instance segmentation

![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMjI5MzgyMzAtZTYzMTYwNTAtMDA3OC00Zjc3LTlmNmItOTUzM2I2M2EwNWI2LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=618&name=image.png&originHeight=618&originWidth=806&search=&size=751885&status=done&width=806)

The final experiment compares our method to competition on the instance segmentation task on a standard large scale dataset, PASCAL VOC 2012 [14].

As in the previous section, we base our method on the MRCNN FPN-101 model. Because we observed that the RPN component is extremely sensitive to changes in the base architecture, we employed a multistage training strategy.

1. First, MRCNN FPN-101 model is trained until convergence
1. then our embeddings are attached and fine-tuned with the rest of the network.
1. We follow [23] and learn using 24 SGD epochs, lowering the initial learning rate of 0.0025 tenfold after the first 12 epochs.
1. Following other approaches, we train on the training set of VOC 2012 and test on the validation set.

#### Results

![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTkyMzE0LzE1NzAwMjI3NDEzMTUtYTJlNzdmY2EtY2NkYi00YjRkLTk2NzctYzZhNmQ3ODIxNmM3LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=355&name=image.png&originHeight=355&originWidth=804&search=&size=65373&status=done&width=804)

The results are given in table 2. Our method attains state of the art on PASCAL VOC 2012 which validates our approach. We further compare in detail against MRCNN in table 3 using the standard COCO instance segmentation metrics from [36]. Our method outperforms MRCNN on the considered metrics, confirming the contribution of the proposed semi-convolutional embedding.

## 结论

这篇文章汇总，我们考虑了针对实例分割的密集像素嵌入，从依赖于平移不变卷积神经网络的标准方法出发，我们提出了半卷积算子，只需对卷积算子进行简单的修改即可获得半卷积算子。除了它们的理论优势外，我们还通过经验证明，它们更适合于区分同一目标的几个相同实例，并且与标准的Mask RCNN方法互补。

## 参考链接

- 论文：[https://arxiv.org/abs/1807.10712](https://arxiv.org/abs/1807.10712)
- 分析：[https://blog.csdn.net/chao_shine/article/details/85927631](https://blog.csdn.net/chao_shine/article/details/85927631)
- 代码：[https://github.com/christianlandgraf/keras_semiconv](https://github.com/christianlandgraf/keras_semiconv)
- 机器学习：Mean Shift聚类算法：[https://www.chardlau.com/mean-shift/](https://www.chardlau.com/mean-shift/)
- 总结一下遇到的各种核函数~：[https://blog.csdn.net/wsj998689aa/article/details/47027365](https://blog.csdn.net/wsj998689aa/article/details/47027365)
