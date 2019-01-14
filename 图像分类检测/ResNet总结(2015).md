# ResNet(2015)

* [ResNet(2015)](#resnet2015)
  * [前言](#前言)
  * [概要](#概要)
  * [构思](#构思)
    * [深度](#深度)
    * [收敛](#收敛)
    * [退化](#退化)
    * [解决](#解决)
    * [相关工作](#相关工作)
      * [残差表示](#残差表示)
      * [快捷链接](#快捷链接)
  * [新意](#新意)
    * [差异](#差异)
    * [结构](#结构)
    * [例子](#例子)
    * [总结](#总结)
  * [架构](#架构)
      * [简单网络](#简单网络)
      * [残差网络](#残差网络)
  * [实际](#实际)
    * [预处理](#预处理)
    * [ImageNet分类](#imagenet分类)
    * [CIFAR-10和分析](#cifar-10和分析)
  * [代码](#代码)

## 前言

2015年12月[ResNet](https://arxiv.org/pdf/1512.03385v1.pdf)发表了, 时间上大概与Inception v3网络一起发表的.

ResNet的结构可以极快地加速超深神经网络的训练, 模型的准确率也有非常大的提升.

ResNet最初的灵感出自这个问题：在不断加神经网络的深度时, 会出现一个Degradation的问题, 即准确率会先上升然后达到饱和, 再持续增加深度则会导致准确率下降.

这并不是过拟合的问题, 因为不光在测试集上误差增大, 训练集本身误差也会增大.

假设有一个比较浅的网络达到了饱和的准确率, 那么**后面再加上几个的全等映射层, 起码误差不会增加**, 即更深的网络不应该带来训练集上误差上升.

**而这里提到的使用全等映射直接将前一层输出传到后面的思想, 就是ResNet的灵感来源**.

## 概要

更深的神经网络更难训练. 我们提出了一种残差学习框架来减轻网络训练, 这些网络比以前使用的网络更深. 我们明确地**将层变为学习关于层输入的残差函数, 而不是学习未参考的函数**. 我们提供了全面的经验证据说明这些**残差网络很容易优化, 并可以显著增加深度来提高准确性**. 在ImageNet数据集上我们评估了深度高达152层的残差网络——比VGG[40]深8倍但仍具有较低的复杂度.

## 构思

### 深度

深度网络自然地将低/中/高级特征[49]和分类器以端到端多层方式进行集成, 特征的“级别”可以通过堆叠层的数量（深度）来丰富.

> 在深度重要性的推动下, 出现了一个问题：学些更好的网络是否像堆叠更多的层一样容易？

### 收敛

回答这个问题的一个障碍是**梯度消失/爆炸[14, 1, 8]**这个众所周知的问题, 它从一开始就阻碍了收敛.

然而, 这个问题**通过标准初始化[23, 8, 36, 12]和中间标准化层[16]**在很大程度上已经解决, 这使得数十层的网络能通过具有反向传播的随机梯度下降(SGD)开始收敛.

![1540206189573](./assets/1540206189573.png)

> 当更深的网络能够开始收敛时, 暴露了一个退化问题：随着网络深度的增加, 准确率达到饱和(这可能并不奇怪)然后迅速下降.

意外的是, **这种下降不是由过拟合引起的(因为训练误差和测试误差都很高), 并且在适当的深度模型上添加更多的层会导致更高的训练误差**, 正如[10, 41]中报告的那样, 并且由我们的实验完全证实. 上图显示了一个典型的例子.

### 退化

退化(训练准确率)表明**不是所有的系统都很容易优化**.

> :question:
>
> 关于这里的退化问题,为什么会出现退化问题?
>
> 为什么如此设计的残差模型可以解决退化问题?

让我们考虑一个较浅的架构及其更深层次的对象, 为其添加更多的层. 存在通过构建得到更深层模型的解决方案：**添加的层是恒等映射, 其他层是从学习到的较浅模型的拷贝**.

> 对任意[集合](https://baike.baidu.com/item/%E9%9B%86%E5%90%88/2908117)A, 如果[映射](https://baike.baidu.com/item/%E6%98%A0%E5%B0%84/20402621)$f:A→A$定义为$f(a)=a$, 即规定A中每个元素a与自身对应, 则称f为A上的**恒等映射**(identical [identity] mapping).

这种构造解决方案的存在表明, 较深的模型不应该产生比其对应的较浅模型更高的训练误差.

> 但是实验表明, 我们目前现有的解决方案无法找到与构建的解决方案相比相对不错或更好的解决方案(或在合理的时间内无法实现).

### 解决

在本文中, 我们通过引入*深度残差学习*框架解决了退化问题.

我们明确地让这些层**拟合残差映射**, 而不是希望每几个堆叠的层直接拟合**期望的基础映射**.

> 让残差块里的层去拟合残差映射,也就是最终的块的输出的总的映射(期望的基础映射)减去输入的部分所得的部分.

形式上的表示:

* **期望的基础映射**表示为$H(x)$
* **堆叠的非线性层**拟合另一个映射$F(x):=H(x)−x$
* **原始的映射**重写为$F(x)+x$

我们**假设残差映射比原始的、未参考的映射更容易优化**.

在极端情况下, 如果一个恒等映射是最优的, 那么**将残差置为零比通过一堆非线性层来拟合恒等映射更容易**.

> 原问题就转化为 *使残差函数(F(x)=H(x)-x)逼近0值*, 而不用直接去拟合一个恒等函数H’(x)(这在实际中是非常难的),
>
> > 转化为优化残差函数的优势
> >
> > https://www.jianshu.com/p/ca6bee9eb888

公式$F(x)+x$可以通过带有“快捷连接”的前向神经网络来实现. 快捷连接[2,33,48]是那些跳过一层或更多层的连接. 在我们的案例中, 快捷连接简单地执行恒等映射, 并将其输出添加到堆叠层的输出.

> 假设有一个比较浅的网络达到了饱和的准确率, 那么**后面再加上几个的全等映射层, 起码误差不会增加**, 即更深的网络不应该带来训练集上误差上升.
>
> **而这里提到的使用全等映射直接将前一层输出传到后面的思想, 就是ResNet的灵感来源**.

恒等快捷连接既不增加额外的参数也不增加计算复杂度. 整个网络仍然可以由带有反向传播的SGD进行端到端的训练.

我们在ImageNet[35]上进行了综合实验来显示退化问题并评估我们的方法. 我们发现：1)我们极深的残差网络**易于优化**, 但当深度增加时, 对应的“简单”网络(简单堆叠层)表现出更高的训练误差；2)我们的深度残差网络可以**从大大增加的深度中轻松获得准确性收益**, 生成的结果实质上比以前的网络更好.

> 关于退化问题的反直觉现象激发了这种重构. 正如我们前面讨论的那样, **如果添加的层可以被构建为恒等映射, 更深模型的训练误差应该不大于它对应的更浅版本**.
>
> 退化问题表明求解器**通过多个非线性层来近似恒等映射可能有困难**.
>
> 通过残差学习的重构, 如果恒等映射是最优的, 求解器可能**简单地将多个非线性连接的权重推向零来接近恒等映射**.
>
> 在实际情况下, 恒等映射不太可能是最优的, 但是我们的重构可能有助于对问题进行预处理.
>
> *如果最优函数比零映射更接近于恒等映射, 则求解器应该更容易找到关于恒等映射的**抖动**, 而不是将该函数作为新函数来学习. *
>
> > 这里反映出来, ReNet 在学习时**对“波动”更敏感.**
> >
> > https://www.jianshu.com/p/ca6bee9eb888

### 相关工作

#### 残差表示

* 在图像识别中, VLAD[18]**是一种通过关于字典的残差向量进行编码的表示形式**, Fisher矢量[30]可以表示为VLAD的概率版本[18].

  它们都是图像检索和图像分类[4,47]中强大的浅层表示. 对于矢量量化, **编码残差矢量[17]被证明比编码原始矢量更有效**.

* 在低级视觉和计算机图形学中, 为了求解偏微分方程(PDE), 广泛使用的Multigrid方法[3]将系统重构为在多个尺度上的子问题, 其中**每个子问题负责较粗尺度和较细尺度的残差解**.

  Multigrid的替代方法是层次化基础预处理[44,45], 它依赖于表示两个**尺度之间残差向量**的变量.

  已经被证明[3,44,45]**这些求解器比不知道解的残差性质的标准求解器收敛得更快**.

这些方法表明, 良好的重构或预处理可以简化优化.

#### 快捷链接

导致快捷连接[2,33,48]的实践和理论已经被研究了很长时间.

* 训练多层感知机(MLP)的早期实践是**添加一个线性层来连接网络的输入和输出**[33,48].

* 在[43,24]中, **一些中间层直接连接到辅助分类器, 用于解决梯度消失/爆炸**.

* 论文[38,37,31,46]提出了**通过快捷连接实现层间响应, 梯度和传播误差的方法**.

* 在[43]中, 一个“inception”层由**一个快捷分支和一些更深的分支**组成.

* 和我们同时进行的工作, “highway networks” [41, 42]提出了门功能[15]的快捷连接.

  这些门是数据相关且有参数的, 与我们不具有参数的恒等快捷连接相反. 当门控快捷连接“关闭”(接近零)时, highway networks中的层表示非残差函数.

  > 此时输出即为**期望的基础映射**,而非残差.

  相反, 我们的公式总是学习残差函数；我们的恒等快捷连接永远不会关闭, 所有的信息总是通过, 还有额外的残差函数要学习.

  此外, highway networks还没有证实极度增加的深度(例如, 超过100个层)带来的准确性收益.

## 新意

### 差异

残差网络主要存在三点与传统卷积网络不同的地方：

> https://zhuanlan.zhihu.com/p/32173684

1. 残差网络Shortcut  Connections的引入实现了**恒等映射**, 使得数据流可以跨层流动.

2. Idendity mapping的出现使网络相对传统结构而言变得更深.

3. 从残差网络结构中移除一层后并不会对网络的性能产生大的影响, 对于传统网络如VGG-16,VGG-19, 删除任何一层都会大幅降低网络性能.

   这可能与网络多路径的叠加有关, 这样的叠加实现了较好的正则化.

   > 关于这里的表述,个人理解为对于网络的具体依赖减弱, 类似于dropout被认为是一种正则化手段的理解方式来理解.

### 结构

其中ResNet的一个重要的思想是：输出的是两个连续的卷积层, 并且输入时绕到下一层去. 这句话不太好理解可以看下图.

![img](https://chenzomi12.github.io/2016/12/13/CNN-Architectures/resnetb.jpg)

> 我们每隔几个堆叠层采用残差学习. 构建块如图所示.
>
> 在本文中我们考虑构建块正式定义为：$y=F(x,W_i)+x.$ $x$和$y$是考虑的层的输入和输出向量. 函数$F(x,W_i)$表示要学习的残差映射.
>
> 图中的例子有两层, $F=W_2 σ(W_1  x)$中$σ$表示$ReLU$[29], 为了简化写法忽略偏置项. $F+x$操作通过快捷连接和各个元素相加来执行. 在相加之后我们采纳了第二种非线性(即$σ(y)$).
>
> **上述方程中$x$和$F$的维度必须是相等的. **
>
> 如果不是这种情况(例如, 当更改输入/输出通道时), 我们可以通过快捷连接执行线性投影$W_s$来匹配维度：$y=F(x,W_i)+W_sx$.
>
> 尽管上述符号是关于全连接层的, 但它们同样适用于卷积层.
>
> 函数$F(x, W_i)$可以表示多个卷积层. 元素加法在两个特征图上**逐通道**进行.
>
> > 这里也就是说, 输出的F的H,W参数也要和x一致.

在2层之后绕过是一个关键, 因为绕过单层的话实践上表明并没有太多的帮助, 然而**绕过2层可以看做是在网络中的一个小分类器**！

> 残差函数F的形式是可变的. 本文中的实验包括有两层或三层的函数F, 同时可能有更多的层. 但如果F只有一层, 方程类似于线性层：$y=W_1x+x$, 我们没有看到优势.

看上去好像没什么感觉, 但这是很致命的一种架构, 因为通过这种架构最后实现了神经网络超过1000层.

### 例子

![img](https://chenzomi12.github.io/2016/12/13/CNN-Architectures/resnetbottleneck.jpg)

* 该层首先使用1x1卷积然后输出原来特征数的1/4, 然后使用3×3的卷积核, 然后再次使用1x1的卷积核.

  但是这次**输出的特征数为原来输入的大小**. 如果原来输入的是256个特征, 输出的也是256个特征.

  这样的结构, 就像Bottleneck Layer那样说的**大量地减少了计算量, 但是却保留了丰富的高维特征信息**.

* ResNet一开始的时候是使用一个7x7大小的卷积核, 然后跟一个pooling层.

* 最后的分类器跟GoogleNet一样是一个pooling层加上一个softmax作为分类器.

### 总结

* ResNet可以被看作并行和串行多个模块的结合
* ResNet上部分的输入和输出一样, 所以看上去有点像RNN, 因此可以看做是一个更好的生物神经网络的模型

## 架构

下图左边是VGG19拥有190万个参数, 右图是34层的ResNet只需要36万个参数, 有更少的滤波器和更低的复杂度.

我们的34层基准有36亿FLOP(乘加), 仅是VGG-19(196亿FLOP)的18%.

**网络架构图**

![Figure 3](http://upload-images.jianshu.io/upload_images/3232548-d9f10353626839c3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**网络细节图**

![Table 1](http://upload-images.jianshu.io/upload_images/3232548-9e352332b837d0d6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上图中,第一行的"xx-layer"表示不同类型的resnet架构对应的结构参数.与前面图对应的是"34-layer".

#### 简单网络

我们简单网络的基准(图中间)主要受到VGG网络[40](图左)的哲学启发. 卷积层主要有3×3的滤波器, 并遵循两个简单的设计规则：

* 对于相同的输出特征图尺寸的层, 具有相同数量的滤波器；
* 如果特征图尺寸减半, 则滤波器数量加倍, 以便保持每层的时间复杂度.

我们通过步长为2的卷积层直接执行下采样. 网络以**全局平均池化层**和具有**softmax的1000维全连接层**结束.

#### 残差网络

基于上述的简单网络, 我们插入快捷连接(图右), 将网络转换为其对应的残差版本.

当输入和输出具有相同的维度时(图右中的实线快捷连接)时, 可以直接使用恒等快捷连接(方程(1)).

当维度增加(图右中的虚线快捷连接)时, 我们考虑两个选项：

1. 快捷连接仍然执行恒等映射, 额外填充零输入以增加维度. 此选项不会引入额外的参数；
2. 使用方程(2)中的投影快捷连接, 使用匹配维度(由1×1卷积完成).

对于这两个选项, 当快捷连接跨越两种尺寸的特征图时, 它们执行时步长为2.

## 实际

### 预处理

1. 调整图像大小, 其**较短的边在[256,480]之间进行随机采样**, 用于尺度增强[40].
2. **224×224裁剪**是从图像或其水平翻转中随机采样, 并**逐像素减去均值[21]**.
3. 使用了[21]中的**标准颜色增强**.
4. 在每个*卷积之后和激活之前*, 我们采用**批量归一化**(BN)[16].
5. 我们按照[12]的方法初始化权重, 从零开始训练所有的简单/残差网络.
6. 我们使用批大小为256的**SGD方法**.
7. 学习速度从0.1开始, 当**误差稳定时学习率除以10**, 并且模型训练高达60×104次迭代.
8. 我们使用的**权重衰减为0.0001, 动量为0.9**.
9. 根据[16]的实践, 我们不使用Dropout[13].

### ImageNet分类

1. 较深的34层简单网络比较浅的18层简单网络有更高的验证误差

2. 18层的简单/残差网络同样地准确, 但18层ResNet收敛更快.

   > 当网络“不过度深”时(18层),目前的SGD求解器仍能在简单网络中找到好的解. 在这种情况下, ResNet通过在早期提供更快的收敛简便了优化.

3. 34层ResNet比18层ResNet更好(2.8％),50/101/152层ResNet比34层ResNet的准确性要高得多(表3和4).

4. 34层ResNet显示出较低的训练误差, 并且可以泛化到验证数据

   > 退化问题得到了很好的解决, 我们从增加的深度中设法获得了准确性收益

5. 没有参数的恒等快捷连接有助于训练
6. 投影快捷连接对于解决退化问题不是至关重要的
7. 恒等快捷连接对于不增加瓶颈结构的复杂性尤为重要
8. 尽管深度显著增加, 但152层ResNet(113亿FLOP)仍然比VGG-16/19网络(153/196亿FLOP)具有更低的复杂度.

### CIFAR-10和分析

1. ResNet的响应比其对应的简单网络的响应更小

   > 这些结果支持了我们的基本动机(第3.1节), 残差函数通常具有比非残差函数更接近零. 我们还注意到, 更深的ResNet具有较小的响应幅度, 如图7中ResNet-20, 56和110之间的比较所证明的.
   >
   > 当层数更多时, 单层ResNet趋向于更少地修改信号.

2. 1202层的网络, 其训练表现不错.

   > 我们的方法显示没有优化困难, 这个1000层网络能够实现训练误差<0.1％(图6, 右图). 其测试误差仍然很好(7.93％, 表6).
   >
   > 这种极深的模型仍然存在着开放的问题.
   >
   > 这个1202层网络的测试结果比我们的110层网络的测试结果更差, 虽然两者都具有类似的训练误差. 我们认为这是因为**过拟合**. 对于这种小型数据集, 1202层网络可能是不必要的大(19.4M).

## 代码

```python
@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    # 创建快捷链接
    # 如果输出的深度和输入相同,则不需要调整快捷链接的深度,下采样即可,
    # 若是深度不匹配,则需要利用1x1卷积来实现通道匹配
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=tf.nn.relu6 if use_bounded_activations else None,
          scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    # 创建卷积层
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              reuse=None,
              scope=None):
  """Generator for v1 ResNet models.
  接受输入inputs和中间的残差块网络结构,进行组合,添加总的开头结尾和快捷链接

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  > This place has shrunk 32 times.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode. If this is set
      to None, the callers can specify slim.batch_norm's is_training parameter
      from an outer slim.arg_scope.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        To use this parameter, the input images must be smaller than 300x300
        pixels, in which case the output logit layer does not contain spatial
        information and can be removed.
    store_non_strided_activations: If True, we compute non-strided (undecimated)
      activations at the last unit of each block and store them in the
      `outputs_collections` before subsampling them. This gives us access to
      higher resolution intermediate activations which are useful in some
      dense prediction problems but increases 4x the computation and memory cost
      at the last unit of each block.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes a non-zero integer, net contains the
      pre-softmax activations.
      > 如果global_pool为False, 则height_out和width_out与相应的height_in和width_in
      > 相比减少了output_stride因子, 否则height_out和width_out都等于1.
      > 如果num_classes为0或None, 则net是最后一个ResNet块的输出, 可能在全局平均池之后
      > 如果num_classes是非零整数, 则net包含pre-softmax激活.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with (slim.arg_scope([slim.batch_norm], is_training=is_training)
            if is_training is not None else NoOpScope()):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError(
                'The output_stride needs to be a multiple of 4.')
            output_stride /= 4

          # 创建resnet最前面的64输出通道的步长为2的7*7卷积
          net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          # 然后接最大池化
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
          # 经历过两个步长为2的层图片缩为1/4

        # 将残差学习模块组生成好
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride,
                                              store_non_strided_activations)
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)

        # 根据标记添加全局平均池化层
        if global_pool:
          # Global average pooling.
          # tf.reduce_mean实现全局平均池化效率比avg_pool高
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
          end_points['global_pool'] = net

        if num_classes:
          # 添加一个输出通道num_classes的1*1的卷积
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
          end_points[sc.name + '/logits'] = net

          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            end_points[sc.name + '/spatial_squeeze'] =

          # 输出网络结果
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points


resnet_v1.default_image_size = 224


def resnet_v1_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
  # Args:：
  # 'block1'：Block名称(或scope)
  # bottleneck：ResNet V2残差学习单元
  # [(256, 64, 1)] * 2 + [(256, 64, 2)]：Block的Args, Args是一个列表.
  # 其中每个元素都对应一个bottleneck,
  # 前两个元素都是(256, 64, 1), 最后一个是(256, 64, 2).
  # 每个元素都是一个三元tuple, 即(depth, depth_bottleneck, stride).
  # (256, 64, 3)代表构建的bottleneck残差学习单元(每个残差学习单元包含三个卷积层)中,
  # 第三层输出通道数depth为256, 前两层输出通道数depth_bottleneck为64, 且中间那层步长3.
  # 这个残差学习单元结构为：
  # [(1*1/s1,64),(3*3/s2,64),(1*1/s1,256)]
resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 store_non_strided_activations=False,
                 reuse=None,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  # 对应于结构34-layer结构图上的情况.每一个block表示一个残差块(with shortcut).
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)


resnet_v1_50.default_image_size = resnet_v1.default_image_size
```
