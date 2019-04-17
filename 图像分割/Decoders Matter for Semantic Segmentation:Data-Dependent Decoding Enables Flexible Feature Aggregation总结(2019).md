# Decoders Matter for Semantic Segmentation:Data-Dependent Decoding Enables Flexible Feature Aggregation

> 论文: https://arxiv.org/abs/1903.02120
> 他人复现: https://github.com/LinZhuoChen/DUpsampling

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552466187381-b49962c8-47dc-4177-8f17-20601b25df23.png#align=left&display=inline&height=275&name=image.png&originHeight=275&originWidth=1232&size=47894&status=done&width=1232)

* [Decoders Matter for Semantic Segmentation:Data-Dependent Decoding Enables Flexible Feature Aggregation](#decoders-matter-for-semantic-segmentationdata-dependent-decoding-enables-flexible-feature-aggregation)
  * [动机](#动机)
  * [贡献](#贡献)
    * [重建误差](#重建误差)
    * [损失函数](#损失函数)
    * [数据依赖上采样](#数据依赖上采样)
    * [自适应温度Softmax](#自适应温度softmax)
    * [灵活的集成策略](#灵活的集成策略)
  * [效果](#效果)
  * [总结](#总结)

## 动机

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552466569544-98a7f66c-f51e-489c-8f7b-b1fdaa5de70f.png#align=left&display=inline&height=557&name=image.png&originHeight=557&originWidth=612&size=93583&status=done&width=612)

语义分割领域最常用的**编解码方案**中, 上采样是一个重要的环节, 用来恢复分辨率. 常用的是, 双线性插值和卷积的配合. 相较于具有一定的棋盘效应的转置卷积, 双线性插值简单快捷, 而且配合后续卷积, 也可以实现和转置卷积类似的效果, 而其他的方法, 如外围补零, 则是会引入过多的冗余无用信息(个人理解, 这个操作[在UNet中使用了](https://github.com/lartpang/Machine-Deep-Learning/blob/master/%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/U-Net:%20Convolutional%20Networks%20for%20Biomedical%20Image%20Segmentation%E7%BF%BB%E8%AF%91(2015).md)).

> 还有一种方案是扩张卷积为主的方案, 早期的deeplab设计中使用的. *Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs.*
> 常见的上采样方式: [https://github.com/lartpang/Machine-Deep-Learning/issues/25#issuecomment-448911117](https://github.com/lartpang/Machine-Deep-Learning/issues/25#issuecomment-448911117)

以当前最优秀的语义分割网络deeplabv3+的结构为例(**结合扩张卷积与编解码结构的方案**). 可以看见, 为了恢复经过下采样之后特征图, 使用了很多的双线性插值.

为了进一步提升当前研究的效果水平, 人们的研究一般是两个方向, 一个是创新, 一个是改进, 当然后者居多. 当前的进步更多是使用了第二个思路, 但是"改进"的思路却也是最让后来的研究者人找到方向的, 也是一般研究者更愿意做的事情. 这篇文章, 就做的是改进的工作.

他们把目光放在了这个最常见的上采样过程上. 文章重点考虑了双线性插值所带来的问题:  过简单的双线性上采样的一个缺点是, 其在准确恢复像素预测方面的**能力有限. **双线性上采样**不考虑每个像素的预测之间的相关性**，因为它是数据独立的.

因此，卷积解码器被需要来产生相对较高分辨率的特征图, 以便获得良好的最终预测. 但是这个需要也会导致两个问题:

1. 编码的整体步长必须通过使用多个**扩张卷积**来进行非常大的降低. 代价是更高的计算复杂性和内存占用，阻碍了大量数据的训练过程和实时应用的部署. 例如deeplabv3+就很慢. (*扩张卷积的主要缺点是计算复杂度和更大的内存需求, 因为这些卷积核的大小以及结果的特征映射变得更大.*)
1. 解码器需要融合来自更低层级的特征. 由于**双线性上采样能力的不足**, 导致最终预测的精细程度, 主要由更低层级的特征的分辨率占据主导地位. 结果, 为了产生高分辨率预测, 解码器必须将高分辨率特征置于低级层次上, 这样的约束限制缩小了特征聚合的设计空间, 因此可能导致次优的特征组合在解码器中被聚合.
> 实验表明，在不考虑特征映射分辨率约束的情况下，可以设计出更好的特征聚合策略。
> 虽然当前很多工作已经花费了很多精力来设计更好的解码器, 但到目前为止, 它们几乎都不能绕过对融合特征的分辨率的限制以及很难更好地进行特征聚合.

## 贡献

为了处理这些问题, 文章提出了一个可学习的"上采样"模块:  DUpsamling. 来替换被广泛使用的双线性插值上采样方法, 恢复特征图的分辨率. 来利用分割标签空间的冗余, 以及准确的恢复像素级预测. 减少了对卷积解码器的精确响应的需要. 因此, 编码器不再需要**过度减少其整体步幅**, 从而大大减少整个分割框架的计算时间和内存占用.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552469396542-24601954-580d-4fca-8f58-582f8877b49b.png#align=left&display=inline&height=585&name=image.png&originHeight=585&originWidth=1233&size=144118&status=done&width=1233)

同时, 由于 DUpsampling 的有效性, 它允许解码器**在合并之前将融合的特征降采样到特征映射的最低分辨率. **这种下行采样不仅减少了解码器的计算量, 更重要的是它**将融合特征的分辨率和最终预测的分辨率解耦**. 这种解耦使解码器能够利用任意特征聚合, 从而可以利用更好的特征聚合, 从而尽可能提高分割性能.

最后, DUpsampling可以通过标准的 1x1 卷积无缝地合并到网络中, 因此不需要ad-hoc编码. (也就是不需要专门为了使用它而设计网络)
> Ad-Hoc，wiki的定义是: 是拉丁文常用短语中的一个短语. 意思是“特设的、特定目的的、即席的、临时的、将就的、专案的”.  通常用来形容一些特殊的、不能用于其它方面的的，为一个特定的问题、任务而专门设定的解决方案。
> [https://zhuanlan.zhihu.com/p/24268597](https://zhuanlan.zhihu.com/p/24268597)


总体而言, 贡献为**提出了一种新的解码器方案**:

1. 提出了一种简单而有效的数据依赖上采样 (DUpsampling) 方法, 从卷积解码器的粗略输出中恢复像素级分割预测, 替换以前方法中广泛使用的效果较差的双线性.
2. 利用提出的 DUpsampling, 可以避免**过度减少编码器的整体步幅, **显著减少了语义分割方法的计算时间和内存占用 3 倍左右.
3. DUpsampling 还允许解码器将融合的特征在融合它们之前, 降采样到特征图的最低分辨率.下采样不仅减少了解码器的计算量, 大大扩大了特征聚合的设计空间, 使得解码器能够利用更好的特征聚合.

### 重建误差

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552469396542-24601954-580d-4fca-8f58-582f8877b49b.png#align=left&display=inline&height=585&name=image.png&originHeight=585&originWidth=1233&size=144118&status=done&width=1233)
一个重要的观察是, 图片的语义分割标签Y并不是独立同分布的(i.i.d, 也就是互相有依赖), 其中包含着结构信息, 以至于Y可以被压缩而不会造成太大的损失. 因此, 这里不像之前的文章上采样编码器卷积输出F到标签真值F大小, 文章中选择压缩标签真值Y(HxWxC, 已经经过one-hot编码处理, 一个像素只有一个通道对应的类别位置标记为1)到Ywide, 其size和F(Hwide x Wwide x Cwide)一致. 为了更好的进行这个转换, 保证转换中损失更少的信息, 这里设定一个重建误差. 压缩的方法是:

1. 对于原始的F(HxWxC)使用大小为rxr的滑窗处理(这有些类似于pooling操作, 可能还会需要padding操作), 提取数据, 获得一个个的(rxrxC)大小的数据块.
2. 之后各自reshape为大小为1xN(N=rxrxC)的张量
3. 再进行纬度的压缩, 从1xN变为1xCwide, 可以认为是一个1x1的卷积操作. 所有的H/r x W/r 个数据块对应的张量组合得到最终的压缩数据Ywide.
> 感觉可以直接使用kernel_size=stride的卷积操作来实现. 不知道有没有差异.

关于这个重建误差(重建误差也就是要保证变换之后在进行一次反变换后, 与原始信息的差异程度)的计算, 文中给出计算方法:

对于压缩后的数据(1xCwide的那个张量)的反变换(正变换矩阵与反变换矩阵(又称为重建矩阵))后的结果(1xN大小张量)与原始数据(1xN大小的那个张量)使用平方误差累和进行误差计算. 优化目标函数:![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552481968276-bd420c2e-f0b8-4c3b-8c42-0e022051323c.png#align=left&display=inline&height=135&name=image.png&originHeight=135&originWidth=371&size=10758&status=done&width=371)

这个优化问题, 可以利用**迭代的标准SGD算法**优化. 同时, 这里提到, 使用正交约束, 可以简单的**使用PCA来得到这个目标的闭式解**.(确实这里很像PCA降维)

### 损失函数

在网络训练的时候, 要计算损失, 这里考虑了两种计算方法, 后者更为直接:

1. 使用压缩后的Ywide作为真值来监督生成的F, 使用L2损失计算.
    * ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552483197527-c46c01b0-6467-49b7-9ba8-30c54af056ee.png#align=left&display=inline&height=48&name=image.png&originHeight=48&originWidth=224&size=2726&status=done&width=224)
2. 使用未压缩的Y来作为真值来监督恢复分辨率的(softmax(DUpsample(F)))的特征图.
    * ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552483209307-d69a66ee-ccc0-4301-8a31-e5a6e287b377.png#align=left&display=inline&height=50&name=image.png&originHeight=50&originWidth=464&size=6729&status=done&width=464)

### 数据依赖上采样

对于前面的方法2, 使用了*DUpsample*(数据依赖上采样)替换了原始的双线性上采样.

这里提出的方法**相同于在空间维度上应用1x1卷积(可以使用空间上的1x1卷积实现)**, 卷积核存储在W里, 也就是**前面优化目标函数中的反变换矩阵**.
> ~~所以这里实际上就没有用到前面那些, 直接就是使用了提出的这个DUpsample操作, 计算最后的损失即可, 前面的公式都没用...~~没有体会到作者的意思, 这里的**上采样滤波器是前面计算好的**.
> 所以说, 主要用到的公式是这里的第二个损失和前面的目标函数.


除了上面提出的线性上采样, 文章还使用**非线性自编码器**进行了上采样实验. 对自编码器的训练也是为了最大限度地减少重建损失, 并且比线性情况下更通用. 经验上观察到最终的**语义预测精度几乎与使用更简单的线性重建手段是相同的.** 因此文章主要关注与线性重建方法.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552471069685-6a936543-9337-4f1c-afc2-684ac78c48e3.png#align=left&display=inline&height=308&name=image.png&originHeight=308&originWidth=1256&size=57501&status=done&width=1256)

这里提出的上采样操作有点类似于超分辨率重建任务中的亚像素卷积的操作. 最简单的线性形式的 DUpsample 可以被视为一个使用了预计算上采样滤波器的改进 **Depth-to-Space 或 Sub-pixel**.

* Depth-to-Space和Sub-pixel通常为了避免产生太多的可训练参数导致优化困难, 会**使用适度的上采样比例**(例如4)来提升输入分辨率.
* 相反, 文中方法中的**上采样滤波器是预先计算**的. 如果需要, 上采样的比例可以非常大(例如16 或 32).
> ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552471418508-9a0435e4-3eca-4db1-85b0-902b8277b620.png#align=left&display=inline&height=364&name=image.png&originHeight=364&originWidth=1362&size=353561&status=done&width=1362)
> 亚像素卷积层, 图片来自: [https://blog.csdn.net/antkillerfarm/article/details/79956241](https://blog.csdn.net/antkillerfarm/article/details/79956241)

### 自适应温度Softmax

虽然可以通过1x1卷积操作实现DUpsampling, 但直接将其合并到框架中会遇到优化困难, 文章认为, 因为W是利用one-hot编码后的Y来计算的, 原始softmax和提出的DUpsample的计算, 很难产生较为锐利的激活.结果导致交叉熵损失在训练中被卡住, 使得训练过程收敛很慢. 

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552485030477-bdc2df96-1e37-4a9e-be48-c529914ef134.png#align=left&display=inline&height=85&name=image.png&originHeight=85&originWidth=319&size=7186&status=done&width=319)

为了解决这个问题, 这里使用了Hinton著名论文*Distilling the knowledge in a neural network*里提到的"温度"的概念. 对softmax添加了一个"温度"T的参数, 来锐利/软化softmax的激活.这个参数T可以再反向传播中自动学习, 无需调整.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552484837678-dc8af0e9-23ba-4d7a-932c-c2aff7cdd126.png#align=left&display=inline&height=494&name=image.png&originHeight=494&originWidth=607&size=40050&status=done&width=607)

### 灵活的集成策略

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552485432076-3afed647-67bd-458d-ad79-0dae28ff7819.png#align=left&display=inline&height=422&name=image.png&originHeight=422&originWidth=580&size=63855&status=done&width=580)

对比以往的结构使用的集成策略: 

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552485439578-af6920a3-fdaa-4df4-b910-3641cac26192.png#align=left&display=inline&height=47&name=image.png&originHeight=47&originWidth=382&size=5184&status=done&width=382)
> 这里的F是最终的双线性上采样/DUpsample之前的卷积输出(此时通过这些上采样可以得到最终像素级预测)

先对生成的特征图进行上采样, 再结合较低层级的高分辨率特征信息. 这样的设计主要有两个问题:

1. f(CNN计算)在上采样之后计算, 由于f的计算量依赖于输入的空间大小, 这样的安排会导致解码器计算上的抵效, 此外，计算开销阻碍了解码器利用低层级的特征.
2. 融合的低级特征的分辨率与F的分辨率相等, **由于无法使用双线性产生最终像素预测**, 因此通常约为最终预测的1/4分辨率. 为了获得高分辨率预测, 解码器只能选择具有高分辨率低级特征的特征聚合。

相反, 在提出的框架中, 恢复全分辨率预测的**责任在很大程度上已经转移到DUpsampling. **因此, 可以安全地下采样要使用的任何级别的低层特征到最后一个特征图Flast的分辨率(特征图的最低分辨率), 然后融合这些特性以产生最终的预测. 表示为:

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552486374107-fb2064a9-5de5-4008-bd68-05de4a84a782.png#align=left&display=inline&height=42&name=image.png&originHeight=42&originWidth=395&size=5415&status=done&width=395)
> 文章中使用双线性下采样.

这种重排不仅使特征**始终以最低分辨率高效计算**, 而且还**使底层特征 Fi 和最终分割预测的分辨率分离, 允许任何级别的功能进行融合.**

在实验中展示了灵活的特征融合使机制能够利用更好的特征融合, 以尽可能提高分割性能.

**只有在与上述 *DUpsampling* 合作时, 下采样低级特征的方案才能工作**. 否则, 性能是由解码器的incapable上采样方法的上界所限定的. 这就是为什么以前的方法需要上采样*低分辨率的高级特征图*到融合的低层特征映射的空间大小.
> 之前的上采样后解码主要是因为上采样方法会限制住最终的性能, 所以更不能使用下采样低级特征的方案, 会使得性能更加被限制.

## 效果

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552486942467-21a6934b-b307-4756-8b23-eefdd7206538.png#align=left&display=inline&height=270&name=image.png&originHeight=270&originWidth=603&size=43371&status=done&width=603)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552487109857-ee1357a0-170b-48bf-a832-eb10d6c7c942.png#align=left&display=inline&height=380&name=image.png&originHeight=380&originWidth=581&size=55352&status=done&width=581)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552487143313-08190846-c2c5-4cf1-b380-1d4ff85f650b.png#align=left&display=inline&height=311&name=image.png&originHeight=311&originWidth=600&size=106558&status=done&width=600)

> In order to understand how the fusion works, we visualize  the  segmentation  results  with  and  without  low-level features in Fig.  4.  Intuitively, the one fusing low-level features yields more consistent segmentation, which suggests **the downsampled low-level features are still able to refine the segmentation prediction substantially**.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552487235275-c07e402b-cf11-4345-807f-bcbeb1412926.png#align=left&display=inline&height=528&name=image.png&originHeight=528&originWidth=605&size=97157&status=done&width=605)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552487268530-58ddb217-48f6-4038-bfc6-6473a04cf87d.png#align=left&display=inline&height=376&name=image.png&originHeight=376&originWidth=570&size=43718&status=done&width=570)![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1552487279615-6ed9c73e-f632-4554-8eaa-b431c2bf242e.png#align=left&display=inline&height=436&name=image.png&originHeight=436&originWidth=528&size=49723&status=done&width=528)

## 总结

提出了一种灵活、轻便的语义图像分割方案. 这种**新型解码器**采用提出的**数据依赖上采样**产生像素预测.

* 这消除了从底层 CNNs 计算效率低下的高分辨率特征图的需要, 并将融合的低级特征和最终预测的分辨率分离.
* 这种解耦扩展了解码器特征聚集的设计空间, 使得几乎任意的特征聚集被利用来尽可能提高分割性能.
* 同时, 本文提出的解码器避免了将低分辨率高级特征图向上采样回高分辨率低层特征映射的空间大小, 大大降低了解码器的计算量.

实验表明, 与以往语义分割方法中广泛使用的普通解码器相比, 提出的解码器具有有效性和高效性。最后, 与提出的解码器的框架达到了最先进的性能, 同时需要比以前的最先进的方法更少的计算. 
