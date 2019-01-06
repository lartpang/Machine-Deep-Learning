# ML&DL笔记

在cs231n笔记的基础上**进一步扩展**, 是我ML&DL学习总结的记录.

## changelog

* 2018年11月15日:

    最近在看目标检测部分内容, 主要会看RCNN->SPP-Net->Fast RCNN->Faster RCNN, 总结的书写方式要改变下, 为了节省时间, 提高效率, 更多增加自己的思考, 决定不再弄论文的翻译复制过来, 在上面的基础上做笔记这样的方式了, 准备按照架构的流程, 理清架构的思路, 以问题推动思考的方式来进行学习.

    希望可以帮助自己更深入的理解.

* 2018年11月19日:

    准备开始看Yolo/SSD系列.

    这几天看完了RCNN系列的文章, 略感心累.

    准备过些日子重新在整理下文档结构吧.

* 2018年11月30日:

    看完了RCNN系列, 看完了SSD&Faster的实现, YOLO简单过了一下, 接下来准备看下其他的关于目标检测算法论文.

* 2018年12月04日:

    看完了RFCN, 可变形卷积, FPN, 再往后就是mask-rcnn等更为牛逼的网络了, 可是自我感觉, 到目前来说, 或许应该暂停下了, 准备开始复现代码了.

    编程能力是基石, 不可缺少, 这个不稳, 一切都不靠谱, 需要上手练练了, 之前只是看了下Faster/SSD的代码, 其他就没有详细读过了, 感觉应该仔细学习下tensorflow, 先从Cifar10的分类网络练手吧!

* 2019年01月06日

    最近在做显著性检测相关的学习, 主要在看一些分割和显著性的论文.

<details>
<summary>已读论文</summary>

1. 2012
    1. AlexNet
2. 2013
    1. NiN
3. 2014
    1. OverFeat
    2. GoogLeNet
    3. VGG
    4. R-CNN
    5. SPP-Net
4. 2015
    1. 深度学习综述(三巨头, 简单看了下)
    2. BN-GoogLeNet
    3. InceptionV2/V3
    4. ResNet
    5. FCN
    6. Deconvolution Network(Semantic Segmentation)
    7. Fast R-CNN
    8. YOLO-V1
    9. SSD
    10. HED(简单看了下)
    11. STN(简单看了下, 为了进一步了解可变形卷积)
    12. DAG-RNN
    13. SegNet
    14. U-Net
5. 2016
    1. Faster R-CNN
    2. YOLO-V2(简单看了下改进)
    3. FPN
    4. R-FCN
    5. Automatic Portrait Segmentation for Image Stylization
    6. Deep Automatic Portrait Matting(0)
    7. DeepLabv2
    8. DHSNet
    9. RFCN(Saliency)
    10. PSPNet
    11. RefineNet
6. 2017
    1. SeNet
    2. DenseNet
    3. SqueezeNet
    4. Deformable ConvNet
    5. Deep Image Matting
7. 2018
    1. YOLO-V3(简单看了下改进)
    2. Deep Propagation Based Image Matting
    3. Dense DAG-RNN
    4. Semantic Human Matting
    5. UNet++(简单看了下思路)

</details>

## 后期想法

调整文件结构

1. 基础文章收集到一起
2. 论文部分进行一下分类归档
3. 删除文章未使用的图片
4. 复现分类网络

---

<details>
<summary> 关于CS231n笔记部分内容 </summary>

CS231n课程笔记的翻译, 始于@杜客在一次回答问题“应该选择TensorFlow还是Theano？”中的机缘巧合, 在取得了授权后申请了**知乎专栏智能单元 - 知乎专栏**独自翻译. 随着翻译的进行, 更多的知友参与进来. 他们是@ShiqingFan, @猴子, @堃堃和@李艺颖.

大家因为认同这件事而聚集在一起, 牺牲了很多个人的时间来进行翻译, 校对和润色. 而翻译的质量, 我们不愿意自我表扬, 还是请各位知友自行阅读评价吧. 现在笔记翻译告一段落, 下面是团队成员的简短感言：

@ShiqingFan：一个偶然的机会让自己加入到这个翻译小队伍里来. CS231n给予了我知识的源泉和思考的灵感, 前期的翻译工作也督促自己快速了学习了这门课程. 虽然科研方向是大数据与并行计算, 不过因为同时对深度学习比较感兴趣, 于是乎现在的工作与两者都紧密相连. Merci!

@猴子：在CS231n翻译小组工作的两个多月的时间非常难忘. 我向杜客申请加入翻译小组的时候, 才刚接触这门课不久, 翻译和校对的工作让我对这门课的内容有了更深刻的理解. 作为一个机器学习的初学者, 我非常荣幸能和翻译小组一起工作并做一点贡献. 希望以后能继续和翻译小组一起工作和学习.

@堃堃：感谢组内各位成员的辛勤付出, 很幸运能够参与这份十分有意义的工作, 希望自己的微小工作能够帮助到大家, 谢谢！

@李艺颖：当你真正沉下心来要做一件事情的时候才是学习和提高最好的状态；当你有热情做事时, 并不会觉得是在牺牲时间, 因为那是有意义并能带给你成就感和充实感的；不需要太过刻意地在乎大牛的巨大光芒, 你只需像傻瓜一样坚持下去就好了, 也许回头一看, 你已前进了很多. 就像老杜说的, 我们就是每一步慢慢走, 怎么就“零星”地把这件事给搞完了呢？

@杜客：做了一点微小的工作, 哈哈.

</details>

> 感谢对于CS231n的笔记的翻译.

~~## 关于Linux相关知识~~ 这部分内容已经转移到了另一个仓库 [LinuxNote](https://github.com/lartpang/LinuxNote)

## 关于Python

主要穿插在笔记代码中.

## 关于`Net-Paper`中的paper

大多数是直接参考slim的代码和http://noahsnail.com博客内容, 博客中没有的直接谷歌.

> https://github.com/tensorflow/models/tree/master/research/slim
>
> "通天塔(有很多论文翻译, 虽然翻译的一般, 但是可以看)": http://tongtianta.site/
>
> https://arxiv.org/

---

感谢开源社区, 感谢网络世界, 感谢帮助到我的所有人.

笔记随时在补充, 更新.
