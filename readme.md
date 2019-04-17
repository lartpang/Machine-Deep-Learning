# ML&DL笔记

[![转载声明](https://img.shields.io/badge/%E5%8D%8F%E8%AE%AE-%E8%BD%AC%E8%BD%BD%E5%A3%B0%E6%98%8E-red.svg?style=for-the-badge&logo=appveyor)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh)
![stars](https://img.shields.io/github/stars/lartpang/Machine-Deep-Learning.svg?style=for-the-badge&logo=appveyor)
![forks](https://img.shields.io/github/forks/lartpang/Machine-Deep-Learning.svg?style=for-the-badge&logo=appveyor)
![issues](https://img.shields.io/github/issues/lartpang/Machine-Deep-Learning.svg?style=for-the-badge&logo=appveyor)

在cs231n笔记的基础上**进一步扩展**, 是我ML&DL学习总结的记录.

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

## changelog

* 2018年11月15日: 最近在看目标检测部分内容, 主要会看RCNN->SPP-Net->Fast RCNN->Faster RCNN, 总结的书写方式要改变下, 为了节省时间, 提高效率, 更多增加自己的思考, 决定不再弄论文的翻译复制过来, 在上面的基础上做笔记这样的方式了, 准备按照架构的流程, 理清架构的思路, 以问题推动思考的方式来进行学习. 希望可以帮助自己更深入的理解.
* 2018年11月19日: 准备开始看Yolo/SSD系列. 这几天看完了RCNN系列的文章, 略感心累. 准备过些日子重新在整理下文档结构吧.
* 2018年11月30日: 看完了RCNN系列, 看完了SSD&Faster的实现, YOLO简单过了一下, 接下来准备看下其他的关于目标检测算法论文.
* 2018年12月04日: 看完了RFCN, 可变形卷积, FPN, 再往后就是mask-rcnn等更为牛逼的网络了, 可是自我感觉, 到目前来说, 或许应该暂停下了, 准备开始复现代码了. 编程能力是基石, 不可缺少, 这个不稳, 一切都不靠谱, 需要上手练练了, 之前只是看了下Faster/SSD的代码, 其他就没有详细读过了, 感觉应该仔细学习下tensorflow, 先从Cifar10的分类网络练手吧!
* 2019年01月06日: 最近在做显著性检测相关的学习, 主要在看一些分割和显著性的论文. 框架最终选择了pytorch.
* 2019年04月17日: 好久没有更新了, 主要是当前更新文档主要存放在[语雀](https://www.yuque.com/lart/papers), 同样支持markdown, 图片公式什么的也还方便. 目前主要看的论文是显著性, 但是会关注下图神经网络和分割相关的论文.

## 后期想法

1. 基础文章收集到一起
2. 论文部分进行一下分类归档
3. 删除文章未使用的图片

## 其他

论文开始的时候多是直接找翻译, 现在大多数是直接看原文, 有时候原文的细节会丰富些.

* https://github.com/tensorflow/models/tree/master/research/slim
* "通天塔(有很多论文翻译, 虽然翻译的一般, 但是可以看)": http://tongtianta.site/
* https://arxiv.org/

## 感谢

感谢开源社区.

笔记随时在补充, 更新. 不过可能后期更新多在[语雀](https://www.yuque.com/lart/papers)上, 有时间会尽可能搬运一下.
