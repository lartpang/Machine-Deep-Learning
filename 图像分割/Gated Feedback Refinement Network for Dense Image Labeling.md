# Gated Feedback Refinement Network for Dense Image Labeling

- [Gated Feedback Refinement Network for Dense Image Labeling](#Gated-Feedback-Refinement-Network-for-Dense-Image-Labeling)
  - [主要工作](#%E4%B8%BB%E8%A6%81%E5%B7%A5%E4%BD%9C)
  - [网络结构](#%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84)
    - [Gate Unit](#Gate-Unit)
    - [Gated Refinement Unit](#Gated-Refinement-Unit)
    - [Stage-wise Supervision](#Stage-wise-Supervision)
  - [实验结果](#%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C)
  - [相关链接](#%E7%9B%B8%E5%85%B3%E9%93%BE%E6%8E%A5)
  - [相关文章——LRN](#%E7%9B%B8%E5%85%B3%E6%96%87%E7%AB%A0LRN)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562667872750-674fd490-3620-4680-9617-bef635169c10.png#align=left&display=inline&height=238&name=image.png&originHeight=238&originWidth=974&size=45867&status=done&width=974)

## 主要工作

本文是17年的CVPR的论文，主要解决了之前基于深度学习的语义分割模型中存在的一些问题，主要针对的问题是**如何更有效的整合和利用低级特征与高级、局部和全局特征。**

有效整合局部和全局信息对于密集标签问题至关重要。大多数基于编解码器架构的现有方法**简单地连接来自较早层的特征，以在细化阶段中获得更高频率的细节**。但是前向传播中的模糊信息，则可能导致最终的细化结果的质量存在限制。

- 本文提出了一种门控反馈细化网络（G-FRNet），这是一种用于密集标记任务的端到端深度学习框架，可解决现有方法的这种局限性。
- 首先G-FRNet进行粗略预测，然后在细化阶段有效地整合局部和全局上下文信息来逐步细化细节。
- 本文引入了控制向前传递信息的门单元，以**借助深层特征来辅助浅层特征滤除其中的信息的模糊与歧义**。使用更深，更有辨别力的层的特征来过滤从辨别能力较差但定位更精细的早期层传递的信息。
- 文章认为深层特征可以帮助浅层信息恢复特征中的模糊性内容，而单独的浅层信息并不能很好的恢复，因为**其感受野不是很大或者并不具有足够的区分性**。

在几个具有挑战性的数据集上的实验结果表明，所提出的模型比现有最先进的方法具有更好性能。 基于消融分析的实验结果揭示了粗到细门控细化值的普遍性。广泛的CNN模型可以从这些简单的体系结构修改中受益。

## 网络结构

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562668205980-c20fb56c-a48c-48df-aba3-f6af97317aed.png#align=left&display=inline&height=632&name=image.png&originHeight=632&originWidth=1233&size=181552&status=done&width=1233)

### Gate Unit

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562668932020-032c1351-b4bc-4e09-9320-6d23730e926e.png#align=left&display=inline&height=325&name=image.png&originHeight=325&originWidth=264&size=14137&status=done&width=264)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670630870-d4407825-3c75-46ca-9944-47d1b7184b66.png#align=left&display=inline&height=155&name=image.png&originHeight=155&originWidth=596&size=21635&status=done&width=596)

这里门控的设计使用深层的特征来控制浅层的特征，得到处理后的特征送入解码器。

### Gated Refinement Unit 

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562668220971-2d8515e7-f122-4037-8485-5490ba5c4090.png#align=left&display=inline&height=253&name=image.png&originHeight=253&originWidth=583&size=34934&status=done&width=583)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670588157-8a581f46-5762-4425-a91b-bdd7770b2608.png#align=left&display=inline&height=36&name=image.png&originHeight=36&originWidth=578&size=7285&status=done&width=578)

就是普通的卷积处理，不过这里有个设定，就是图中 `+` 操作表示的是拼接， `U` 表示上采样。这里拼接的时候，两部分的特征![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670481975-830db5e3-f881-4090-9866-5669e132e2ec.png#align=left&display=inline&height=32&name=image.png&originHeight=32&originWidth=161&size=2452&status=done&width=161)，这里C是总的类别数量。这里给出了两个理由：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670518125-d7b65bd4-74c0-4991-92e6-6f9cc089d993.png#align=left&display=inline&height=233&name=image.png&originHeight=233&originWidth=588&size=64479&status=done&width=588)

- 降低计算量
- 避免通道数少的特征被通道数多的特征所淹没

### Stage-wise Supervision

这里使用了深监督策略，损失如下：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670756492-c8d8077c-57b9-4fbf-89fd-5ae9d0ca4cde.png#align=left&display=inline&height=105&name=image.png&originHeight=105&originWidth=581&size=16183&status=done&width=581)

其中的η表示原始真值，这里的R表示放缩到对应的特征图大小后的真值。这里使用交叉熵，最终损失直接加和，权重都为1。

## 实验结果

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670847825-9096dd0f-8ea3-4e27-b95f-6777e3e2f2cc.png#align=left&display=inline&height=422&name=image.png&originHeight=422&originWidth=1242&size=116051&status=done&width=1242)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670862667-9b4083d9-934d-4f4c-a311-2ed8bb86e108.png#align=left&display=inline&height=402&name=image.png&originHeight=402&originWidth=584&size=58215&status=done&width=584)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670881118-ded266fa-5b6d-4118-90f3-b94ca89a2070.png#align=left&display=inline&height=412&name=image.png&originHeight=412&originWidth=1204&size=125029&status=done&width=1204)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670900090-e606e7dd-b608-419f-900f-859b78715437.png#align=left&display=inline&height=338&name=image.png&originHeight=338&originWidth=1226&size=85209&status=done&width=1226)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670914198-fb309d36-b098-401c-811a-99ba906758b2.png#align=left&display=inline&height=291&name=image.png&originHeight=291&originWidth=1213&size=87877&status=done&width=1213)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562670947081-77108d37-9b30-4ae0-aebc-b8c7924efa2d.png#align=left&display=inline&height=426&name=image.png&originHeight=426&originWidth=595&size=40497&status=done&width=595)

## 相关链接

- 论文：[https://www.cs.umanitoba.ca/~ywang/papers/cvpr17.pdf](https://www.cs.umanitoba.ca/~ywang/papers/cvpr17.pdf)

## 相关文章——LRN

这应该是本文的初期工作，本文在此基础上添加了门控机制，这篇LRN的文章的结果在本文GFRNet中也有展现：[https://arxiv.org/pdf/1703.00551.pdf](https://arxiv.org/pdf/1703.00551.pdf)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562667924415-691d3a3e-9d82-4f18-b4e1-186a2525ba51.png#align=left&display=inline&height=318&name=image.png&originHeight=318&originWidth=1355&size=64344&status=done&width=1355)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562667995050-a1761b74-a7e1-48f7-a490-6f36f44ded97.png#align=left&display=inline&height=638&name=image.png&originHeight=638&originWidth=1487&size=196309&status=done&width=1487)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1562668073787-b8ea99fc-a60b-4f0e-9a63-1694d0bf6920.png#align=left&display=inline&height=716&name=image.png&originHeight=716&originWidth=415&size=114897&status=done&width=415)
