# A Bi-directional Message Passing Model for Salient Object Detection

- [A Bi-directional Message Passing Model for Salient Object Detection](#a-bi-directional-message-passing-model-for-salient-object-detection)
  - [主要工作](#主要工作)
  - [网络结构](#网络结构)
  - [实验细节](#实验细节)
  - [相关链接](#相关链接)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563008258700-720984cd-1518-4e3b-adef-5564f43c70ed.png#align=left&display=inline&height=302&name=image.png&originHeight=302&originWidth=1070&size=52261&status=done&width=1070)

这是CVPR2018的一篇文章, 这里做一个简短的记录. 文章主要的目的就是通过设计网络来更好地集成不同层级得到的特征信息.

## 主要工作

通过一些工作的分析, 文章提出了存在的一些不足:

1. **现有的工作受限于有限的感受野**, 学习到的特征可能不能包含丰富的上下文信息来准确的检测尺度、 形状和位置多变的目标.
1. 现有工作主要使用来自深层卷积层的高级特征, 低级的空间信息的缺少可能导致显著性图难以保持细致的边缘. 一些新工作的开始**尝试利用多层级卷积特征来进行显著性检测**.
  1. 现有的使用短连接沟通深层与浅层特征的方式, 大多是仅仅将深层的侧输出连接到浅层, 并且**忽略了反向的信息传输**. 因此深层侧输出仍然缺乏低级信息, 这些存在于浅层侧输出中.
  1. 现有的工作也有直接拼接所有的来自深层和浅层的特征来集成多级特征, 然而, **直接的拼接所有层级, 没有考虑互相的重要性权重, 这并不是一个有效的融合的方式**. 因为多层级特征并不总是对于每一个输入图像有效.

针对这些问题, 本文提出了几个针对性的解决方案:

1. 设计了一个**利用不同扩张率的卷积层并行处理编码器特征的模块**(多尺度上下文特征提取模块: MCFEM), 之后进行拼接, 以获得融合了多尺度的上下文信息的特征.
1. **引入门控双向信息传递模块(GBMPM), 提供一个自适应并且有效的策略来集成多层级特征**. 集成的特征互相补充, 并且对于处理不同的场景下的情况具有一定的鲁棒性.
  1. **使用双向结构来在不同层级之间的特征传递信息**, 高层语义信息传递到浅层, 低级空间细节包含在浅层特征并传递到相反的方向. 这样语义信息和细节信息被插入到每一个层级.
  1. **使用门控结构, 来控制信息的传递**, 从而传递有用的特征, 丢弃多余的特征.

总结起来就是:

* 多尺度特征融合
* 双向信息传递机制
* 门控机制, 控制信息传递

## 网络结构

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563069387589-117ee158-2156-4a09-a3c1-5cf0587ed1ad.png#align=left&display=inline&height=725&name=image.png&originHeight=725&originWidth=1238&size=206304&status=done&width=1238)

图中主要包含了这样一些过程:

1. 使用VGG16进行特征提取, 移除原始VGG的全连接层和最后一个池化层.
1. Multi-scale Context-aware Feature Extraction
  1. 使用MCFEM通过拼接并行的扩张卷积处理结果获得融合了不同尺度信息的特征.
3. Gated Bi-directional Message Passing
  1. 使用2获得的特征进一步送入GBMPM的三部分结构:h, h, G.
    1. ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563069962007-ab3bea90-0bdf-4c19-aa06-42bce180aee6.png#align=left&display=inline&height=24&name=image.png&originHeight=33&originWidth=89&size=1740&status=done&width=65), ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563069999444-03e84158-366a-4b03-9310-1ca6acca3722.png#align=left&display=inline&height=24&name=image.png&originHeight=37&originWidth=80&size=1492&status=done&width=52), ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563070021616-1be57017-864b-4c2f-bab0-d89d4a8c4730.png#align=left&display=inline&height=24&name=image.png&originHeight=33&originWidth=70&size=1534&status=done&width=51)
    1. ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563069919305-4c8cb939-996a-472e-8b4e-5ad3b804fd14.png#align=left&display=inline&height=98&name=image.png&originHeight=169&originWidth=552&size=28525&status=done&width=320)
    1. ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563069831712-a586a33e-8214-47c0-9b62-7d5a9fcaeeee.png#align=left&display=inline&height=24&name=image.png&originHeight=37&originWidth=433&size=6776&status=done&width=279)
  2. 使用获得的h, h来获得h.
    1. ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563069864552-8ee9c0de-3fcc-4776-9870-346c80b4d459.png#align=left&display=inline&height=26&name=image.png&originHeight=42&originWidth=432&size=6791&status=done&width=266)
4. Saliency Inference
  1. 使用1x1卷积处理生成阶段性的预测, 并上采样进行融合获得最后的输出
    1. ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563070127100-ad7ced5c-787e-479b-9766-b3108f7e3d46.png#align=left&display=inline&height=43&name=image.png&originHeight=72&originWidth=490&size=14313&status=done&width=294)
  2. 最终使用交叉熵计算损失
    1. ![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563070161635-144b1692-bb4a-4277-b9e4-89c911ef693b.png#align=left&display=inline&height=42&name=image.png&originHeight=73&originWidth=534&size=7734&status=done&width=307)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563069593537-32638847-65df-4127-8486-6aa5fd02b894.png#align=left&display=inline&height=357&name=image.png&originHeight=357&originWidth=573&size=71214&status=done&width=573)

## 实验细节

* We utilize the training set of DUTS dataset to train our proposed model. It contains 10553 images with high-quality pixel-wise annotations.
* We augment the training set by horizontal flipping and cropping the images to relieve the over-fitting problem, as suggested in [18]. We don't use the validation set and train the model until its training loss converges.
* A NVIDIA Titan X GPU is used for training and testing.
* The parameters of the first 13 convolutional layers are initialized by VGG-16 net. For other convolutional layers, we **initialize the weights using truncated normal method**.
* The convolutional parameters of our message passing module in Sec. 3.3 are not shared, and the **upsampling and downsampling are conducted simply by bilinear interpolation**.
* Our model is trained using Adam with an initial learning rate at 1e-6. The training process of our model takes about 30 hours and converges after 12 epochs.
* During testing, our proposed model runs **about 22 fps with 256 × 256 resolution**.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563070363403-9d1b501e-aad7-4e0e-a7d7-656d5d4fba52.png#align=left&display=inline&height=509&name=image.png&originHeight=509&originWidth=1152&size=158893&status=done&width=1152)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1563070436102-c30c505b-e213-4d4f-8f87-8d60cd9a8aa8.png#align=left&display=inline&height=283&name=image.png&originHeight=283&originWidth=556&size=47318&status=done&width=556)

## 相关链接

* 论文:[http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_A_Bi-Directional_Message_CVPR_2018_paper.pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_A_Bi-Directional_Message_CVPR_2018_paper.pdf)
