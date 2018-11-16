# Faster R-CNN(2016)

> 论文翻译: https://alvinzhu.xyz/2017/10/12/faster-r-cnn/
>
> 论文arxiv: 

对于Fast R-CNN，其仍然需要selective search方法来生产候选区域，这是非常费时的。为了解决这个问题，Faster R-CNN模型（The Faster Region-based Convolutional Network, [S. Ren and al. 2016](https://arxiv.org/pdf/1506.01497.pdf)）引入了RPN (Region Proposal Network)直接产生候选区域。Faster R-CNN可以看成是RPN和Fast R-CNN模型的组合体，即**Faster R-CNN = RPN + Fast R-CNN**。

对于RPN网络，先采用一个CNN模型（一般称为特征提取器）接收整张图片并提取特征图。然后在这个特征图上采用一个N* N（文中是3* 3）的滑动窗口，对于每个滑窗位置都映射一个低维度的特征（如256-d）。然后这个特征分别送入两个全连接层，一个用于分类预测，另外一个用于回归。对于每个窗口位置一般设置k个不同大小或比例的先验框（anchors, default bounding boxes），这意味着每个位置预测$k$个候选区域（region proposals）。对于分类层，其输出大小是2k，表示各个候选区域包含物体或者是背景的概率值，而回归层输出4k个坐标值，表示各个候选区域的位置（相对各个先验框）。对于每个滑窗位置，这两个全连接层是共享的。因此，RPN可以采用卷积层来实现：首先是一个n*n卷积得到低维特征，然后是两个1* 1的卷积，分别用于分类与回归。

![img](https://ask.qcloudimg.com/http-save/yehe-1342338/dypqdgwfur.jpeg?imageView2/2/w/1620)

可以看到RPN采用的是二分类，仅区分背景与物体，但是不预测物体的类别，即class-agnostic。由于要同时预测坐标值，在训练时，要先将先验框与ground-truth box进行匹配，原则为：（1）与某个ground-truth box的IoU最高的先验框；（2）与某个ground-truth box的IoU值大于0.7的先验框，只要满足一个，先验框就可以匹配一个ground-truth，这样该先验框就是正样本（属于物体），并以这个ground-truth为回归目标。对于那些与任何一个ground-truth box的IoU值都低于0.3的先验框，其认为是负样本。RPN网络是可以单独训练的，并且单独训练出来的RPN模型给出很多region proposals。由于先验框数量庞大，RPN预测的候选区域很多是重叠的，要先进行NMS(non-maximum suppression，IoU阈值设为0.7）操作来减少候选区域的数量，然后按照置信度降序排列，选择top-N个region proposals来用于训练Fast R-CNN模型。RPN的作用就是代替了Selective search的作用，但是速度更快，因此Faster R-CNN无论是训练还是预测都可以加速。

Faster R-CNN模型采用一种4步迭代的训练策略：（1）首先在ImageNet上预训练RPN，并在PASCAL VOC数据集上finetuning；（2）使用训练的PRN产生的region proposals单独训练一个Fast R-CNN模型，这个模型也先在ImageNet上预训练；（3）用Fast R-CNN的CNN模型部分（特征提取器）初始化RPN，然后对RPN中剩余层进行finetuning，此时Fast R-CNN与RPN的特征提取器是共享的；（4）固定特征提取器，对Fast R-CNN剩余层进行finetuning。这样经过多次迭代，Fast R-CNN可以与RPN有机融合在一起，形成一个统一的网络。其实还有另外一中近似联合训练策略，将RPN的2个loss和Fast R-CNN的2个loss结合在一起，然后共同训练。注意这个过程，Fast R-CNN的loss不对RPN产生的region proposals反向传播，所以这是一种近似（如果考虑这个反向传播，那就是非近似联合训练）。应该来说，联合训练速度更快，并且可以训练出同样的性能。

最好的Faster R-CNN模型在 2007 PASCAL VOC测试集上的mAP为78.8% ，而在2012 PASCAL VOC测试集上的mAP为75.9%。论文中还在 COCO数据集上进行了测试。Faster R-CNN中的某个模型可以比采用selective search方法的Fast R-CNN模型快34倍。可以看到，采用了RPN之后，无论是准确度还是速度，Faster R-CNN模型均有很大的提升。Faster R-CNN采用RPN代替启发式region proposal的方法，这是一个重大变革，后面的two-stage方法的研究基本上都采用这种基本框架，而且和后面算法相比，Faster R-CNN在准确度仍然占据上风。

![img](https://ask.qcloudimg.com/http-save/yehe-1342338/p07jj4rgze.jpeg?imageView2/2/w/1620)

