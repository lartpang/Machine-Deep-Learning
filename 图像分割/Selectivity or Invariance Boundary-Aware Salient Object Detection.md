# Selectivity or Invariance: Boundary-Aware Salient Object Detection

- [Selectivity or Invariance: Boundary-Aware Salient Object Detection](#selectivity-or-invariance-boundary-aware-salient-object-detection)
  - [主要贡献](#主要贡献)
  - [针对问题](#针对问题)
  - [主要方法](#主要方法)
  - [实验细节](#实验细节)
  - [参考链接](#参考链接)

> 原始文档: https://www.yuque.com/lart/papers/banet

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576586336961-58817976-e336-465d-898b-94fb556629d8.png#align=left&display=inline&height=293&name=image.png&originHeight=293&originWidth=1212&size=163887&status=done&style=none&width=1212)

## 主要贡献

1. 对于显著性目标检测任务进一步明确其主要的两个难点,是一个对于变与不变都有需求的问题.
1. 针对变与不变,提出了一种分而治之的模型,三支路各自实现不同的任务,三者相互补充.
1. 提出了一种新颖的ASPP的替代结构.不同扩张率的分支的特征逐个传递,实现了丰富的多尺度上下文信息的提取.

## 针对问题

这篇文章还是从显著性目标边界的角度入手，着重解决两个问题：

1. First, **the interiors of a large salient object may have large appearance change**, making it difficult to detect the salient object as a whole.
1. Second, **the boundaries of salient objects may be very weak** so that they cannot be distinguished from the surrounding background regions.

这实际上也就是所谓的 `selectivity–invariance dilemma(困境)` （我们**需要一组特征，它们能够选择性地响应图片中的重要部分，而对图片中不重要部分的变化保持不变性**）。显着对象的不同区域（内部与边界）对SOD模型提出了**不同的要求**，而这种困境实际上阻止了具有各种大小，外观和上下文的显着对象的完美分割。

> In the interiors, the features extracted by a SOD model should **be invariant to various appearance changes** such as size, color and texture. Such invariant features ensure that the salient object can pop-out as a whole. However, the features at boundaries should **be sufficiently selective at the same time** so that the minor difference between salient objects and background regions can be well distinguished.

## 主要方法

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576587314470-44256e25-b9ec-4210-9039-23d12460eff2.png#align=left&display=inline&height=694&name=image.png&originHeight=694&originWidth=1318&size=404684&status=done&style=none&width=1318)

> 图中的特征提取网络使用的是ResNet50，但是对于第四个和第五个卷积块的步长设置为1，使其不改变分辨率，同时为了扩大感受野，在这两个结构里又使用了扩张率分别为2和4的扩张卷积。最终该网络仅会下采样输入的1/8。

这篇文章提出的应对方法就是adopt different feature extraction strategies at object interiors and boundaries，在分析文章https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/99687783中指出，这里与BASNet的策略有些相似，只是这里使用了分支网络来实现边界的增强，而BASNet使用了损失函数来处理。

这里在特征提取网络的基础上构建了三个分支，来进行可选择性（selective）和不变性（invariance）特征的提取，同时修正边界与内部过渡区域的误判情况。分别称为：

1. **boundary localization stream**：the boundary localization stream is a simple subnetwork that aims to extract selective features for detecting the boundaries of salient objects
1. **interior perception stream**：the interior perception stream emphasizes the feature invariance in detecting the salient objects
1. **transition compensation stream**：a transition compensation stream is adopted to amend the probable failures that may occur in the transitional regions between interiors and boundaries, where the feature requirement gradually changes from invariance to selectivity

同时也提出了integrated successive dilation module来增强interior perception和transition compensation两个信息流，可以获得丰富的上下文信息产生，以使其可以对于各样的视觉模式都能提取不变的特征，并引入来自低级特征的跳跃连接以促进边界的选择性表示。该模块结构如下：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576588871268-2aa3056f-2d1b-432e-9e28-18a19f4020fa.png#align=left&display=inline&height=369&name=image.png&originHeight=507&originWidth=632&size=88471&status=done&style=none&width=460)

> The ISD module with N parallel branches with skip connections is denoted as ISD-N, and we show the structureof ISD-5 in Fig.5 as an example.
> 从实现的角度来看，五个支路是需要有先后顺序的，得从左往右开构建。

* The first layer of each branch is a convolutional layer with 1×1 kernels that is used for channel compression.
* The second layer of each branch adopts dilated convolution, in which **the dilation rates start from 1 in the first branch and double in the subsequent branch**.
  + 通过不同分支之间的跳跃连接，the feature map from the first branch of the second layer is also encoded in the feature maps of subsequent branches, which actually gets processed by successive dilation rates. 
* After that, the third and the forth layers adopt 1×1 kernels to integrate feature maps formed under various dilation rates.

In practice, we **use ISD-5 in the interior perception stream** and **ISD-3 in the transition compensation streams.**

这里损失函数包含几部分：

1. 边界交叉熵损失E，这里的GB表示the boundary map of salient objects：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576590241090-07a6c5d9-f159-4451-a5f5-3396d320c8e8.png#align=left&display=inline&height=35&name=image.png&originHeight=35&originWidth=302&size=9115&status=done&style=none&width=302)

2. 内部交叉熵损失E：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576590228626-64db155c-5ff7-49f6-a4b0-9e0c65cb815d.png#align=left&display=inline&height=35&name=image.png&originHeight=35&originWidth=284&size=7983&status=done&style=none&width=284)

3. 最后的交叉熵损失E：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576590210708-cdcf091d-df29-4875-91a9-3fb17543dd47.png#align=left&display=inline&height=36&name=image.png&originHeight=36&originWidth=230&size=6439&status=done&style=none&width=230)

这里第三个损失中，Sig(M)表示最终的预测结果，这里的M是对三路信息流特征的整合，这里的整合方式没有使用常规的元素级加法或者拼接, 因为相对而言效果不好. 作者自行设计了一种方法：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576590397429-8f52a708-937a-48ce-83bf-307a5dbb5c5c.png#align=left&display=inline&height=70&name=image.png&originHeight=70&originWidth=580&size=21005&status=done&style=none&width=580)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576590737937-48a2c21f-6b50-4a17-8528-a35c969e3fa1.png#align=left&display=inline&height=29&name=image.png&originHeight=29&originWidth=175&size=5106&status=done&style=none&width=175)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576590744185-cac57e65-d15c-44d9-8309-31889e5dbfd4.png#align=left&display=inline&height=30&name=image.png&originHeight=30&originWidth=165&size=4839&status=done&style=none&width=165)

这里的三个$\phi$表示B(边界定位)/I(内部感知)/T(过渡补偿)三个分支输出的单通道特征图.
最终训练用的损失是一个三者的组合:

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576590777703-77972cd2-1b0c-4f2b-8f27-5a6cb48808aa.png#align=left&display=inline&height=51&name=image.png&originHeight=51&originWidth=333&size=7721&status=done&style=none&width=333)

## 实验细节

* The training images are not done with any special treatment except the **horizontal flipping**.
* The training process takes about 15 hours and converges after **200k iterations** with **mini-batch of size 1**.
* During testing, the proposed network removes all the losses, and each image is **directly fed into the network to obtain its saliency map without any pre-processing**.
* The proposed method runs at about **13 fps with about 400 × 300 resolution** on our computer with a 3.60GHz CPU and a GTX 1080ti GPU.

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576591551941-ee72c1db-f8f0-42a7-b3d3-bf1cb53abbc8.png#align=left&display=inline&height=447&name=image.png&originHeight=447&originWidth=720&size=200644&status=done&style=none&width=720)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/192314/1576591538404-c6bdf3c4-052c-496d-a7e8-98186d07d764.png#align=left&display=inline&height=242&name=image.png&originHeight=242&originWidth=718&size=99008&status=done&style=none&width=718)

## 参考链接

* 论文: [http://openaccess.thecvf.com/content_ICCV_2019/papers/Su_Selectivity_or_Invariance_Boundary-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf](http://openaccess.thecvf.com/content_ICCV_2019/papers/Su_Selectivity_or_Invariance_Boundary-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf)
* 解析: [https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/99687783](https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/99687783)

