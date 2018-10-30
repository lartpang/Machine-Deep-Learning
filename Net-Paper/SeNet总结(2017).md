# SeNet

## 前言

> https://chtseng.wordpress.com/2017/11/20/ilsvrc-%E6%AD%B7%E5%B1%86%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E6%A8%A1%E5%9E%8B/

2017年的ILSVRC競賽是由新加坡國立大學與奇虎360 合作的SeNet以2.3% top-5 error rate取得冠軍，錯誤率較前兩年的ResNet減少了36%。

SeNet全稱是Squeeze-and-Excitation Networks，它會依據loss function的學習來調整不同屬性的特徵權重，讓有效的feature map權重加大，無效或效果小的feature map權重變小，使得模型訓練達到更好的結果。這些動作SeNet稱為「feature re-calibration」，包含了Squeeze → Excitation→ Scaling 這三個程序。

SeNet開發者巧妙的將這三個步驟包成一個程序稱為Squeeze-and-Excitation
Block中，讓SeNet能夠當作一種附加功能整合到其它的網路當中，例如這次參與競賽的SeNet便是由ResNet的改進版ResNeXt加上Squeeze-and-Excitation
Block的model。如果您看到SE開頭的深度網路名稱，例如SE-ResNet或SE-Inception，便知道那是ResNet或GoogLeNet與SeNet整合的models。

## 概要

卷积神经网络（CNNs）已被证明是解决各种视觉任务的有效模型[19,23,29,41]。对于每个卷积层，沿着输入通道学习一组滤波器来表达局部空间连接模式。换句话说，**期望卷积滤波器通过融合空间信息和通道信息进行信息组合**，然而受限于局部感受野。

通过叠加一系列非线性和下采样交织的卷积层，CNN能够捕获**具有全局感受野的分层模式**作为强大的图像描述。

最近的工作已经证明，网络的性能可以通过**显式地嵌入学习机制**来改善，这种学习机制有助于捕捉空间相关性而不需要额外的监督。Inception架构推广了一种这样的方法[14,39]，这表明网络可以通过在其模块中**嵌入多尺度处理来取得有竞争力的准确度**。最近的工作在寻找更好地模拟空间依赖[1,27]并结合空间注意力[17]。

与这些方法相反，通过引入新的架构单元，我们称之为*“Squeeze-and-Excitation”* (SE)块，我们研究了架构设计的一个不同方向——通道关系。我们的**目标是通过显式地建模卷积特征通道之间的相互依赖性来提高网络的表示能力**。为了达到这个目的，我们提出了一种机制，使网络能够执行*特征重新校准*，通过这种机制可以学习使用全局信息来选择性地强调信息特征并抑制不太有用的特征。