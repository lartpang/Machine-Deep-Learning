##   概要

数据预处理在众多深度学习算法中都起着重要作用，实际情况中，将数据做归一化和白化处理后，很多算法能够发挥最佳效果。然而除非对这些算法有丰富的使用经验，否则预处理的精确参数并非显而易见。在本页中，我们希望能够揭开预处理方法的神秘面纱，同时为预处理数据提供技巧（和标准流程）。 

提示：当我们开始处理数据时，首先要做的事是观察数据并获知其特性。本部分将介绍一些通用的技术，在实际中应该针对具体数据选择合适的预处理技术。例如一种标准的预处理方法是对每一个数据点（样本）都减去它的均值（也被称为 **移除直流分量/局部均值消减/消减归一化**），<u>这一方法对诸如自然图像这类数据是有效的，但对非平稳的数据则不然</u>。 

##   数据归一化

数据预处理中，**标准的第一步是数据归一化**。虽然这里有一系列可行的方法，但是这一步通常是根据数据的具体情况而明确选择的。特征归一化常用的方法包含如下几种： 

*  简单缩放 
*  逐样本均值消减(也称为移除直流分量) 
*  特征标准化(使数据集中所有特征都具有**零均值和单位方差**) 

### 简单缩放——元素维度

在简单缩放中，我们的目的是通过对数据的每一个维度的值进行重新调节（这些维度可能是相互独立的），使得最终的数据向量**落在 [0,1]或[ − 1,1] 的区间内**（根据数据情况而定）。这对后续的处理十分重要，因为很多*默认*参数（如 PCA-白化中的 epsilon）都假定数据已被缩放到合理区间。 

**例子:**在处理自然图像时，我们获得的像素值在 [0,255] 区间中，常用的处理是将这些像素值除以 255，使它们缩放到 [0,1] 中. 

### 逐样本均值消减——样本维度

如果你的数据是**平稳**的（即数据每一个维度的统计都服从相同分布），那么你可以考虑在每个样本上**减去数据的统计平均值(逐样本计算)**。 

**例子：**对于图像，这种归一化可以移除图像的平均亮度值 (intensity)。很多情况下我们对图像的照度并不感兴趣，而更多地关注其内容，这时对每个数据点移除像素的均值是有意义的。

**注意：**虽然该方法广泛地应用于图像，但在处理彩色图像时需要格外小心，具体来说，是因为不同色彩通道中的像素并不都存在平稳特性。 

### 特征标准化——特征维度

特征标准化指的是（独立地）使得数据的每一个维度具有零均值和单位方差。这是归一化中最常见的方法并被广泛地使用（**例如**，在使用支持向量机（SVM）时，特征标准化常被建议用作预处理的一部分）。

在实际应用中，特征标准化的具体做法是：首先**计算每一个维度上数据的均值**（使用全体数据计算），之后在**每一个维度上都减去该均值**。下一步便是在数据的**每一维度上除以该维度上数据的标准差**。 

**例子**:处理音频数据时，常用 Mel 倒频系数 [MFCCs](http://en.wikipedia.org/wiki/Mel-frequency_cepstrum) 来表征数据。然而MFCC特征的第一个分量（表示直流分量）数值太大，常常会掩盖其他分量。这种情况下，为了平衡各个分量的影响，通常对特征的每个分量独立地使用标准化处理。 

## PCA/ZCA白化

在做完简单的归一化后，白化通常会被用来作为接下来的预处理步骤，它会使我们的算法工作得更好。实际上许多深度学习算法都依赖于白化来获得好的特征。 

在进行 PCA/ZCA 白化时，首先使特征零均值化是很有必要的，这保证了 ![ \frac{1}{m} \sum_i x^{(i)} = 0 ](http://ufldl.stanford.edu/wiki/images/math/e/3/8/e38353138423fe3c99226921e02ee649.png)。特别地，这一步需要在计算协方差矩阵前完成。（唯一**例外**的情况是已经进行了逐样本均值消减，并且数据在各维度上或像素上是平稳的。） 

接下来在 PCA/ZCA 白化中我们需要选择合适的 `epsilon`（回忆一下，这是规则化项，*对数据有低通滤波作用*）。 选取合适的 `epsilon` 值对特征学习起着很大作用，下面讨论在两种不同场合下如何选取 `epsilon`： 

###   基于重构的模型

在基于重构的模型中(包括自编码器，稀疏编码，受限 Boltzman 机（RBM），k-均值（K-Means）)，经常倾向于选取合适的 `epsilon` 以使得白化达到低通滤波的效果。

（**译注**：通常认为数据中的高频分量是噪声，低通滤波的作用就是尽可能抑制这些噪声，同时保留有用的信息。在 PCA 等方法中，假设数据的信息主要分布在方差较高的方向，方差较低的方向是噪声（即高频分量），因此后文中 `epsilon` 的选择与特征值有关）。

一种检验 `epsilon` 是否合适的方法是用该值对数据进行 ZCA 白化，然后<u>对白化前后的数据进行可视化</u>。如果 `epsilon` 值过低，白化后的数据会显得噪声很大；相反，如果 `epsilon` 值过高，白化后的数据与原始数据相比就过于模糊。

一种直观上得到 `epsilon` 大小的方法是以<u>图形方式画出数据的特征值</u>，如下图的例子所示，你可以看到一条"长尾"，它**对应于数据中的高频噪声部分**。

你需要选取合适的 `epsilon`，使其能够在很大程度上过滤掉这条"长尾"，也就是说，选取的 `epsilon` 应大于大多数较小的、反映数据中噪声的特征值。 

[![ZCA Eigenvalues Plot.png](http://ufldl.stanford.edu/wiki/images/9/91/ZCA_Eigenvalues_Plot.png)](http://ufldl.stanford.edu/wiki/index.php/File:ZCA_Eigenvalues_Plot.png) 

在基于重构的模型中，损失函数有一项是用于惩罚那些与原始输入数据差异较大的重构结果（**译注**：以自动编码机为例，要求输入数据经过编码和解码之后还能尽可能的还原输入数据）。

如果 `epsilon` 太小，白化后的数据中就会包含很多噪声，而模型要拟合这些噪声，以达到很好的重构结果。因此，**对于基于重构的模型来说，对原始数据进行低通滤波就显得非常重要**。 

提示：如果数据已被缩放到合理范围(如[0,1])，可以从*epsilon* = 0.01或*epsilon* = 0.1开始调节`epsilon`。 

###   基于正交化ICA（独立成分分析）的模型

对基于正交化ICA的模型来说，保证输入数据尽可能地白化（即协方差矩阵为单位矩阵）非常重要。

这是因为：**这类模型需要对学习到的特征做正交化，以解除不同维度之间的相关性**（详细内容请参考 [ ICA ](http://ufldl.stanford.edu/wiki/index.php/Independent_Component_Analysis) 一节）。因此在这种情况下，`epsilon` 要足够小（比如 *e**p**s**i**l**o**n* = 1*e* − 6）。 

**提示**：我们也可以在PCA白化过程中同时降低数据的维度。这是一个很好的主意，因为这样可以大大提升算法的速度（减少了运算量和参数数目）。确定要保留的主成分数目有一个经验法则：**即所保留的成分的总方差达到总样本方差的 99% 以上**。(详细内容请参考[ PCA ](http://ufldl.stanford.edu/wiki/index.php/PCA#Number_of_components_to_retain)) 

**注意**: 在使用分类框架时，我们应该只基于训练集上的数据计算PCA/ZCA白化矩阵。需要保存以下两个参数留待测试集合使用：

* 用于零均值化数据的平均值向量；

* 白化矩阵。测试集需要采用这两组保存的参数来进行相同的预处理。 

##   大图像

对于大图像，采用基于 PCA/ZCA 的白化方法是不切实际的，因为协方差矩阵太大。在这些情况下我们退而使用 **1/f 白化方法**（更多内容后续再讲）。 

##   标准流程

在这一部分中，我们将介绍几种在一些数据集上有良好表现的预处理标准流程. 

###   自然灰度图像

==均值消减->PCS/ZCA白化==

灰度图像具有平稳特性，我们通常在第一步对每个数据样本分别做均值消减（即减去直流分量），然后采用 PCA/ZCA 白化处理，其中的 `epsilon` 要**足够大以达到低通滤波**的效果。 

###   彩色图像

==简单缩放->PCA/ZCA白化==

对于彩色图像，<u>色彩通道间并不存在平稳特性</u>。因此我们通常首先对数据进行特征缩放（使像素值位于 [0,1] 区间），然后使用足够大的 `epsilon` 来做 PCA/ZCA。**注意**在进行 PCA 变换前需要对特征进行分量均值归零化。 

###   音频 (MFCC/频谱图)

==特征标准化->PCA/ZCA 白化==

对于音频数据 (MFCC 和频谱图)，每一维度的取值范围（方差）不同。

例如 MFCC  的第一分量是直流分量，通常其幅度远大于其他分量，尤其当特征中包含时域导数 (temporal derivatives)  时（这是音频处理中的常用方法）更是如此。因此，**对这类数据的预处理通常从简单的数据标准化开始**（即使得数据的每一维度均值为零、方差为 1），然后进行  PCA/ZCA 白化（使用合适的 `epsilon`）。 

###   MNIST 手写数字

==简单缩放/逐样本均值消减（->PCA/ZCA 白化）==

MNIST 数据集的像素值在 [0,255] 区间中。我们首先将其缩放到 [0,1] 区间。实际上，进行逐样本均值消去也有助于特征学习。

*注：也可选择以对 MNIST 进行 PCA/ZCA 白化，但这在实践中不常用。*

## 补充

### one hot 独热编码

> https://www.cnblogs.com/haobang008/p/5911466.html
>
> https://yq.aliyun.com/articles/126741

#### 问题由来

在很多机器学习任务中，特征并不总是连续值，而有可能是分类值。例如，考虑一下的三个特征：

```
["male", "female"]
["from Europe", "from US", "from Asia"]
["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]
```

如果将上述特征用数字表示，效率会高很多。例如：

```
["male", "from US", "uses Internet Explorer"] 表示为[0, 1, 3]
["female", "from Asia", "uses Chrome"] 表示为[1, 2, 1]
```

但是，即使转化为数字表示后，上述数据也不能直接用在我们的分类器中。因为，分类器往往默认数据数据是连续的，并且是有序的。但是，按照我们上述的表示，数字并不是有序的，而是随机分配的。

#### 独热编码

为了解决上述问题，其中一种可能的解决方法是采用独热编码（One-Hot Encoding）。

独热编码即 One-Hot 编码，又称一位有效编码，其方法是**使用N位状态寄存器来对N个状态进行编码**，每个状态都由他独立的寄存器位，并且**在任意时候，其中只有一位有效**。

例如：

```text
自然状态码为：000,001,010,011,100,101
独热编码为：000001,000010,000100,001000,010000,100000
```

可以这样理解，对于每一个特征，如果它有m个可能值，那么经过独热编码后，**就变成了m个二元特征**。并且，这些特征互斥，每次只有一个激活。因此，数据会变成稀疏的。

这样做的好处主要有：

1. 解决了分类器不好处理属性数据的问题
2. 在一定程度上也起到了扩充特征的作用

#### 对上述问题的应用

对于上述的问题，性别的属性是二维的，同理，地区是三维的，浏览器则是四维的。

这样，我们可以采用One-Hot编码的方式对上述的样本“["male"，"US"，"Internet Explorer"]”编码：

* “male”则对应着[1，0]
* 同理“US”对应着[0，1，0]
* “Internet Explorer”对应着[0,0,0,1]

则完整的特征数字化的结果为：[1,0,0,1,0,0,0,0,1]。**这样导致的一个结果就是数据会变得非常的稀疏**。

> 将各自对应的编码结果串联到一个向量里

#### 简单应用举例

```python
from numpy import argmax

# define input string
data = 'hello world'
print(data)

# define universe of possible input values
alphabet = 'abcdefghijklmnopqrstuvwxyz '

# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# integer encode input data
# 把data中的数据转换为对应的整数编码
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)

# one hot encode
onehot_encoded = list()
for value in integer_encoded:
    # 每个data中的字符都对应一个单独的letter列表，来存放对应的编码
    letter = [0 for _ in range(len(alphabet))]
    # 只在出现对应字母的位置上让其置一
    letter[value] = 1
    onehot_encoded.append(letter)
print(onehot_encoded)

# invert encoding
inverted = int_to_char[argmax(onehot_encoded[0])]
print("invert encoding=", inverted)
```

所有可能的输入的映射都是从char值创建为整数值。然后使用该映射对输入字符串进行编码。我们可以看到输入'h'中的第一个字母被编码为7。然后将整数编码转换为one hot编码。一次完成一个整数编码的字符。创建0个值的列表，以便字母表的长度可以表示任何预期的字符的长度。

接下来，特定字符的索引标记为1。我们可以看到，编码为7的第一个字母“h”整数由二进制向量表示，长度为27，第七个索引标记为1。

最后，我们反转第一个字母的编码并打印结果。我们通过使用`NumPy argmax()`函数定位具有最大值的二进制向量中的索引，然后使用字符值的反向查找表中的整数来查找对应的字符。

输出：

```python
hello world
[7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3]
# 这里调整了下格式
[
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
invert encoding= h
```

#### One-Hot Encode with scikit-learn

在这个例子中，我们假设你有一个输出序列如下3个标签：

```
cold，warm，hot
```

10个时间序列的示例顺序可以是：

```
cold，cold，warm，cold，hot，hot，warm，cold，warm，hot
```

这将首**先需要一个整数编码**，如1，2，3，然后是**整数到one hot编码**具有3个值的二进制向量，例如[1,0,0]。这个情况下提供序列中每个可能值的至少一个示例。因此，我们可以使用自动方法来定义整数到二进制向量的映射。

在这个例子中，我们将使用scikit学习库的编码器。具体来说，[LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html?spm=a2c4e.11153940.blogcont126741.7.3f862e6fEtWKyS)创建标签的整数编码，[OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html?spm=a2c4e.11153940.blogcont126741.8.3f862e6fEtWKyS)用于创建整数编码值的one hot编码。

```python
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)
```

输出：

```python
['cold' 'cold' 'warm' 'cold' 'hot' 'hot' 'warm' 'cold' 'warm' 'hot']
# 这里将 cold->0 warm->2 hot->1
[0 0 2 0 1 1 2 0 2 1]
#cold hot warm
[[ 1.  0.  0.]
 [ 1.  0.  0.]
 [ 0.  0.  1.]
 [ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]
 [ 1.  0.  0.]
 [ 0.  0.  1.]
 [ 0.  1.  0.]]
['cold']
```

运行示例首先打印标签序列。之后是标签的整数编码，最后是one hot编码。训练数据包含所有可能示例的集合，因此我们可以依靠整数和one hot编码变换来创建标签到编码的完整映射。

默认情况下，OneHotEncoder类将返回更高效的稀疏编码，这可能不适用于某些应用程序。例如使用Keras深度学习库。在这种情况下，我们通过设置sparse = False这个参数来禁用稀疏返回类型。

首先，我们可以使用` NumPy argmax()` 函数来找到具有最大值的列的索引。然后可以将其输入到LabelEncoder，以计算反向变换回文本标签。

#### One Hot Encode with Keras

您可能有一个**已经是整数编码的序列**。

经过处理后，您可以直接使用整数。或者，您可以直接对整数进行one hot 编码。这是非常重要的——如果**整数没有真正的顺序关系并且只是标签的占位符**。

Keras提供了一个名为`to_categorical()`的函数，它可以帮助你使用one hot编码整数数据。接下来我们看一个小例子。在这个例子中，我们有4个整数值[0,1,2,3]，我们有以下10个数字的输入序列：

```
Data = [1,3,2,0,3,2,2,1,0,1]
```

该序列**具有已知了的所有值**，因此我们可以直接使用`to_categorical()`函数。以下列出了此功能的完整示例。

```python
>>> from numpy import array
>>> from numpy import argmax
>>> from keras.utils import to_categorical
Using TensorFlow backend.
>>> 
>>> # define example
... data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
>>> data = array(data)
>>> print(data)
[1 3 2 0 3 2 2 1 0 1]
>>> 
>>> # one hot encode
... encoded = to_categorical(data)
>>> print(encoded)
[[0. 1. 0. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]
 [1. 0. 0. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]
 [0. 0. 1. 0.]
 [0. 1. 0. 0.]
 [1. 0. 0. 0.]
 [0. 1. 0. 0.]]
>>> 
>>> # invert encoding
... inverted = argmax(encoded[0])
>>> print(inverted)
1
```

运行示例并打印输入序列。

然后将整数编码为二进制向量并打印。我们可以看到，正如我们预期的那样，第一个整数值1被编码为[0，1，0，0]。然后，我们使用`NumPy argmax()`函数反转编码，该函数返回第一个整数的期望值1的序列中的第一个值。

#### 处理离散型特征和连续型特征并存的情况，如何做归一化

> 参考博客进行了总结：
> https://www.quora.com/What-are-good-ways-to-handle-discrete-and-continuous-inputs-together

总结如下：

1. 拿到获取的原始特征，必须对每一特征分别进行归一化，比如，特征A的取值范围是[-1000,1000]，特征B的取值范围是[-1,1]。如果使用logistic回归，`w1*x1+w2*x2`，因为x1的取值太大了，所以x2基本起不了作用。
   所以，必须进行特征的归一化，每个特征都单独进行归一化。

2. 连续型特征归一化的常用方法：

   1. Rescale bounded continuous features**: **

      All continuous input that are bounded, rescale them to [-1, 1] through x = (x - (max - min)/2)/(max - min).

      线性放缩到[-1,1]

   2. Standardize all continuous features:

      All continuous input should be standardized and by this I mean, for every continuous feature, compute its mean (u) and standard deviation (s) and do x = (x - u)/s.

      放缩到均值为0，方差为1

3. 离散型特征的处理方法：

   Binarize categorical/discrete features:

   For all categorical features, represent them as multiple boolean features. For example, instead of having one feature called `marriage_status`, have 3 boolean features - `married_status_single`, `married_status_married`, `married_status_divorced` and appropriately set these features to 1 or -1. As you can see, for every categorical feature, you are adding k binary feature where k is the number of values that the categorical feature takes.

   对于离散的特征基本就是按照one-hot编码，该离散特征有多少取值，就用多少维来表示该特征。

#### 为什么使用one-hot编码来处理离散型特征

这是有理由的，不是随便拍脑袋想出来的！！！

具体原因，分下面几点来阐述： 

##### Why do we binarize categorical features?

We binarize the categorical input so that they can be thought of as a vector from the Euclidean space (we call this as embedding the vector in the Euclidean space).

使用one-hot编码，将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点。

##### Why do we embed the feature vectors in the Euclidean space?

Because many algorithms for classification/regression/clustering etc. requires computing distances between features or similarities between features. And many definitions of distances and similarities are defined over features in Euclidean space. So, we would like our features to lie in the Euclidean space as well.

将离散特征通过one-hot编码映射到欧式空间，是因为，在回归，分类，聚类等机器学习算法中，特征之间距离的计算或相似度的计算是非常重要的，而我们常用的距离或相似度的计算都是在欧式空间的相似度计算，计算余弦相似性，基于的就是欧式空间。

##### Why does embedding the feature vector in Euclidean space require us to binarize categorical features?

Let us take an example of a dataset with just one feature (say `job_type` as per your example) and let us say it takes three values 1,2,3.

Now, let us take three feature vectors $x_1 = (1), x_2 = (2), x_3 = (3)$. What is the euclidean distance between $x_1$ and$ x_2$, $ x_2$ and $x_3 $& $ x_1 $and$ x_3$? $d(x_1, x_2) = 1, d(x_2, x_3) = 1, d(x_1, x_3) = 2$. This shows that distance between job type 1 and job type 2 is smaller than job type 1 and job type 3. 

**Does this make sense**? Can we even rationally define a proper distance between different job types? In many cases of categorical features, we can properly define distance between different values that the categorical feature takes. In such cases, isn't it fair to assume that all categorical features are equally far away from each other?

Now, let us see what happens when we binary the same feature vectors. Then, x_1 = (1, 0, 0), x_2 = (0, 1, 0), x_3 = (0, 0, 1). Now, what are the distances between them? They are sqrt(2). So, essentially, when we binarize the input, we implicitly state that all values of the categorical features are equally away from each other.

将离散型特征使用one-hot编码，确实会让特征之间的距离计算更加合理。

比如，有一个离散型特征，代表工作类型，该离散型特征，共有三个取值，不使用one-hot编码，其表示分别是$x_1 = (1), x_2 = (2), x_3 = (3)$。两个工作之间的距离是，$d(x_1, x_2) = 1, d(x_2, x_3) = 1, d(x_1, x_3) = 2$。那么$x_1$和$x_3$工作之间就越不相似吗？显然这样的表示，计算出来的特征的距离是不合理。那如果使用one-hot编码，则得到$x_1 = (1, 0, 0), x_2 = (0, 1, 0), x_3 = (0, 0, 1)$，那么两个工作之间的距离就都是`sqrt(2)`.即每两个工作之间的距离是一样的，显得更合理。

##### About the original question?

Note that our reason for why binarize the categorical features is independent of the number of the values the categorical features take, so yes, even if the categorical feature takes 1000 values, we still would prefer to do binarization.

对离散型特征进行one-hot编码是为了让距离的计算显得更加合理。

##### Are there cases when we can avoid doing binarization?

Yes. As we figured out earlier, the reason we binarize is because we want some meaningful distance relationship between the different values. As long as there is some meaningful distance relationship, we can avoid binarizing the categorical feature. 

For example, if you are building a classifier to classify a webpage as important entity page (a page important to a particular entity) or not and let us say that you have the rank of the webpage in the search result for that entity as a feature, then note that the rank feature is categorical, rank 1 and rank 2 are clearly closer to each other than rank 1 and rank 3, so the rank feature defines a meaningful distance relationship and so, in this case, **we don't have to binarize the categorical rank feature**.

More generally, if you can cluster the categorical values into disjoint subsets such that the subsets have meaningful distance relationship amongst them, then you don't have binarize fully, instead you can split them only over these clusters. 

For example, if there is a categorical feature with 1000 values, but you can split these 1000 values into 2 groups of 400 and 600 (say) and within each group, the values have meaningful distance relationship, then instead of fully binarizing, **you can just add 2 features, one for each cluster and that should be fine**.

将离散型特征进行one-hot编码的作用，是为了让距离计算更合理，但如果特征是离散的，并且不用one-hot编码就可以很合理的计算出距离，那么就没必要进行one-hot编码。

比如，该离散特征共有1000个取值，我们分成两组，分别是400和600，两个小组之间的距离有合适的定义，组内的距离也有合适的定义，那就没必要用one-hot 编码

离散特征进行one-hot编码后，编码后的特征，其实每一维度的特征都可以看做是连续的特征。就可以跟对连续型特征的归一化方法一样，对每一维特征进行归一化。比如归一化到[-1,1]或归一化到均值为0,方差为1

##### 有些情况不需要进行特征的归一化

It depends on your ML algorithms, some methods requires almost no efforts to normalize features or handle both continuous and discrete features, like tree based methods: c4.5, Cart, random Forrest, bagging or boosting. But most of parametric models (generalized linear models, neural network, SVM,etc) or methods using distance metrics (KNN, kernels, etc) will require careful work to achieve good results. Standard approaches including binary all features, 0 mean unit variance all continuous features, etc。

* **基于树的方法是不需要进行特征的归一化**，例如随机森林，bagging 和 boosting等。
* **基于参数的模型或基于距离的模型，都是要进行特征的归一化**。

#### one-hot编码为什么可以解决类别型数据的离散值问题

> 暂时没搞明白