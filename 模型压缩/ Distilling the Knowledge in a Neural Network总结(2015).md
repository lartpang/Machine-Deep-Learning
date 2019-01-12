# Distilling the Knowledge in a Neural Network

> 论文: <https://arxiv.org/abs/1503.02531v1>

## Abstract

A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions [3].

Unfortunately, making predictions using a whole ensemble of models is cumbersome(笨重的) and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets.

Caruana and his collaborators [1] have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique.

We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trainedrapidly and in parallel

提高几乎所有机器学习算法性能的一种非常简单的方法是在相同的数据上训练许多不同的模型，然后对它们的预测进行平均[3]。

不幸的是，使用整个模型集合进行预测是很麻烦的，并且可能在计算上太昂贵而无法部署到大量用户，特别是如果单个模型是大型神经网络。

Caruana和他的合作者[1]已经证明，有可能将整体中的知识压缩成一个更易于部署的单一模型，并且我们使用不同的压缩技术进一步开发这种方法。

我们在MNIST上取得了一些惊人的成果，我们表明，**通过将模型集合中的知识提炼到单个模型**中，我们可以显着改善大量使用的商业系统的声学模型。

我们还介绍了一种**由一个或多个完整模型和许多专业模型组成的新型集合，它们学会区分完整模型混淆的细粒度类**。与专家的混合体不同，这些专业模型可以快速且并行地进行训练

## Introduction

Many insects have a larval form that is optimized for extracting energy and nutrients from the environment and a completely different adult form that is optimized for the very different requirements of traveling and reproduction. In large-scale machine learning, we typically use very similar models for the training stage and the deployment stage despite their very different requirements: For tasks like speech and object recognition, training must extract structure from very large, highly redundant datasets but it does not need to operate in real time and it can use a huge amount of computation. Deployment to a large number of users, however, has much more stringent requirements on latency and computational resources. The analogy with insects suggests that we should be willing to train very cumbersome models if that makes it easier to extract structure from the data. The cumbersome model could be an ensemble of separately trained models or a single very large model trained with a very strong regularizer such as dropout [9]. Once the cumbersome model has been trained, we can then use a different kind of training, which we call “distillation” to transfer the knowledge fromthe cumbersome model to a small model that is more suitable for deployment. A version of thisstrategy has already been pioneered by Rich Caruana and his collaborators [1]. **In their important paper they demonstrate convincingly that the knowledge acquired by a large ensemble of models can be transferred to a single small model.**

许多昆虫具有幼虫形式，其优化用于从环境中提取能量和营养物，以及完全不同的成体形式，其针对旅行和繁殖的非常不同的要求进行了优化。在大规模机器学习中，我们通常在训练阶段和部署阶段使用非常相似的模型，尽管它们的要求非常不同：对于语音和对象识别等任务，训练必须从非常大，高度冗余的数据集中提取结构，但它不会需要实时操作，它可以使用大量的计算。然而，对大量用户的部署对延迟和计算资源有更严格的要求。

与昆虫的类比表明，如果能够更容易地从数据中提取结构，我们应该愿意训练非常繁琐的模型。繁琐的模型可以是单独训练的模型的集合，也可以是使用非常强的正则化器（例如辍学）训练的单个非常大的模型[9]。

一旦训练了繁琐的模型，我们就可以使用不同类型的训练，我们称之为“蒸馏”，*将知识从繁琐的模型转移到更适合部署的小模型*。这种策略的一个版本已经由Rich Caruana及其合作者开创了[1]。 **在他们的重要论文中，他们令人信服地证明，大型模型集合所获得的知识可以转移到一个小型模型中。**

A conceptual block that may have prevented more investigation of this very promising approach is that we tend to identify the knowledge in a trained model with the learned parameter values and this makes it hard to see how we can change the form of the model but keep the same knowledge. A more abstract view of the knowledge, that frees it from any particular instantiation, is that it is a learned mapping from input vectors to output vectors.

一个概念的障碍, 可能已经阻碍了这个非常有前景的方法的探索, 这也是我们倾向于使用学习的参数值来识别训练模型中的知识，这使我们很难看到如何改变模型的形式但是保持相同的知识。从任何特定实例中释放知识的更抽象的视图是它是**从输入向量到输出向量的学习到的映射**。

For cumbersome models that learn to discriminate between a large number of classes,  the normal training objective is to maximize the average logprobability of the correct answer, but a side-effect of the learning is that the trained model assigns probabilities to all of the incorrect answers and even when these probabilities are very small, some of them are much larger than others. The relative probabilities of incorrect answers tell us a lot about how the cumbersome model tends to generalize.

对于学习区分大量类别的繁琐模型，正常的训练目标是最大化正确答案的平均对数概率，但**学习的副作用是训练的模型为所有不正确的答案分配概率, 即使这些概率非常小，其中一些比其他概率大得多**。不正确答案的相对概率告诉我们很多关于繁琐模型如何泛化的信息。

An image of a BMW, for example, may only havea very small chance of being mistaken for a garbage truck, but that mistake is still many times more probable than mistaking it for a carrot.

例如，宝马的图像可能只有很小的机会被误认为是垃圾车，但这种错误仍然比将其误认为胡萝卜的可能性高很多倍。

It is generally accepted that the objective function used for training should reflect the true objective of the user as closely as possible. Despite this, models are usually trained to optimize performance on the training data when the real objective is to generalize well to new data.  It would clearly be better to train models to generalize well, but **this requires information about the correct way to generalize and this information is not normally available**.

人们普遍认为，用于训练的目标函数应尽可能地反映用户的真实目标。尽管如此，当真正的目标是很好地泛化到新数据时，通常训练模型以优化训练数据的性能。显然可以更好地训练模型进行泛化，但**这需要有关正确泛化方法的信息，而且这些信息通常不可用**。

When we are distilling the knowledge from a large model into a small one, however, we can train the small model to generalize in the same way as the large model.  If the cumbersome model generalizes well, because, for example, it is the average of a large ensemble of different models, a small model trained to generalize in the same way will typically do much better on test data than a small model that is trained in the normal way on the same training set as was used to train the ensemble.

然而，当我们将大型模型中的知识提炼成小型模型时，我们可以训练小模型以与大型模型相同的方式进行泛化。如果繁琐的模型很好地泛化，因为，例如，它是不同模型的大集合的平均值，*训练以相同方式进行泛化的小模型通常在测试数据上比按照正常方式在相同的训练集上受过训练的小模型上做得更好*。与用于训练整体的训练集相同的正常方式。

An obvious way to transfer the generalization ability of the cumbersome model to a small model is to use the class probabilities produced by the cumbersome model as “soft targets” for training the small model.

将繁琐模型的泛化能力转移到小模型的一种显而易见的方法是**使用由繁琐模型产生的类概率作为训练小模型的“软目标”**。

For this transfer stage, we could use the same training set or a separate “transfer” set. When the cumbersome model is a large ensemble of simpler models, we can use an arithmetic or geometric mean of their individual predictive distributions as the soft targets.

对于此转移阶段，我们可以使用相同的训练集或单独的“转移”集。*当繁琐的模型是一个较大的简单模型集合时，我们可以使用其各自预测分布的算术或等距平均值作为软目标*。

When the soft target shave high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases, so the small model can often be trained on much less data than the original cumbersome model and using a much higher learning rate.

将繁琐模型的泛化能力转移到小模型的一种显而易见的方法是使用由笨重模型产生的类概率作为训练小模型的“软目标”。对于此转移阶段，我们可以使用相同的训练集或单独的“转移”集。当繁琐的模型是较大的简单模型集合时，我们可以使用其各自预测分布的算术或几何平均值作为软目标。当软目标具有高熵时，它们为每个训练案例提供的信息远多于硬目标，并且在训练案例之间梯度的变化更小，因此小模型通常可以在比原始繁琐模型少得多的数据上训练并使用更高的学习率。

For tasks like MNIST in which the cumbersome model almost always produces the correct answer with very high confidence, much of the information about the learned function resides in the ratios of very small probabilities in the soft targets. For example,  one version of a 2 may be given aprobability of $10^{−6}$ of being a 3 and $10^{−9}$ of being a 7 whereas for another version it may be the other way around. This is valuable information that defines arich similarity structure over the data(i.  e.it says which 2’s look like 3’s and which look like 7’s) but it has very little influence on the cross-entropy cost function during the transfer stage because the probabilities are so close to zero. Caruana and his collaborators circumvent this problem by using the logits (the inputs to the final softmax) rather than the probabilities produced by the softmax as the targets for learning the small model and they minimize the squared difference between the logits produced by the cumbersome model and the logits produced by the small model.

Our more general solution, called “distillation”,is to raise the temperature of the final softmax until the cumbersome model produces a suitably soft set of targets. We then use the same high temperature when training the small model to match these soft targets. We show later that matching the logits of the cumbersome model is actually a special case of distillation.

对于像MNIST这样的任务，其中繁琐的模型几乎总能以非常高的信度产生正确的答案，关于学习函数的大部分信息都存在于软目标中非常小的概率的比率中。例如，2的一个版本可以被赋予3的概率为$10^{−6}$，而7为$10^{−9}$，而对于另一个版本，它可能是另一种方式。这是有价值的信息，它定义了数据的丰富相似性结构（即它表示哪个2看起来像3，哪个看起来像7），但它在转移阶段对交叉熵成本函数的影响非常小，因为概率是如此接近于零。Caruana和他的合作者通过**使用logits（最终softmax的输入）来解决这个问题，而不是用softmax产生的概率作为学习小模型的目标**，并且他们最小化了繁琐模型产生的logits和小模型产生的logits之间的平方差异。

我们更通用的解决方案，称为“蒸馏”，是提高最终softmax的温度，直到笨重的模型产生适当柔软的目标组。然后我们在训练小模型时使用相同的高温来匹配这些软目标。我们后来表明，匹配繁琐模型的logits实际上是一个特殊的蒸馏情况。

The transfer set that is used to train the small model could consist entirely of unlabeled data [1]or we could use the original training set.  We have found that using the original training set workswell, especially if we add a small term to the objective function that encourages the small model to predict the true targets as well as matching the soft targets provided by the cumbersome model. Typically, the small model cannot exactly match the soft targets and erring in the direction of the correct answer turns out to be helpful.

用于训练小模型的传递集可以完全由未标记的数据组成[1]，或者我们可以使用原始训练集。我们发现**使用原始训练集效果很好**，特别是如果我们在目标函数中添加一个小项，鼓励小模型预测真实目标以及匹配繁琐模型提供的软目标。通常，小模型不能与软目标完全匹配，并且在正确答案的方向上犯错是有帮助的。

## Distillation

神经网络通常通过使用“softmax”输出层产生类概率，该输出层转换logit zi(通过将zi与其他logits进行比较, 来为每个类计算概率qi)。

![img](assets/2019-01-12-10-40-23.png)

这里的T是一个温度, 通常被设为1. 使用一个高一点的T, 将产生一个更为柔软的类别概率分布.

In the simplest form of distillation, knowledge is transferred to the distilled model by training it ona transfer set and using a soft target distribution for each case in the transfer set that is produced byusing the cumbersome model with a high temperature in its softmax. The same high temperature isused when training the distilled model, but after it has beentrained it uses a temperature of 1.

在最简单的蒸馏形式中，通过在转移集上训练知识并将转移集中的每个案例使用软目标分布，将知识转移到蒸馏模型，该转移集通过使用其softmax中具有高温的繁琐模型产生。训练蒸馏模型时使用相同的高温，但被训练之后, 使用温度为1.

When the correct labels are known for all or some of the transfer set, this method can be significantly improved by also training the distilled model to produce the correct labels.

One way to do this is to use the correct labels to modify the soft targets, but we found that **a better way is to simply use a weighted average of two different objective functions**.

* The first objective function is the cross entropy with the soft targets and this cross entropy is computed using the same high temperature in the softmax of the distilled model as was used for generating the soft targets from the cumbersome model.
* The second objective function is the cross entropy with the correct labels. This is computed using exactly the same logits in softmax of the distilled model but at a temperature of 1.

We found that thed best results were generally obtained by using a condiderably lower weight on the second objective function. Since the magnitudes of the gradients produced by the soft targets scale as $1/T^2$ it is important to multiply them by $T^2$ when using both hard and soft targets.

This ensures that the relative contributions of the hard and soft targets remain roughly unchanged if the temperature used for distillation is changed while experimenting with meta-parameters

当已知所有或部分转移集的正确标签时，通过训练蒸馏模型以产生正确的标签，可以显着改善该方法。一种方法是使用正确的标签来修改软目标，但我们发现更好的方法是简单地使用两个不同目标函数的加权平均值。

* 第一个目标函数是具有软目标的交叉熵，并且使用与用于从笨重模型生成软目标的蒸馏模型的softmax中相同的高温来计算该交叉熵。
* 第二个目标函数是具有正确标签的交叉熵。这是使用蒸馏模型的softmax中的完全相同的logits计算的，但温度为1。

我们发现通常通过在第二目标函数上使用可忽略不计的较低权重来获得最佳结果。**由于软目标产生的梯度的大小按比例缩放为$1/T^2$，因此在使用硬目标和软目标时，将它们乘以非常重要$T^2$**。这确保了如果在试验元参数时改变用于蒸馏的温度，则硬和软目标的相对贡献保持大致不变。

###
