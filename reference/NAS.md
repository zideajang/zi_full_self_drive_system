
无人驾驶是未来趋势，在我看来答案是肯定的。随着新能源电动车不断普及，以及 5G 技术日益成熟，带有无人驾驶智能汽车必将是一个趋势。不过有些技术还是需要突破，毫米波激光雷达价格，以及是否通过其他、其他方式将像 waymo 头顶笨重的耗电激光雷达换一种形式。通过感知拿到计算机视觉也还有一定发展空间，如何实现多任务学习，提高精度和神经网络的可解释性，这些方面还是一定空间。还有就是无人驾驶相关法规还需要完善，


## 现搭建基础网络

最近比较火 NAS 
可能看了之后会有一些疑问，我们今天带着这些疑问再重新看一看 NAS，我们知道在 NAS 有 3 个部分分别是搜索空间、搜索策略和性能评估。

**NAS** 并不是最近才提出的算法，早在上个世纪 19 年代就已经被提出，只不过和深度学习类似，因为最近算力的提升、以及深度学习发展才再次发光发热，简单来说就是用于搜索最佳网络的架构网络，和 Meta 学习也是有点相似的，不过 Meta 学习是基于大量先验任务，通过学习潜在学习方法来实现算法，更加适合受数据量限制或者难以学习到显著特征的任务。而 NAS 这是优化网络结构设计，或者说是一种调参方法。优化方法有点类似暴力搜索，所以最近大家都在做的是对网络进行减枝。

<img src="./neural_architecture_search/001.png">

其实这里想所说一句就是，

RNN 可以看成控制器，

<img src="./neural_architecture_search/002.png">
- 分别是特征图的高度和宽度
- 步进(Stride Height)和 stride Width
- 这里 anchor 表示该层将跳转到哪一个层，这是因为在 ResNet 和 DenseNet 中存在一些残差边

为什么用 RNN 来做控制器好处也就是在网络结构中每一层都会改变 Tensor 的形状，我们需要 Tensor 形状在整个神经网络中变换是连续，因为每一层神经网络的输入都是上一层的输出。

这样来得到控制器用于生成策略，那么状态又是什么呢

### 搜索空间
这里状态空间就是所有可能
如何来计算梯度


有两种搜索方式，跨层多分支，中间层的类型都是可以被搜索，卷积核的结构都包括在搜索空间内。会将把网络切分为基础单元(cell)。

定义适当的搜索空间更利于提升网络的性能的下限，搜索空间复杂程度决定了网络结构的潜力。

<img src="./neural_architecture_search/005.jpg">

#### 单元(cell)

神经网络映射，例如我们通常网络神经网结构都是一个有向无环图，早期比较简单简单的全连接神经网络和卷积神经网络结构可以看成一个链表，cell 也可以看成张量 tensor 的变换从一个结构。但是对于 inception net 和 resnet 以及 densenet 中的结点通常多个前驱和多个后置，这样的结构会变得相对复杂，图类型也是由搜索空间来定义，除此之外层的超参数也都包括在搜索空间内。

#### 块(block)
每个模块符号化了表示为一个 5 元组，其中两个输入 $I_1,I_2$ 各自的操作为 $O_1,O_2$ 和 C 表示如何合并到一起去得到输出

$$|\cal{B}_b| = |\cal{I}_b|^2 \times |\cal{O}|^2 \times |\cal{C}|$$

<img src="./neural_architecture_search/002.png">

#### 离散变连续，使用松弛 relaxation
##### 对张量的操作有若干中
- 3x3 depthwise-separatable conv
- 5x5 sep 
- 3x3 max pooling
- 3x3 avarage pooling
通过 softmax 将标量变为二选一，给定义输入 x 给出一个输出，每一次都是硬性地选择一个。强化学习则是换成概率

$$O^{i,j} = \sum_{o \in \cal{O}}\frac{\exp (\alpha_o^{(i,j)})}{\sum_{o^{\prime}\in \cal{O}} \exp(\alpha_{o^{\prime}}^{(i,j)})} o(x)$$

- 优化、训练完后，就选择概率最大的那项即可

$$o^{(i,j)} = \argmax_{x \in \cal{O}} \alpha_o^{(i,j)}$$

#### 损失函数
- 优化目标函数是用验证集$L_{val}$
- 当网络结构固定后，用训练集训练网络参数$L_{train}$
- BiLevel 优化问题，目标优化验证集最小，训练集上损失函数最小
$$\begin{aligned}
    \min_{\alpha} L_{val}(w^*(\alpha),\alpha)\\
    s.t. \, w^*(\alpha) = \argmin_w L_{train}(w,\alpha)
\end{aligned}$$
#### BiLevel 优化问题的优化方法
- 不能双层循环、每回进行以
- 联想到 SVM 时候的 KKT 算法也就是有条件约束问题
- 可以让 train 和 val 两个 loss 一起优化
- SGD 同时一步来完成优化

$$\nabla_{\alpha} L_{eval}(w^*(\alpha),\alpha) \approx \nabla_{\alpha}(w - \epsilon \nabla_w L_{train}(w,\alpha),\alpha)$$

### 搜索策略
- 本质上就是超参数优化的问题，主流搜索策略强化学习、遗传学习以及基于梯度的优化。强化学习，有点类似 GAN，生成的是网络结构，鉴别的是网络效果。用控制器(RNN)生成一个不定长子网络串，并且训练子网络，以子网络评估得到精度值作为反馈，更新控制器参数。网络生成问题可以简化序列生成问题，按层依次预测网络结构，每 5 个输出用于定义网络结构层，
- 目标就是子网络在测试集上数学期望，



#### 基于强化学习问题


#### 网络的拓扑结构
#### 每层的类型
除了输入和输出层以外的中间层的类型是可选的，这些可选包括全连接层、卷基层、反卷积层、空洞卷积、池化层和激活层等
#### 每层内部的超参数

#### 编码器(encoder)

#### 性能评估策略
NAS 算法需要估计一个给定神经网络结构的性能，这称为性能评估策略


## NAS

今天尽管大多数流行和经典的神经网络模型的架构都是由人类的专家设计出来的，但这并不意味着我们已经探索了整个网络架构的空间并确定了最佳方案。到现在为止我们还没有一个确定的方向，或者说是一套理论，通过这套理论可以设计有效的神经网络。还在摸索中，特别参数调试都是在不断摸索查看如何设计或拿到一个好的方案。那么这些繁琐和暴力事情，或者可能其中可能蕴含规律是不是可以交给机器来做，这是 NAS 的初衷，也是大家热捧的动力。

如果我们采用系统化和自动化的方式来学习高性能的模型架构，我们将有更好的机会找到最佳解决方案。

Automatically learning and evolving network topologies is not a new idea (Stanley & Miikkulainen, 2002). In recent years, the pioneering work by Zoph & Le 2017 and Baker et al. 2017 has attracted a lot of attention into the field of Neural Architecture Search (NAS), leading to many interesting ideas for better, faster and more cost-efficient NAS methods.

其实这种自动化的学习并不断优化网络拓扑结构并不是一个新想法(Stanley & Miikkulainen, 2002)，也就是最近大家想设计出一个模型，这个模型负责设计出一个网络结构，也可以理解为结构化学习一种。随着近年来，Zoph & Le 2017 和 Baker 等人的开创性工作，这些工作吸引了很多人对神经网络结构搜索(NAS)这个研究领域的更多关注，从而带来了更多有趣的想法，基于这些想法出现了更好、更快、更有成本效益的 NAS 方法。

As I started looking into NAS, I found this nice survey very helpful by Elsken, et al 2019. They characterize NAS as a system with three major components, which is clean & concise, and also commonly adopted in other NAS papers.

当我开始研究 NAS 时，看到 Elsken, et al 在 2019 发表那份漂亮的调查报告，这个调查报告对 NAS 的理解非常有帮助。他们将 NAS 描述为一个有三个主要组成部分的系统，也是其他 NAS 论文中普遍采用的。

NAS search algorithms sample a population of child networks. It receives the child models’ performance metrics as rewards and learns to generate high-performance architecture candidates. You may a lot in common with the field of hyperparameter search.


Search space: The NAS search space defines a set of operations (e.g. convolution, fully-connected, pooling) and how operations can be connected to form valid network architectures. The design of search space usually involves human expertise, as well as unavoidably human biases.

**搜索空间**: NAS 的搜索空间定义了一组 tensor 的操作（如卷积、全连接、集合）以及如何将操作连接起来以形成有效的网络结构。搜索空间的设计通常涉及人类的专业知识，这样多少就会收到人类认知所影响。

Search algorithm: A NAS search algorithm samples a population of network architecture candidates. It receives the child model performance metrics as rewards (e.g. high accuracy, low latency) and optimizes to generate high-performance architecture candidates.

**搜索算法**: 一个 NAS 搜索算法对网络架构候选集合进行采样。它接收儿童模型的性能指标作为奖励（如高精确度、低延迟），并进行优化以产生高性能的架构候选者。

**评价策略**: 我们需要测量、估计或预测大量提议的子模型的性能，作为搜索算法的学习的反馈来z。候选人评估的过程可能非常昂贵，许多新的方法已经被提出来以节省时间或计算资源。

The most naive way to design the search space for neural network architectures is to depict network topologies, either CNN or RNN, with a list of sequential layer-wise operations, as seen in the early work of Zoph & Le 2017 & Baker et al. 2017. The serialization of network representation requires a decent amount of expert knowledge, since each operation is associated with different layer-specific parameters and such associations need to be hardcoded. For example, after predicting a conv op, the model should output kernel size, stride size, etc; or after predicting an FC op, we need to see the number of units as the next prediction.

设计神经网络架构的搜索空间的最直观简单方式，是用顺序将神经网络层级，类似链表形式形成的网络拓扑结构，无论是 CNN 还是 RNN，在早期设计出经典神经网络如 AlexNet、VGG 都是这种顺序连接而形成的。网络表示的序列化需要相当数量的专家知识，因为每个 tensor 操作都与其他层的特定参数()相关，例如通过卷积后 tensor 形状发生了改变，为了保持 tensor 形状在整个网络连续性也就是上一层输出是下一层的输入，而且这种关联需要硬编码。例如，要预测连接在卷积后层(操作)，模型应该输出内核大小、跨度大小等；或者在预测一个FC操作后，我们需要看到单位的数量作为下一个预测。



Inspired by the design of using repeated modules in successful vision model architectures (e.g. Inception, ResNet), the NASNet search space (Zoph et al. 2018) defines the architecture of a conv net as the same cell getting repeated multiple times and each cell contains several operations predicted by the NAS algorithm. A well-designed cell module enables transferability between datasets. It is also easy to scale down or up the model size by adjusting the number of cell repeats.

### Cell-based Representation
最近神经网络主要主要集中模式识别，也就是空间表现优越的卷积神经网络和时序表现优越的循环神经网络，受经典的视觉任务基于卷积神经网络架构（如Inception、ResNet）的启发，在这些网络中通常都是重复使用一些相同结构，通过堆叠而得到这些网络。

NASNe t搜索空间（Zoph等人，2018）将定卷积网的架构定义为同一个单元得到多次重复，每个单元包含 NAS 算法预测的几个操作。一个精心设计的单元模块能够在数据集之间进行转移。通过调整单元格重复的数量，也很容易缩小或扩大模型的规模。

### 搜索策略
NAS 搜索算法对一个子网络空间进行采样。将子网络模型的性能指标作为奖励，来学习生成高性能的架构。也适用与超参数搜索领域。
