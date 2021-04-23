## Meta learning
我们看了机器学习(Machine Learning)和深度学习(Deep Learning)，特别是深度学习，通常定义函数模型，也就是定义好一个神经网络结构，
所谓 Meta learning 也就是让机器学会学习，这一次学习任务不是具体图像识别、分类还是语音识别或者机器学习而是让机器学会学习这个任务。

听起来有那么点高大尚，通过对比深度学习的流程来看一看，通常我们设计一个神经网络结构，接下来初始参数，输入样本到模型进行训练，然后我们来定义目标，也可以理解衡量模型，或者更进一步说是衡量模型参数的函数。对于回归问题损失函数可能是最小平方，对于分类可以是交叉熵。接下来就是通过反向传播以及结合选择优化器来更新参数，将这个参数作为下一轮迭代的初始参数，就这样不断迭代来进行学习。


尽管大多数流行和经典的神经网络模型的架构都是由人类专家设计出来的，但这并不意味着我们已经探索了整个网络架构空间并确定了最佳方案。到现在为止我们还没有一个确定的方向，或者说是一套理论，通过这套理论可以设计有效的神经网络。还在摸索中，特别参数调试都是在不断摸索查看如何设计或拿到一个好的方案。那么这些繁琐和暴力事情，或者可能其中可能蕴含规律是不是可以交给机器来做，这是 NAS 的初衷，也是大家热捧的动力。

如果我们采用系统化和自动化的方式来学习高性能的模型架构，我们将有更好的机会找到最佳解决方案。

NAS search algorithms sample a population of child networks. It receives the child models’ performance metrics as rewards and learns to generate high-performance architecture candidates. You may a lot in common with the field of hyperparameter search.

Inspired by the design of using repeated modules in successful vision model architectures (e.g. Inception, ResNet), the NASNet search space (Zoph et al. 2018) defines the architecture of a conv net as the same cell getting repeated multiple times and each cell contains several operations predicted by the NAS algorithm. A well-designed cell module enables transferability between datasets. It is also easy to scale down or up the model size by adjusting the number of cell repeats.

### Cell-based Representation
最近神经网络主要主要集中模式识别，也就是空间表现优越的卷积神经网络和时序表现优越的循环神经网络，受经典的视觉任务基于卷积神经网络架构（如Inception、ResNet）的启发，在这些网络中通常都是重复使用一些相同结构，通过堆叠而得到这些网络。

NASNet搜索空间（Zoph等人，2018）将定卷积网的架构定义为同一个单元得到多次重复，每个单元包含 NAS 算法预测的几个操作。一个精心设计的单元模块能够在数据集之间进行转移。通过调整单元格重复的数量，也很容易缩小或扩大模型的规模。

### 搜索策略
NAS 搜索算法对一个子网络空间进行采样。将子网络模型的性能指标作为奖励，来学习生成高性能的架构。也适用与超参数搜索领域。


而对于 Meta

今天语言作为信息传递的工具，有了语言的载体知识可以得以保留、继承和传播。信息共享让我们通过其他人的认知来扩展对这个世界认知。


## 多任务
- 这些任务

### 跨摄像头的多任务
- 在特斯拉周围安装了 8 个摄像头，每个摄像头都会有识别任务，同样场景会出现不同摄像头视图，从而实现了跨摄像头对网络结构提取特征的共享
### 多任务之间关联
- 很多任务之间是存在一定关系的，有的任务之间是一种相辅相成的关系，有的任务之间是一种竞争的关系。所以有效地分析任务之间关联性，合理设计神经网络架构，让神经网络中结构得到充分利用。
### 时空上任务之间关系
- 从时序上任务也是有关联的，如何设计网络结构可以跨时间来共享信息。

谁让



### 多任务中需要考虑问题
- 多任务中不同任务可能尺度的损失函数
- 多任务中不同任务间可能存在差异，有些任务的重要性会由于其他任务
- 从任务的难易程度上，不同任务之间也会有差异，现在很多自适应网络会对样本识别难易程度预测而确定采取模型或者
- 从采集数据上来看，不同任务样本也是不均衡的，有些任务样本会多一些，有些任务的样本可能会少一些
- 数据噪音在不同任务之间也会有差异