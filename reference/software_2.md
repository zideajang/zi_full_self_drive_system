今天来聊聊 software 2.0 ，在 software 2.0 引入人工智能，大家可能一想到人工智能写代码场景，脑海出现场景是屏幕上一行一行输出人写代码，想到机器是如何学习人类语言，是不是也要用 RNN 或者 transformer 这样模型来实现人工智能编码呢？其实不然，人工智能为什么需要模仿人去写那些对于人类 friendly 的代码呢? 这样不就是失去人工智能优势了，人工智能本身就是都是运行代码。我们可以一些程序逻辑转换一下，从一行一行指令转换成一个神经网络，将那些人工智能善于工作交给他来做。

software 1.0 中可以将理解为一个经典栈，在栈中将我们写的 Python、C++ 等编程语言编写代码压入栈。由程序员写给计算机的明确指令组成。通过编写每一行代码，程序员在程序空间中确定了一个具有某种行为的特定点。这句话听起来有点抽象，什么是程序空间，我们不是在编程吗，其实我们可以将每条语句或者每块代码理解对应一段逻辑，语句可以理解特征的组合，这些特征的组合对应到程序空间(空间维度与特征对应)的一个点。

尽管在过去几十年里 software 1.0 取得了很大的成功，但进入深度学习时代后，人们逐渐发现，单纯的 software 1.0 模式已经不能够解决很多实际的问题，例如——图像识别。早期的图像识别(~2010年之前)，工程师们需要写复杂的代码提取各式各样的特征(图像边缘、纹理特征等)，也可以理解为工程师自己对例如 cat 理解，以及如何用于制定规则来描述一只猫，其实这种模式通过自己定义规则或特征工程并不会得到好的效果，神经网络好处就是可以通过学习来自己完成特征工程的学习。

事实证明，很大一部分现实世界的问题都具有这样的特性：收集数据或者更一般地说，确定一个的行为，比明确地编写程序要容易得多。由于这个原因以及我将在下面讨论的软件2.0程序的许多其他好处，我们正在目睹整个行业的大规模转型，许多1.0代码被移植到2.0代码中。软件（1.0）正在吞噬世界，现在AI（软件2.0）正在吞噬软件。

在提取出的特征之上使用规则或者传统机器学习方法如支持向量机进行图像的识别。现在，工程师们通过获取大量的图像数据、设计学习的方法、训练一个卷积神经网络，大幅提升了计算机在图像识别上的表现。

Databases. More traditional systems outside of Artificial Intelligence are also seeing early hints of a transition. For instance, “The Case for Learned Index Structures” replaces core components of a data management system with a neural network, outperforming cache-optimized B-Trees by up to 70% in speed while saving an order-of-magnitude in memory.


数据库。人工智能以外的更多传统系统也看到了转型的早期迹象。例如，"The Case for Learned Index Structures "用神经网络取代了数据管理系统的核心组件，在速度上优于缓存优化的B-Trees 达70%，同时节省了一个数量级的内存。

那么什么是 B-Tree 也就是二叉树，有关二叉树的相关知识就不多介绍了。


Computationally homogeneous. A typical neural network is, to the first order, made up of a sandwich of only two operations: matrix multiplication and thresholding at zero (ReLU). Compare that with the instruction set of classical software, which is significantly more heterogenous and complex. Because you only have to provide Software 1.0 implementation for a small number of the core computational primitives (e.g. matrix multiply), it is much easier to make various correctness/performance guarantees.

每一个应用都是为了解决一个实际问题，

强化学习 GTP3.0 API

机器

其实神经网络可以进行密集计算


计算上的同质性。一个典型的神经网络在第一阶上是由只有两种操作的三明治组成的：矩阵乘法和零点阈值（ReLU）。与经典软件的指令集相比，它的异质性和复杂性要高得多。因为你只需要为少量的核心计算基元（如矩阵乘法）提供软件1.0的实现，所以更容易做出各种正确性/性能保证。


Simple to bake into silicon. As a corollary, since the instruction set of a neural network is relatively small, it is significantly easier to implement these networks much closer to silicon, e.g. with custom ASICs, neuromorphic chips, and so on. The world will change when low-powered intelligence becomes pervasive around us. E.g., small, inexpensive chips could come with a pretrained ConvNet, a speech recognizer, and a WaveNet speech synthesis network all integrated in a small protobrain that you can attach to stuff.

烘烤到硅中很简单。作为一个推论，由于神经网络的指令集相对较小，实现这些网络明显更容易接近硅，例如使用定制的ASIC、神经形态芯片等等。当低功率的智能在我们周围普遍存在时，世界将发生变化。例如，小型、廉价的芯片可以配备一个预先训练好的ConvNet、一个语音识别器和一个WaveNet语音合成网络，这些都集成在一个小型的原生大脑中，你可以把它安装在东西上。

Constant running time. Every iteration of a typical neural net forward pass takes exactly the same amount of FLOPS. There is zero variability based on the different execution paths your code could take through some sprawling C++ code base. Of course, you could have dynamic compute graphs but the execution flow is normally still significantly constrained. This way we are also almost guaranteed to never find ourselves in unintended infinite loops.

恒定的运行时间。一个典型的神经网络前向传递的每一次迭代都需要完全相同的FLOPS量。基于你的代码在一些庞大的C++代码库中可能采取的不同执行路径，其变化性为零。当然，你可以有动态的计算图，但执行流程通常仍然受到很大的限制。这样一来，我们也几乎可以保证永远不会发现自己处于无意的无限循环中。

Constant memory use. Related to the above, there is no dynamically allocated memory anywhere so there is also little possibility of swapping to disk, or memory leaks that you have to hunt down in your code.
It is highly portable. A sequence of matrix multiplies is significantly easier to run on arbitrary computational configurations compared to classical binaries or scripts.

持续的内存使用。与此相关的是，在任何地方都没有动态分配的内存，所以也几乎没有交换到磁盘的可能性，或者你必须在代码中寻找的内存泄漏。

软件更容易移植，与 java 这样字节码或者 JavaScript 这样需要解释器的脚步语言编写的应用，主要由矩阵以及操作矩阵运算符更容易运行在不同平台上。(它是高度可移植的。与经典的二进制文件或脚本相比，一连串的矩阵乘法明显更容易在任意的计算配置上运行。)


