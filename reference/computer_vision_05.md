

### 单视图测量

现在我们的目标就是从单张图像恢复场景结构，就必须建立单张视图上几何元素点和三维场景几何元素的对应关系，只有建立了这种关系才能够根据单张图像来建立三维视图。

之前我们学习了相机标定，知道了从世界坐标到摄像机坐标以及最终到成像平面之间的关系。那么我们今天就想利用学到的知识看一看能否通过一张图像将 3D 场景还原呢？

那么我们知道一个摄像机的内外参数，有了这些信息我们是否能够根据这些信息重建出三维图像。也就是可以根据单个图像的测量值 p 去估算 p 呢？答案一般是不能能，这是因为 p 可能位于 C 和 p 定义的直线上的任何位置

但是如果我们图像中一些面，例如建筑的不同立面之间的夹角是 90 度这样的关系，建筑侧立面和地面之间是垂直关系，有了这些信息。

### 2D 变换
#### 等距变换
- 就是把平面上的点经过旋转、平移后变换为另一个点
  $$
  \begin{bmatrix}
    x^{\prime}\\
    y^{\prime}\\
    1
  \end{bmatrix} = \begin{bmatrix}
    R & t\\
    0 & 1
  \end{bmatrix} \begin{bmatrix}
    x\\
    y\\
    1
  \end{bmatrix} = H_e \begin{bmatrix}
    x\\
    y\\
    1
  \end{bmatrix} 
  $$
  

  

- 保留长度和面积

- 3 DOF(自由度)

- 刚性物体的运动
#### 相似变换
- 相似变换就是在等距变换上加了 S 矩阵

$$\begin{bmatrix}
    x^{\prime}\\
    y^{\prime}\\
    1
\end{bmatrix} = \begin{bmatrix}
    SR & t\\
    0 & 1
\end{bmatrix} \begin{bmatrix}
    x\\
    y\\
    1
\end{bmatrix} = H_s \begin{bmatrix}
    x\\
    y\\
    1
\end{bmatrix} S = \begin{bmatrix}
    s & 0\\
    0 & s
\end{bmatrix}$$
- 长度的比值和角度不变
- 4 DOF(自由度)
#### 仿射变换


$$\begin{bmatrix}
    x^{\prime}\\
    y^{\prime}\\
    1
\end{bmatrix} = \begin{bmatrix}
    A & t\\
    0 & 1
\end{bmatrix} \begin{bmatrix}
    x\\
    y\\
    1
\end{bmatrix} = H_a \begin{bmatrix}
    x\\
    y\\
    1
\end{bmatrix} $$

- 仿射变换对于 A 矩阵没有什么要求
- 6 DOF
- 具有平行线的平行关系、面积比值的不变性


#### 射影变换

$$\begin{bmatrix}
    x^{\prime}\\
    y^{\prime}\\
    1
\end{bmatrix} = \begin{bmatrix}
    A & t\\
    v & 1
\end{bmatrix} \begin{bmatrix}
    x\\
    y\\
    1
\end{bmatrix} = H_a \begin{bmatrix}
    x\\
    y\\
    1
\end{bmatrix} $$

### 影消点与影消线
- 

- 也叫做透视变换，和仿射变换的区别就是 v 位置也不再为 0
- 8 DOF
- 共线性、四共线点的交比

#### 内容该要
- 平面上的平行线的交点、**无穷远点**和**无穷远线**
- 无穷远点和无穷远线的 2D 变换，分析一下 2D 环境下投影和映射之间的关系
- 三维空间中的点线面、影消点与影消线，在 3D 点线面经过变换得到了图像上对应的影消点和影消线的这些元素
- 影消点、影消线与三维空间中的直线的方向与面的关系

#### 平面上的线
$$
ax + by + c = 0
$$

这是我们在中学几何中学到用方程形式来表示平面上一条线，可以用向量形式来表示直线方程

$$
l = \begin{bmatrix}
    a\\
    b\\
    c
\end{bmatrix}
$$

$$
If \, x = \begin{bmatrix}
    x_1,x_2
\end{bmatrix}^T \in l \, \begin{bmatrix}
    x_1\\
    x_2\\
    1\\
\end{bmatrix}^T \begin{bmatrix}
    a\\
    b\\
    c
\end{bmatrix} = 0
$$


#### 两条直线的交点
$$
x = l \times l^{\prime}
$$

两条直线的交点就是两个直线参数向量的叉乘，接下来给大家解释一下。我们知道两个向量叉乘后在与其中一个向量进行点乘结果一定为 0。
$$
\begin{aligned}
  l \times l^{\prime} \perp l \rightarrow (l \times l^{\prime}) l = 0 \rightarrow x \in l\\
  l \times l^{\prime} \perp l \rightarrow (l \times l^{\prime}) l^{\prime} = 0 \rightarrow x \in l^{\prime}\\
\end{aligned}
$$


#### 2D 无穷远点
在欧式坐标中点如果变换为齐次坐标后第 3 维为 0 ，此点位于无穷点。因为在欧式坐标中 $x^{\prime}_1/0$

$$
x_{\infty} = \begin{bmatrix}
    x_1^{\prime}\\
    x_2^{\prime}\\
    0
\end{bmatrix}
$$


在我们初中学习过程中，两条平行线是不会相交的，但是上面我们通过两条直线参数进行叉乘可以得到交点

假设有两条平行线分别表示为 $l$ 和 $l^{\prime}$ 因为存在 $-a/b = -a^{\prime}/b^{\prime}$ 所以两条直线是平行关系

$$
l \times l^{\prime} \propto \begin{bmatrix}
    b\\
    -a\\
    0
\end{bmatrix} = x_{\infty}
$$

$$
l^Tx_{\infty} = \begin{bmatrix}
    a & b & c
\end{bmatrix} \begin{bmatrix}
    b\\
    -a\\
    0
\end{bmatrix} = 0
$$


#### 无穷远直线
无穷远点集位于称为无穷远线的一条线上

$$
x_{\infty}^{\prime} = \begin{bmatrix}
    b^{\prime}\\
    -a^{\prime}\\
    0
\end{bmatrix} \, x_{\infty}^{\prime\prime} = \begin{bmatrix}
    b^{\prime\prime}\\
    -a^{\prime\prime}\\
    0
\end{bmatrix} 
$$

$$
\begin{bmatrix}
    x_1\\
    x_2\\
    0
\end{bmatrix}^T \begin{bmatrix}
    0\\
    0\\
    1
\end{bmatrix} = 0
$$


无穷远线可以认为是平面上线的“方向”的集合

$$l_{\infty} = \begin{bmatrix}
    0\\
    0\\
    1
\end{bmatrix}$$

#### 无穷远点的透视变换(2D)
$$H = \begin{bmatrix}
    A & t\\
    v & b
\end{bmatrix}$$

$$p^{\prime} = Hp$$

$$Hp_{\infty} = ? = \begin{bmatrix}
    A & t\\
    v & b
\end{bmatrix}\begin{bmatrix}
    1\\
    1\\
    0
\end{bmatrix}$$