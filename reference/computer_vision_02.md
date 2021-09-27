## 摄像机标定
所谓摄像机的标定，就是求解摄像机的内外参数。摄像机内、外参数描述了 3 维世界到 2 维像素的映射关系。
$$
P^{\prime} = MP_w = K\begin{bmatrix}
    R & T
\end{bmatrix}P_w
$$


- $K$ 为内参数
- $\begin{bmatrix}
    R & T
    \end{bmatrix}$ 为外参数

那么摄像机标定的目标就是，通过 1 张图片或者多张图像来估算内外参数点，注意这里是估算。所以求解摄像机内、外参数就是要先求解 M 进而求解 R 和 T 来

图，在进行计算机标定时候，我们通常会定义一个标定装置，

- 在世界坐标系中$p_1,p_2,\cdots$ 的位置已知，在标定装置中的世界坐标
- 图像中这些点 $p_1,p_2,\cdots$ 对应点的位置也是已知的
- 目的就是利用这些点来联立方程求解计算摄像机的内，外参数

$$
p_i = \begin{bmatrix}
    u_i\\
    v_i
\end{bmatrix} = \begin{bmatrix}
    \frac{m_1p_i}{m_3p_i}\\
    \frac{m_2p_i}{m_3p_i}\\
\end{bmatrix}\, M = \begin{bmatrix}
    m_1\\
    m_2\\
    m_3\\
\end{bmatrix}
$$



- 其实我们要明白一些问题，也就是矩阵形状，这一点我们时时刻把握住，例如这里 $M$ 就是一个 $3 \times 4$ 的矩阵，有 11 个自由度，其中对于内参数 $c_x,c_y$ 是中心坐标偏移，$\theta$ 是两个坐标夹角以及 $\alpha,\beta$ 这样 5 内参数的自由度，而对于外参数这是 3 个旋转自由度和 3 个平移自由度，所以一共 11 个自由度，那么也就是有 11 未知量。
- 那么图像平面上的一对点 $u_i,v_i$ 可以提供两个方程，例如，$u_i = \frac{m_1p_i}{m_3p_i}$ 和 $v_i = \frac{m_2p_i}{m_3p_i}$，那么也就是通过 6 点就可以求解这个摄像机矩阵，但是在实际操作中使用多于 6 对点来获取更加鲁棒的结果。
- 这里每一 $m_i$ 都是一个1 行 4 列向量

$$
\begin{bmatrix}
    u_i\\
    v_i
\end{bmatrix} = \begin{bmatrix}
    \frac{m_1 p_i}{m_3 p_i}\\
    \frac{m_2 p_i}{m_3 p_i}\\
\end{bmatrix}
$$

上面公式也就是像平面上一对点 $u_i,v_i$ 对应于 $p_i$ 的表达式，
$$
u_i = \frac{m_1 p_i}{m_3 p_i} \rightarrow u_i(m_3 p_i) = m_1 p_i \rightarrow u_i(m_3 p_i) - m_1 p_i =0
$$

$$
v_i = \frac{m_2 p_i}{m_3 p_i} \rightarrow v_i(m_3 p_i) = m_1 p_i \rightarrow v_i(m_3 p_i) - m_2 p_i =0
$$

在下面这个方程组，$u_i,v_i,p_i$ 都是已知的，这里 $m_i$ 是一个 $1 \times 4$ 的向量，而 $p_i$ 则是 $4 \times 1$ 所以 $m_i$ 和 $p_i$ 点乘得到一个数。

$$
\begin{aligned}
    u_1(m_3 p_1) - m_1 p_1 =0\\
    v_1(m_3 p_1) - m_2 p_1 =0\\
    u_2(m_3 p_2) - m_1 p_2 =0\\
    v_2(m_3 p_2) - m_2 p_2 =0\\
    \vdots\\
    u_n(m_3 p_n) - m_1 p_n =0\\
    v_n(m_3 p_n) - m_2 p_n =0\\
\end{aligned}
$$


##### 标定问题

$$
Pm = 0
$$

这里用大写 P 表示矩阵，在这个矩阵中所有元素都是已知的，
$$
P = \begin{bmatrix}
    P_1^T & 0^T & -u_1P_1^T\\
    0^T & P_1^T & -v_1P_1^T\\
    \vdots\\
    P_n^T & 0^T & -u_1P_n^T\\
    0^T & P_n^T & -v_1P_n^T\\
\end{bmatrix}
$$


- $P$ 是一个 $2n \times 12$ 的矩阵

$$m = \begin{bmatrix}
    m_1^T\\
    m_2^T\\
    m_3^T\\
\end{bmatrix}$$

- $m$ 是一个 $12 \times 1$ 的列向量

#### 齐次线性方程组
- $M$ = 方程数 = $2n$
- $N$ = 未知参数 = $11$
当 $M > N$ 时，超定方程组(不少于 6 对点)
- 0 总是一个解，不存在非零解
- 目标：求解非零解 $min_{m}||Pm||,s.t. ||m||=1$，如果不加约束这里$m$ 会无限地小下去也就是没有最小值，所以才对 $m$ 添加约束。

$$Pm = 0$$
这里结论就是，$m$ 就是 P 矩阵最小奇异值的右奇异向量，且 $||m||=1$，拿到这个向量还需要将其转换为矩阵的形式，就得到了。

$$ m = \begin{bmatrix}
    m_1^T\\
    m_2^T\\
    m_3^T\\
\end{bmatrix} \rightarrow M = \begin{bmatrix}
    m_1\\
    m_2\\
    m_3\\
\end{bmatrix} = \begin{bmatrix}
    A & b
\end{bmatrix}$$



