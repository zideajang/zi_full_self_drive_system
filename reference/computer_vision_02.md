## 摄像机标定
- 所谓摄像机的标定，就是求解摄像机的内外参数。摄像机内、外参数描述了 3 维世界到 2 维像素的映射关系。

$$P^{\prime} = MP_w = K\begin{bmatrix}
    R & T
\end{bmatrix}P_w$$
- $K$ 为内参数
- $\begin{bmatrix}
    R & T
\end{bmatrix}$ 为外参数

那么摄像机标定的目标就是，通过 1 张图片或者多张图像来估算内外参数点，注意这里是估算

- 在世界坐标系中$p_1,p_2,\cdots$ 的位置已知，在标定装置中的世界坐标
- 图像中这些点 $p_1,p_2,\cdots$ 对应点的位置也是已知的
- 目的就是利用这些点来联立方程求解计算摄像机的内，外参数

$$p_i = \begin{bmatrix}
    u_i\\
    v_i
\end{bmatrix} = \begin{bmatrix}
    \frac{m_1p_i}{m_3p_i}\\
    \frac{m_2p_i}{m_3p_i}\\
\end{bmatrix}\, M = \begin{bmatrix}
    m_1\\
    m_2\\
    m_3\\
\end{bmatrix}$$

- 其实我们要明白一些问题，也就是矩阵形状，这一点我们时时刻把握住，例如这里 $M$ 就是一个 $3 \times 4$ 的矩阵，有 11 个自由度，其中对于内参数 $c_x,c_y$ 是中心坐标偏移，$\theta$ 是两个坐标夹角以及 $\alpha,\beta$ 这样 5 内参数的自由度，而对于外参数这是 3 个旋转自由度和 3 个平移自由度，所以一共 11 个自由度。
- 那么图像平面上的一对点$u_i,v_i$ 可以提供两个方程例如，$u_i = \frac{m_1p_i}{m_3p_i}$ 和 $v_i = \frac{m_2p_i}{m_3p_i}$，那么也就是通过 6 点就可以求解矩阵，但是在实际操作中使用多于 6 对点来获取更加鲁棒的结果。
- 这里每一 $m_i$ 都是一个1 行 4 列向量


$$ \begin{bmatrix}
    u_i\\
    v_i
\end{bmatrix} = \begin{bmatrix}
    \frac{m_1 p_i}{m_3 p3}\\
    \frac{m_2 p_i}{m_3 p3}\\
\end{bmatrix}$$

$$u_i = \frac{m_1 p_i}{m_3 p_i} \rightarrow u_i(m_3 p_i) = m_1 p_i \rightarrow u_i(m_3 p_i) - m_1 p_i =0$$
$$v_i = \frac{m_2 p_i}{m_3 p_i} \rightarrow v_i(m_3 p_i) = m_1 p_i \rightarrow v_i(m_3 p_i) - m_2 p_i =0$$

$$\begin{aligned}
    u_1(m_3 p_1) - m_1 p_1 =0\\
    v_1(m_3 p_1) - m_2 p_1 =0\\
    u_2(m_3 p_2) - m_1 p_2 =0\\
    v_2(m_3 p_2) - m_2 p_2 =0\\
    \vdots\\
    u_n(m_3 p_n) - m_1 p_n =0\\
    v_n(m_3 p_n) - m_2 p_n =0\\
\end{aligned}$$

##### 标定问题

$$Pm = 0$$

$$P = \begin{bmatrix}
    P_1^T & 0^T & -u_1P_1^T\\
    0^T & P_1^T & -v_1P_1^T\\
    \vdots\\
    P_n^T & 0^T & -u_1P_n^T\\
    0^T & P_n^T & -v_1P_n^T\\
\end{bmatrix}$$

- $P$ 是一个 $2n \times 12$ 的矩阵

$$m = \begin{bmatrix}
    m_1^T\\
    m_2^T\\
    m_3^T\\
\end{bmatrix}$$

- $m$ 是一个 $12 \times 1$ 的列向量

#### 齐次线性方程组