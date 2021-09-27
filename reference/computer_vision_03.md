### 提取摄像机参数

$$P^{\prime} = MP_w = K \begin{bmatrix}
    R & T
\end{bmatrix}P_w$$

在世界坐标系下的点 $P_w$ 经过一个投影矩阵 $M$ 就可以得到像素坐标下的点 $P^{\prime}$ 
接下来我们通过 $M$ 来推导摄像机的内外参数

这里 R 为 3 x 3 旋转矩阵而 T 为 3 x 1 的平移矩阵 K 表示摄像机的内参数矩阵 [R T] 表示

$$K = \begin{bmatrix}
   \alpha & - \alpha \cos \theta & u_0\\ 
   0 & \frac{\beta}{\sin \theta}  & v_0\\ 
   0 & 0 & 1\\ 
\end{bmatrix}\, R = \begin{bmatrix}
    r_1^T\\
    r_2^T\\
    r_3^T\\
\end{bmatrix} \, T = \begin{bmatrix}
    t_x\\
    t_y\\
    t_z\\
\end{bmatrix}$$

$$M = \left( \begin{matrix}
    \alpha r_1^T - \alpha \cos \theta r_2^T + u_0r_3^T &  \alpha t_x - \alpha \cos \theta t_y + u_0t_z\\
     \frac{\beta}{\sin \theta} r_2^T + v_0r_3^T &  \frac{\beta}{\sin \theta} t_y + v_0t_z\\
     r_3^T & t_z
\end{matrix} \right)$$

$$\rho \begin{bmatrix}
    A&b
\end{bmatrix} = K \begin{bmatrix}
    R&T
\end{bmatrix}$$

$$\rho A = \rho \left( \begin{matrix}
    a_1^T\\
    a_2^T\\
    a_3^T
\end{matrix} \right) =  \left( \begin{matrix}
     \alpha r_1^T - \alpha \cos \theta r_2^T\\
     \frac{\beta}{\sin \theta} r_2^T + v_0r_3^T\\
     r_3^T 
\end{matrix} \right) = KR$$

$$\rho = \frac{1}{|a_3|}$$

$$ \begin{aligned}
    \rho^2(a_1a_3) = u_0\\
    \rho^2(a_2a_3) = v_0
\end{aligned}$$
$$\begin{aligned}
    \rho^2(a_1 \times a_3) = \alpha r_2 - \alpha \cos \theta r_1\\
    \rho^2 (a_2 \times a_3) = \frac{\beta}{\sin \theta} r_1
\end{aligned}$$

$$\begin{aligned}
    
\end{aligned}$$