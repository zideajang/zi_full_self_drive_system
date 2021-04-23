### 摄像机几乎
- 小孔成像
- 针孔摄像机
- 凸透镜

将胶片直接放置在物体前方，不过这样做法问题，所以在胶片和物体之间放上一个隔板，隔板的中间开上小孔，通过带有小孔的隔板就可以减少模糊。这就是针孔

<img src="./computer_vision_01/002.png">

- image plane 像平面
- 2D image 
- forcal length 焦距
- pinhole 小孔，光圈
- virtual image plane 虚拟像平面
- 3D object 真实物体
- 虚拟像平面和像平面除了方向不一致，其他完全都一样

<img src="./computer_vision_01/003.png">

$$\frac{y^{\prime}}{f} = \frac{y}{z} \rightarrow y^{\prime} = f\frac{y}{z}$$

$$\frac{x^{\prime}}{f} = \frac{x}{z} \rightarrow x^{\prime} = f\frac{x}{z}$$

利用三角形相似法制，

$$p = \begin{bmatrix}
    x\\
    y\\
    z
\end{bmatrix} \rightarrow p^{\prime} = \begin{bmatrix}
    x^{\prime}\\
    y^{\prime}\\
\end{bmatrix}$$

<img src="./computer_vision_01/007.jpeg">

- 小孔也叫做光圈，通过上图我们可以看出调整光圈的大小对成像的影响，光圈越小图像就越清晰，也就是胶片上一个点对应真实世界多个点的坐标。缩小光圈又会因为到达胶片上光线变少而图像变暗。


<img src="./computer_vision_01/008.jpg">

- 凸透镜将多条光线聚焦到胶片上，增加了照片的亮度
  
<img src="./computer_vision_01/009.jpeg">

- 凸透镜将光线聚焦到胶片上
- 所有平行于光轴的光线都会会聚到焦点，焦点到透镜中心点的距离称为**焦距**
- 穿过中心的光线的方向不发生改变

#### 近轴折射模型