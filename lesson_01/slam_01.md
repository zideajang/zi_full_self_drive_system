

特斯拉是通过视觉来感知周围，搜集信息，然后根据信息给出决策。



正式开始编写无人驾驶的代码，作为程序员有时候只有看到 code 才踏实。现在数据是手头有一段行车录像，是由车载前置摄像头记录下来的一段道路上车辆行驶记录，路况很简单就看见一辆从对面行驶过来车辆。

接下来我们的一个大目标就是基于这段行车视频，通过 SLAM 技术来绘制出点云来还原真实场景同时实时记录下摄像机位置。知道提到无人驾驶就少不了 SLAM 技术，SLAM 技术是将现有技术综合来实现机器人通过在位置环境下运动来实现导航同时绘制地图。这里重点是同时，因为 SLAM 技术一次就是把定位和地图绘制两件事做了，现在比较火是 ORB SLAM，有关 SLAM 技术的大概思路以及会用的各个技术点我们在 coding 过程中会一一个逐一介绍，以避免开始就大量理论，从而让分享显得乏味，也希望和大家一起互动，希望大家多多留言，有些问题还要请教大家，我是只是牵一个头。

要做这一切，我们需要做一些准备，同时也看看这个分享是否适合你，首先我们只需要简单了解 linux，这里准备在 ubuntu 16 这个版本上运行代码，其他平台当然也可以，现在我也是在 Mac 上开发。在系统上需要安装 python 3.6 以上版本，OpenCV 不建议用太新版本，建议用 3.4.1 版本。显示这里用了 sdl2 ，你也可以根据自己喜好用 pygame 或者 opencv 显示。

#### 显示行车记录

接下来我们第一件要做的是就是将视频显示出来，有关 test.mp4 大家可以上下载或者  , 然后用 OpenCV 读取视频流

```Python
if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        # frame = cv2.resize(frame,(W,H),cv2.INTER_AREA)
        if ret == True:
            process_frame(frame)
        else:
            break
```



上面的代码功能是 Opencv 读取视频流，然后将读取到每一帧图像交给 frame 进行处理，仅此而已，如果大家有关 Opencv 还不熟悉可以参考我分享[从基础到实践 OpenCV (2)—读取视频流](https://juejin.cn/post/6973888174061781022)。接下来我们定义 Display 类用于显示图像，具体代码如下



```python
class Display(object):
    def __init__(self,W, H):
        super().__init__()
        sdl2.ext.init()

        self.W = W
        self.H = H

        self.window = sdl2.ext.Window("The Slam Video",size=(W,H),position=(100,100))
        self.window.show()


    def paint(self,img):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:,:,0:3] = img.swapaxes(0,1)
        self.window.refresh()
```


```python
sdl2.ext.init()
```
然后定义窗口,为窗口指定名称，指定窗口大小以及窗口在屏幕上位置，通过指定窗口左上角距离屏幕左上角的距离来指定位置。

```python
self.window = sdl2.ext.Window("The Slam Video",size=(W,H),position=(100,100))
self.window.show()
```
现在对上面代码给予简单解释，定义 Display 类，在初始化方法中，我们指定要显示画面宽和高。然后通过 ext 提供 Window 实例化一个窗口，给出窗口一些基本设定、例如标题、窗口的大小以及窗口左上角出现位置。
```python
events = sdl2.ext.get_events()
for event in events:
    if event.type == sdl2.SDL_QUIT:
        exit(0)
```

get event 方法会返回一个事件，我们根据事件类型给出对应操作，当遇到是退出事件时，这做退出应用的处理。

```python
surf = sdl2.ext.pixels3d(self.window.get_surface())
surf[:,:,0:3] = img.swapaxes(0,1)
```
接下就是具体往定义好的窗口添加显示内容，调用 window 的 get surface 来拿到窗口的画布，接下来就是将读取帧图像绘制到画布上，pixels3d 将画布格式定义宽、高以及通道的三维矩阵，最后一个维度 RGBA 包括透明通道。但是我们读取通道并不包含透明通道所以只取其前 3 个通道后将图片填充到这个矩阵，swapaxes 用于调整维度，对于矩阵操作有时候需要旋转维度，二维转置直接.T较为简单，如果是高维的话，需要借助 swapaxes 调整维度。


```python
import cv2
import time,sys

import pygame
from pygame.locals import *


if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")
    FPS = int(round(cap.get(cv2.CAP_PROP_FPS)))
    FramePerSec = pygame.time.Clock()

    Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_ratio = 2

    pygame.init()
    pygame.display.set_caption('OpenCV Video')
    screen =  pygame.display.set_mode((Width//scale_ratio,Height//scale_ratio),0,32)

    screen.fill([0,0,0])
    num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if num == 0:
            T0 = time.time()
        if time.time()-T0 > num*(1./FPS):
            ret, frame = cap.read()

            frame = cv2.resize(frame,(Width//2,Height//2),interpolation = cv2.INTER_AREA)
            TimeStamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ret == False:
                print('Total Time:', time.time()-T0)
                pygame.quit()
                sys.exit()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.transpose(frame)
            frame = pygame.surfarray.make_surface(frame)
            screen.blit(frame, (0,0))
            
            pygame.display.update()
            
            num += 1
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()

```