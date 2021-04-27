import cv2
import sdl2
import sdl2.ext

import numpy as np

"""
python 3.6
opencv

"""

# weight, height of video
H = 1080 // 2
W = 1920 // 2



# display video 
class Display(object):
    def __init__(self,W,H):
        sdl2.ext.init()

        self.W = W
        self.H = H

        self.window = sdl2.ext.Window("SLAM",size=(self.W,self.H),position=(-500,-500))
        self.window.show()

    def paint(self,frame):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)
        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:,:,0:3] = frame.swapaxes(0,1)
        self.window.refresh()
display = Display(W,H)

class FeatureExtractor(object):
    def __init__(self):

        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)


        self.last = None

        # 定义检测角点的最大值
        self.kp_count = 3000
        # 检测特征点的质量等级
        self.quality_level = 0.01
        # 两个角点间最小距离
        self.min_distance = 2
    def extract(self,frame):
        # print("extracting...")
        # print(frame.shape)
        # detect and computer


        # BGR opencv 
        frame = np.mean(frame,axis=2).astype(np.uint8)

        """
        extract features
        compute description for features
        match descripiont of featurs btw two frame 
        """

        feats = cv2.goodFeaturesToTrack(
            frame,
            self.kp_count,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance)

        # [ x=f[0][0],y=f[0][1],_size=20 for f in feats]
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in feats]
        kps, des = self.orb.compute(frame,kps)        

        ret = []

        if self.last is not None:
            matches = matches = self.bf.knnMatch(des,self.last['des'],k=2)
            # print(type(matches))
            for m,_ in matches:
                kp1 = kps[m.queryIdx].pt
                kp2 = self.last['kps'][m.trainIdx].pt
                ret.append((kp1,kp2))
        # keyPoint()

        # print(feats.shape)
        # kp1, des1 = self.orb.detectAndCompute(frame,None)
        # KeyPoint 0x136c1c240
        # return kp1

        self.last = {'kps':kps,'des':des}

        return ret
fe = FeatureExtractor()


def process_frame(frame):
    # resize 
    resized_frame = cv2.resize(frame,(W,H))
    # feature extrat,
    
    matches = fe.extract(resized_frame)


    # print(kps)
    
    for pt1,pt2 in matches:
        u1,v1 = map(lambda x:int(round(x)),pt1)
        u2,v2 = map(lambda x:int(round(x)),pt2)
        # uv = int(round(kp.pt[0])),int(round(kp.pt[1]))
        cv2.circle(resized_frame,(u1,v1),color=(0,255,0),radius=3)
        cv2.circle(resized_frame,(u2,v2),color=(0,255,255),radius=3)

    display.paint(resized_frame)

if __name__ == "__main__":

    
    # print("hello slam")
    cap = cv2.VideoCapture("clip/countryroad.mp4")
    while cap.isOpened():
        # (1080, 1920, 3)
        ret, frame = cap.read()
        if ret:
            process_frame(frame)

