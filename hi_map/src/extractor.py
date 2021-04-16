import cv2
import numpy as np

class FeatureExtractor:

    def __init__(self):

        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher()

        self.last = None

        self.kp_count = 3000
        self.quality_level = 0.01
        self.min_distance = 3
    def extract(self,frame):
        # 转换图像数据格式
        frame = np.mean(frame,axis=2).astype(np.uint8)
        # 检测关键点
        feats = cv2.goodFeaturesToTrack(frame,self.kp_count,qualityLevel=self.quality_level,minDistance=self.min_distance)
        
        # 提取关键点和描述
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in feats]
        kps, des = self.orb.compute(frame,kps)

        # 匹配
        matches = None
        if self.last is not None:
            matches = self.bf.match(des,self.last['des'])
            print(matches)

        self.last = {'kps':kps,'des':des}

        return kps,des,matches