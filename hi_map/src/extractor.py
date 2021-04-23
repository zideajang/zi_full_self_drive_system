import cv2
import numpy as np

# 
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

# return [[x,y]]->[[x,y,1]]
def add_ones(x):
    return np.concatenate([x,np.ones((x.shape[0],1))],axis=1)

class FeatureExtractor:

    def __init__(self,K):

        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.last = None

        # 定义检测角点的最大值
        self.kp_count = 3000
        # 检测特征点的质量等级
        self.quality_level = 0.01
        # 两个角点间最小距离
        self.min_distance = 2

        self.K = K
        self.Kinv = np.linalg.inv(self.K)
    
    def denormalize(self,pt):
        # return int(round(pt[0]+frame.shape[0]/2)),int(round(pt[1] + frame.shape[1]/2))
        ret = np.dot(self.K,np.array([pt[0],pt[1],1.0]).T)
        print(ret)
        return int(round(ret[0])),int(round(ret[1]))

    
    def extract(self,frame):
        # 转换图像数据格式
        frame = np.mean(frame,axis=2).astype(np.uint8)
        # 检测关键点
        feats = cv2.goodFeaturesToTrack(frame,self.kp_count,qualityLevel=self.quality_level,minDistance=self.min_distance)
        
        # 提取关键点和描述
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=20) for f in feats]
        
        # 计算特征点
        kps, des = self.orb.compute(frame,kps)

        # 匹配
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des,self.last['des'],k=2)
            for m,n in matches:
                if m.distance < 0.75* n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt

                    ret.append ((kp1,kp2))


        # filter
        if len(ret) > 0:
            ret = np.array(ret)
            ret[:,0,:] = np.dot(self.Kinv,add_ones(ret[:,0,:]).T).T[:,0:2]
            ret[:,1,:] = np.dot(self.Kinv,add_ones(ret[:,1,:]).T).T[:,0:2]
            # ret[:,:,0] -= frame.shape[0]//2
            # ret[:,:,1] -= frame.shape[1]//2
            model, inliers = ransac((ret[:,0],ret[:,1]),
                                    FundamentalMatrixTransform,#
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)
            # print(sum(inliers))
            ret = ret[inliers]

            s,v,d = np.linalg.svd(model.params)
            # print(v)
        self.last = {'kps':kps,'des':des}

        return ret