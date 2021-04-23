import time
import cv2

import numpy as np

from hi_map.src.display import Display
from hi_map.src.extractor import FeatureExtractor

F = 1

W = 1920 // 2
H = 1080 //2 

display = Display(W,H)

K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

fe = FeatureExtractor(K)


def process_frame(frame):
    frame = cv2.resize(frame,(W,H))
    # kp,des = orb.detectAndCompute(frame,None)
    # kp,des,_ = fe.extract(frame)
    matches = fe.extract(frame)
    # print(f"{len(matches)}")
    if matches is not None:
        
        for pt1,pt2 in matches:

            u1,v1 = fe.denormalize(pt1)
            u2,v2 = fe.denormalize(pt2)

            # u,v = map(lambda x:int(round(x)),p.pt)
            # u1,v1 = map(lambda x:int(round(x)),pt1)
            # u2,v2 = map(lambda x:int(round(x)),pt2)
            cv2.circle(frame,(u1,v1),color=(0,255,0),radius=3)
            cv2.line(frame,(u1,v1),(u2,v2),color=(255,0,0))
    display.show(frame)


if __name__  == "__main__":
    cap = cv2.VideoCapture("clip/countryroad.mp4")

    while cap.isOpened():
        ret,frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
    
    # cap.release()
    # cv2.destroyAllWindows()