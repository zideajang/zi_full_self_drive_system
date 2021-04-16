import time
import cv2

import numpy as np

from hi_map.src.display import Display
from hi_map.src.extractor import FeatureExtractor

W = 1920 // 2
H = 1080 //2 

display = Display(W,H)

fe = FeatureExtractor()

def process_frame(frame):
    frame = cv2.resize(frame,(W,H))
    # kp,des = orb.detectAndCompute(frame,None)
    kp,des,_ = fe.extract(frame)
    for p in kp:
        u,v = map(lambda x:int(round(x)),p.pt)
        # u,v = map(lambda x:int(round(x)),p[0])
        cv2.circle(frame,(u,v),color=(0,255,0),radius=3)
    display.show(frame)


if __name__  == "__main__":
    cap = cv2.VideoCapture("clip/countryroad.mp4")

    while cap.isOpened():
        ret,frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break