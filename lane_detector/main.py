import cv2
import pickle
import time

from utils import thresh_pipeline

class LaneDetector:
    def __init__(self):
        self.mtx = None
        self.dist = None
        calibration_pickle = pickle.load( open( "./video/calibration_pickle.p", "rb" ) )
        self.mtx = calibration_pickle["mtx"]
        self.dist = calibration_pickle["dist"]
        # print(self.mtx)

    def detect(self,frame):
        # pass
        res = cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)
        img_thresh = img_thresh = thresh_pipeline(res, 
                                          gradx_thresh=(25,255), 
                                          grady_thresh=(10,255), 
                                          s_thresh=(100, 255), 
                                          v_thresh=(0, 255))
        return img_thresh

class Display:
    def __init__(self):
        self.cap = cv2.VideoCapture("video/project_video.mp4")
        self.img_width = int(self.cap.get(3))
        self.img_height = int(self.cap.get(4))

        self.screenshot_dir = "screenshot"

        self.lane_detector = LaneDetector()

    def show(self):
        while self.cap.isOpened():
            ret,frame = self.cap.read()
            if ret:
                
                frame = self.lane_detector.detect(frame)
                cv2.imshow("frame",frame)

                k = cv2.waitKey(1)
                if k == ord('q'):
                    break
                if k == ord('s'):
                    cv2.imwrite(f"{self.screenshot_dir}/{time.time()}.jpg",frame)
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    print("hello lane detector...")

    source_path = "video/project_video.mp4"


    # lane_detector = LaneDetector()
    display = Display()
    display.show()
    