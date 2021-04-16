import cv2
import sdl2
import sdl2.ext


class Display(object):

    def __init__(self,H,W):
        pass

    def show(self,frame):
        print(frame.shape)
        pass
if __name__ == "__main__":
    print("hello ")

    cap = cv2.VideoCapture("clip/countryroad.mp4")
    display = Display(10,10)

    while cap.isOpened():
        ret,frame = cap.read()
        if ret == True:
            display.show(frame)
        else:
            break