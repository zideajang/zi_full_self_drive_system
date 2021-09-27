import cv2
import time
import numpy as np


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 计算方向上梯度
    
    # 1) 将图片转化为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) 对图片在 x 或者 y
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobelxy = np.sqrt(sobelx**2+sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    return mag_binary

def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    gradient_direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(gradient_direction)
    dir_binary[(gradient_direction>=thresh[0])&(gradient_direction<=thresh[1])] = 1
    
    return dir_binary

def color_thresh(img, s_thresh=(0,255), v_thresh=(0,255)):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel>=s_thresh[0]) & (s_channel<=s_thresh[1])] = 1
    
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel>=v_thresh[0]) & (v_channel<=v_thresh[1])] = 1   
    
    c_binary = np.zeros_like(s_channel)
    c_binary[(s_binary==1) & (v_binary==1)] = 1
    
    return c_binary
    

def thresh_pipeline(img, gradx_thresh=(0,255), grady_thresh=(0,255), s_thresh=(0, 255), v_thresh=(0, 255)):
    
    gradx = abs_sobel_thresh(img, orient='x', thresh=gradx_thresh)
    grady = abs_sobel_thresh(img, orient='y', thresh=grady_thresh)
    c_binary = color_thresh(img, s_thresh=s_thresh, v_thresh=v_thresh)
    
    
    # Combine these thresholded binary images
    thresh_binary = np.zeros_like(img[:,:,0])
    thresh_binary[(gradx==1) & (grady==1) | (c_binary==1)] = 255
#     thresh_binary[c_binary==1] = 255
#     thresh_binary[(sxbinary==1)|(s_binary==1)] = 1 
    
    return thresh_binary
def warper(img, src, dst):

    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def perspective_transformer():
    pass
if __name__  == "__main__":

    src = np.float32([
        [678, 440],[1279, 720],
        [0, 720],[595, 440]])

    # Four desired coordinates
    dst = np.float32([
        [1180, 0],[1180, 720],
        [100, 720],[100, 0]])

    screenshot_dir = "screenshot"

    cap = cv2.VideoCapture("video/project_video.mp4")

    img_width = int(cap.get(3))
    img_height = int(cap.get(4))

    is_screen = True

    while cap.isOpened():
        ret,frame = cap.read()
        if ret == True:
            

            # frame =cv2.resize(frame,(img_width//2,img_height//2))
            # wrap = warper(frame,src//2,dst//2)

            # vis = np.concatenate((frame, wrap), axis=1)
            # for pnt in [(678,440),(1279, 720),(0, 720),(595, 440)]:
            #     frame = cv2.circle(frame,pnt,5,(0,0,255),5)

            # frame = cv2.line(frame,(0, 720),(595, 440),(0,255,0),3)
            # frame = cv2.line(frame,(595, 440),(678,440),(0,255,0),3)
            # frame = cv2.line(frame,(678,440),(1279, 720),(0,255,0),3)

            frame = color_thresh(frame)

            cv2.imshow("frame",frame)
            # process_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if is_screen:
                cv2.imwrite(f"{screenshot_dir}/{time.time()}.jpg",frame)
                is_screen = False
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()