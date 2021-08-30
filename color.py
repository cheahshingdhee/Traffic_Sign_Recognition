
import cv2
import numpy as np
import time
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt


def getEmptyPic(height, width):
    return np.zeros((height,width,1), np.uint8)

def red_blue_normalize(height,width,img):
    results=getEmptyPic(height,width)
    for x in range(height):
        for y in range(width):
            r=img[x,y,2]
            g=img[x,y,1]
            b=img[x,y,0]
            r = int(r)
            g = int(g)
            b = int(b)
            red = r/(r+b+(g*2)+4)
            blue = b/(r+b+(g*2)+4)
            if red>blue :
                results[x,y]=red*255+10
            else:
                results[x, y] = blue*255+10
    return results

def empty(a):
    pass

print("done")

def MSERs(pic):
    mser = cv2.MSER_create(delta=12, max_variation=0.20)
    vis = getEmptyPic(height, width)
    regions, boundingBoxes = mser.detectRegions(pic)
    reglen = len(regions)
    max3 = 0
    index3 = 0
    print("regions ", reglen)
    # find max regions
    for x in range(reglen):
        size = regions[x].shape[0] * regions[x].shape[1]
        # print("MSER ", size)
        if size <= 0.95 * height * width:
            if max3 < size:
                max3 = size
                index3 = x

    arr = []
    num = 0
    # print("index 3",index3)

    # draw bouding box
    for box in boundingBoxes:
        x, y, w, h = box;
        if w * h <= 0.7 * height * width:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)
            temp = np.zeros((height, width, 1), np.uint8)
            temp = img[y:y + h, x:x + w]
            arr.append(temp.copy())
            # cv2.imshow('MSErs', temp)
            # cv2.waitKey(0)

    return vis

def Countour(imgCanny,height,width,img_cpy):
    contours ,hierarchy= cv2.findContours(imgCanny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw=getEmptyPic(height,width)
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        print(area)
        print(len(approx))
        if ((len(approx) >= 3) & (len(approx) <= 20) & (area > 90)):
            x,y,w,h=cv2.boundingRect(approx)
            if((w)*(h)>=1024):
                cv2.rectangle(img_cpy, (x - 5, y - 5), (x + w + 10, y + h + 10), (0, 255, 0), 2)
                contour_list.append(contour)

    print(len(contour_list))
    cv2.drawContours(draw, contour_list, contourIdx=-1 ,color=(255, 255, 255), thickness=1)

    return draw,img_cpy


def colorCLAHE(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(8, 8))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img
def r_normalize(img):
    img = img.astype(np.float32)
    deno = np.sum(img, axis=2)
    deno += 10e-8
    mole = np.minimum(img[..., 2]-img[..., 0], img[..., 2]-img[..., 1])
    C = np.maximum(0, mole/deno)
    return (C*255).astype(np.uint8)


def b_normalize(img):
    img = img.astype(np.float32)
    deno = np.sum(img, axis=2)
    deno += 10e-8
    mole = img[..., 0] - img[..., 2]
    C = np.maximum(0, mole/deno)
    return (C*255).astype(np.uint8)


cap=cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,500)


#while True:
for x in range(100):
    x=x+59
    img=cv2.imread("Img/t{:d}.png".format(x+1))
    #img_r = r_normalize(img)
    #img_b = b_normalize(img)
    img=cv2.resize(img,(500,500))
    #success,img = cap.read()
    img_cpy=img.copy()
    cv2.imshow("Original image", img)
    height, width, channels = img.shape
    results = getEmptyPic(height, width)  # to store the results of red/blue-normalization
    img = colorCLAHE(img)
    img=cv2.GaussianBlur(img,(13,13),sigmaX=1,sigmaY=1)
    #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    cv2.imshow("HSV results", img)

    results = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_g = np.array([36, 25, 25])
    upper_g = np.array([86, 255, 255])
    mask5 = cv2.inRange(results, lower_g, upper_g)
    mask5 = cv2.bitwise_not(mask5)
    #cv2.imshow("Mask5", mask5)
    #red color segmentation
    lower_r =np.array([165,130,50])
    upper_r= np.array([179,255,255])
    mask=cv2.inRange(results,lower_r,upper_r)
    #cv2.imshow("HSV",results)
    mask=mask&mask5

    cv2.imshow("Mask1",mask)

    lower_r2 = np.array([0, 130, 50])
    upper_r2 = np.array([10, 255, 255])
    mask1 = cv2.inRange(results, lower_r2, upper_r2)

    mask1=mask1&mask5

    cv2.imshow("Mask2", mask1)

    #blue color segmentation
    lower_b = np.array([100, 100, 20])
    upper_b = np.array([140, 255, 255])
    mask2 = cv2.inRange(results, lower_b, upper_b)
    mask2=mask2&mask5

    cv2.imshow("Mask3", mask2)

    #yellow color segmentation
    #lower_y = np.array([20, 100, 20])
    #upper_y = np.array([40, 255, 255])
    lower_y = np.array([15, 100, 50])
    upper_y = np.array([30, 255, 255])
    mask4 = cv2.inRange(results, lower_y, upper_y)
    mask4=mask4&mask5
    cv2.imshow("Mask4", mask4)



    # canny edge
    red_thrs=cv2.bitwise_or(mask,mask1)
    red_thres=cv2.Canny(red_thrs,200,300, L2gradient = True)
    red_countour,img_cpy=Countour(red_thrs,height,width,img_cpy)

    ##################################################

    cv2.imshow("red-contour",red_countour)
    blue_thres=cv2.Canny(mask2,200,300,L2gradient=True)
    blue_countour,img_cpy=Countour(blue_thres,height,width,img_cpy)
    #cv2.imshow("blue-canny",blue_thres)
    cv2.imshow("blue-contour",blue_countour)
    yellow_thres = cv2.Canny(mask4, 200, 300, L2gradient=True)
    yellow_countour,img_cpy=Countour(yellow_thres,height,width,img_cpy)
    cv2.imshow("yellow-contour",yellow_countour)
    #cv2.imshow("yellow-canny", yellow_thres)

    Final_2=cv2.bitwise_or(mask2,mask4)
    Final=cv2.bitwise_or(red_thrs,Final_2)
    x=cv2.Canny(Final,200,300,apertureSize=7, L2gradient = True)
    cv2.imshow("Combine",x)
    cv2.imshow("Combine2",img_cpy)


    MSER_red = MSERs(red_thrs)
    MSER_blue = MSERs(mask2)
    MSER_yellow =MSERs(mask4)
    #cv2.imshow('MSER-red', MSER_red)
    #cv2.imshow('MSER-blue', MSER_blue)
    #cv2.imshow('MSER-yellow', MSER_yellow)

    #if cv2.waitKey(1) & 0xFF==ord('q'):
    #    break
    cv2.waitKey(0)
    cv2.destroyAllWindows()



