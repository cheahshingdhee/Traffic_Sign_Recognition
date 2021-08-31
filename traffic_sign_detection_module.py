
import cv2
import numpy as np
import time
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from scipy.special import softmax



def getEmptyPic(height, width):
    return np.zeros((height,width,1), np.uint8)

def empty(a):
    pass

print("done")

def HOG(img):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    return hog_image,fd


def grayscale(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def CLAHE(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs = clahe.apply(img)
    return imgs

def preprocessing(img):
    img =grayscale(img)
    img=CLAHE(img)
    img=img/255
    return img

def Predict_CNN(img,x,model):
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(img.shape[0], img.shape[1], 1)
    test = [img]
    test = np.array(test)
    probabilities = np.amax(model.predict(test))
    class_predicted=np.argmax(model.predict(test))
    name=""
    if (probabilities >= 0.95):
        if (class_predicted == 0):
            name = "cross"
        elif (class_predicted == 1):
            name = "60speed"
        elif (class_predicted == 2):
            name = "no entry"
        elif (class_predicted == 3):
            name = "give way"
        elif (class_predicted == 4):
            name = "stop"
        elif (class_predicted == 5):
            name = "straight or left"
        elif (class_predicted == 6):
            name = "roundabout"
        elif (class_predicted == 7):
            name = "traffic light"
        elif (class_predicted == 8):
            name = "negative"
    return name,probabilities*100

def Countour(imgCanny,height,width,img_cpy,model):
    contours ,hierarchy= cv2.findContours(imgCanny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw=getEmptyPic(height,width)
    contour_list = []
    x=0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) >= 3) & (len(approx) <= 20) & (area > 90)):
            x,y,w,h=cv2.boundingRect(approx)
            if((w)*(h)>=1024):
                pointx=x
                pointy=y
                if x-30>0:
                    pointx=x-30
                if y-30>0:
                    pointy=y-30
                crop_image = img_cpy[pointy:y + h + 20, pointx:x + w + 20]
                class_predicted, prob = Predict_CNN(crop_image, x,model)
                if class_predicted!="negative" and class_predicted!="":
                    cv2.rectangle(img_cpy, (pointx, pointy), (x + w + 30, y + h + 30), (0, 255, 0), 2)
                    cv2.putText(img_cpy,"class= {:s}".format(class_predicted),((x+(w//2)),(y+(h//2))),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255))

                contour_list.append(contour)
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





cap=cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,500)


while True:
    from keras.models import load_model
    model = load_model('mymodel4.h5')
    success,img = cap.read()
    img_cpy=img.copy()
    height, width, channels = img.shape
    results = getEmptyPic(height, width)  # to store the results of red/blue-normalization
    img = colorCLAHE(img)
    img=cv2.GaussianBlur(img,(13,13),sigmaX=1,sigmaY=1)


    ######################## color segmentation ############################
    results = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    ##################### green color segmentation ####################

    lower_g = np.array([36, 25, 25])
    upper_g = np.array([86, 255, 255])
    mask5 = cv2.inRange(results, lower_g, upper_g)
    mask5 = cv2.bitwise_not(mask5)

    ##################### red color segmentation ####################

    lower_r =np.array([165,130,50])
    upper_r= np.array([179,255,255])
    mask=cv2.inRange(results,lower_r,upper_r)
    mask=mask&mask5
    lower_r2 = np.array([0, 130, 50])
    upper_r2 = np.array([10, 255, 255])
    mask1 = cv2.inRange(results, lower_r2, upper_r2)
    mask1=mask1&mask5
    red_thrs = cv2.bitwise_or(mask, mask1)
    cv2.imshow("Mask_red", red_thrs)

    ##################### blue color segmentation ####################
    lower_b = np.array([100, 100, 20])
    upper_b = np.array([140, 255, 255])
    mask2 = cv2.inRange(results, lower_b, upper_b)
    mask2=mask2&mask5

    cv2.imshow("Mask_blue", mask2)

    ##################### yellow color segmentation ####################
    lower_y = np.array([15, 100, 50])
    upper_y = np.array([30, 255, 255])
    mask4 = cv2.inRange(results, lower_y, upper_y)
    mask4=mask4&mask5
    cv2.imshow("Mask_yellow", mask4)



    ############################## canny edge and draw contour on the mask for red/blue/yellow ###########################

    #################### red #########################

    red_thres=cv2.Canny(red_thrs,200,300, L2gradient = True)
    red_countour,img_cpy=Countour(red_thrs,height,width,img_cpy,model)
    #cv2.imshow("red-contour",red_countour)

    #################### blue #########################

    blue_thres=cv2.Canny(mask2,200,300,L2gradient=True)
    blue_countour,img_cpy=Countour(blue_thres,height,width,img_cpy,model)
    #cv2.imshow("blue-canny",blue_thres)
    #cv2.imshow("blue-contour",blue_countour)

    ##################### yellow ##########################

    yellow_thres = cv2.Canny(mask4, 200, 300, L2gradient=True)
    yellow_countour,img_cpy=Countour(yellow_thres,height,width,img_cpy,model)
    #cv2.imshow("yellow-contour",yellow_countour)
    #cv2.imshow("yellow-canny", yellow_thres)

    ###################### combine #######################


    Final_2=cv2.bitwise_or(mask2,mask4)
    Final=cv2.bitwise_or(red_thrs,Final_2)
    x=cv2.Canny(Final,200,300,apertureSize=7, L2gradient = True)
    #cv2.imshow("Combine",x)
    cv2.imshow("Combine2",img_cpy)




    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



