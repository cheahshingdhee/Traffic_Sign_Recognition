import cv2
import numpy as np
import glob
from keras.models import load_model

from utility import preprocess, get_empty_image, get_class_label, predict_image

model = load_model('model/CNN_model.h5')

def getEmptyPic(height, width):
    return np.zeros((height,width,1), np.uint8)





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
    name="Invalid type"
    if (probabilities >= 0.80):
        if (class_predicted == 0):
            name = "non-stopping"
        elif (class_predicted == 1):
            name = "60 speed limit"
        elif (class_predicted == 2):
            name = "no entry"
        elif (class_predicted == 3):
            name = "give way"
        elif (class_predicted == 4):
            name = "stop"
        elif (class_predicted == 5):
            name = "straight or right"
        elif (class_predicted == 6):
            name = "roundabout"
        elif (class_predicted == 7):
            name = "traffic light"
        elif (class_predicted == 8):
            name = "Invalid type"
    return name,probabilities*100


def Countour(imgCanny,height,width,img_cpy):
    contours ,hierarchy= cv2.findContours(imgCanny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw=getEmptyPic(height,width)
    contour_list = []
    x=0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) >= 3) & (len(approx) <= 20) & (area>90)):
            x,y,w,h=cv2.boundingRect(approx)
            if((w)*(h)>=1024):
                pointx=x
                pointy=y
                if x-20>0:
                    pointx=x-20
                if y-20>0:
                    pointy=y-20
                crop_image = img_cpy[pointy:y + h + 20, pointx:x + w + 20]
                class_predicted,prob=Predict_CNN(crop_image,x,model)
                if class_predicted != "":
                    cv2.rectangle(img_cpy, (pointx, pointy), (x + w + 30, y + h + 30), (0, 255, 0), 2)
                    cv2.putText(img_cpy, "{:s}".format(class_predicted), ((x), (y + (h // 2))),
                                cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 0, 255),2)
                x=x+1
                contour_list.append(contour)
    cv2.drawContours(draw, contour_list, contourIdx=-1 ,color=(255, 255, 255), thickness=1)
    return draw,img_cpy


def colorCLAHE(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(8, 8))

    lab_planes[0] = clahe.apply(lab_planes[0])

    #lab_planes[0] = cv2.equalizeHist(lab_planes[0])

    lab = cv2.merge(lab_planes)

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img;


def Traffic_Detection(img,img_cpy):

    img=cv2.resize(img,(500,500))
    img_cpy=img.copy()
    cpy=img.copy()

    #cv2.imshow("Original image", img)
    height, width, channels = img.shape


    results = getEmptyPic(height, width)
    ############### COLOR CLAHE ###################
    img = colorCLAHE(img)

    #cv2.imshow("CLAHE", img)

    ############### Noise Removal ###################

    img=cv2.GaussianBlur(img,(13,13),sigmaX=1,sigmaY=1)
    #cv2.imshow("blur", img)



    ############### HSV segmentation ###################

    results = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #green color segmentation
    lower_g = np.array([36, 25, 25])
    upper_g = np.array([86, 255, 255])
    mask5 = cv2.inRange(results, lower_g, upper_g)
    mask5 = cv2.bitwise_not(mask5)
    #cv2.imshow("Mask5", mask5)

    #red color segmentation - upper bound and lower bound
    lower_r =np.array([165,130,50])
    upper_r= np.array([179,255,255])
    mask=cv2.inRange(results,lower_r,upper_r)

    lower_r2 = np.array([0, 130, 50])
    upper_r2 = np.array([10, 255, 255])
    mask1 = cv2.inRange(results, lower_r2, upper_r2)

    mask=mask&mask5
    mask1=mask1&mask5

    #cv2.imshow("Mask2", mask1)

    #blue color segmentation
    lower_b = np.array([100, 100, 20])
    upper_b = np.array([140, 255, 255])
    mask2 = cv2.inRange(results, lower_b, upper_b)
    #cv2.imshow("Mask3-b4", mask2)

    mask2=mask2&mask5

    #cv2.imshow("Mask3", mask2)

    #yellow color segmentation


    lower_y = np.array([15, 100, 50])
    upper_y = np.array([30, 255, 255])
    mask4 = cv2.inRange(results, lower_y, upper_y)
    #cv2.imshow("Mask4-b4", mask4)

    mask4=cv2.bitwise_and(mask4,mask4,mask=mask5)
    #cv2.imshow("Mask4", mask4)



    ###################  canny edge & find countour #######################

    red_thrs=cv2.bitwise_or(mask,mask1)
    #cv2.imshow("Mask1", red_thrs)
    red_thres=cv2.Canny(red_thrs,200,300, L2gradient = True)
    red_countour,img_cpy=Countour(red_thrs,height,width,img_cpy)
    #cv2.imshow("red-canny", red_thres)
    #cv2.imshow("red-contour", red_countour)
    ##################################################

    #cv2.imshow("red-contour",red_countour)
    blue_thres=cv2.Canny(mask2,200,300,L2gradient=True)
    blue_countour,img_cpy=Countour(blue_thres,height,width,img_cpy)
    #cv2.imshow("blue-canny",blue_thres)
    #cv2.imshow("blue-contour",blue_countour)
    yellow_thres = cv2.Canny(mask4, 200, 300, L2gradient=True)
    yellow_countour,img_cpy=Countour(yellow_thres,height,width,img_cpy)
    #cv2.imshow("yellow-contour",yellow_countour)
    #cv2.imshow("yellow-canny", yellow_thres)


    return img_cpy







# Image Testing
# for x in (glob.glob("detection_data/*.*")):
#
#
#     img = cv2.imread(x)
#
#     results=Traffic_Detection(img,img.copy())
#     cv2.imshow("Final results",results)
#
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



# video camp #########
cap=cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,500)

while True:
    success,img = cap.read()
    results=Traffic_Detection(img,img.copy())
    cv2.imshow("Final results",results)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

