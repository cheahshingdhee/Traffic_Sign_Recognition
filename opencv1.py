import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# def empty(a):
#     pass
#
# print("done")
#
#
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,240)
# cv2.createTrackbar("Hue min","TrackBars",0,179,empty)
# cv2.createTrackbar("Hue max","TrackBars",179,179,empty)
# cv2.createTrackbar("sat min","TrackBars",0,255,empty)
# cv2.createTrackbar("sat max","TrackBars",255,255,empty)
# cv2.createTrackbar("val min","TrackBars",0,255,empty)
# cv2.createTrackbar("val max","TrackBars",255,255,empty)
#
# while True:
#     img=cv2.imread("Img/t6.png")
#     imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     imgHSV=cv2.GaussianBlur(imgHSV,(7,7),sigmaX=1,sigmaY=0)
#     h_min=cv2.getTrackbarPos("Hue min","TrackBars")
#     h_max=cv2.getTrackbarPos("Hue max","TrackBars")
#     s_min=cv2.getTrackbarPos("sat min","TrackBars")
#     s_max=cv2.getTrackbarPos("sat max","TrackBars")
#     v_min=cv2.getTrackbarPos("val min","TrackBars")
#     v_max=cv2.getTrackbarPos("val max","TrackBars")
#     print(h_min,h_max,s_min,s_max,v_min,v_max)
#     lower =np.array([h_min,s_min,v_min])
#     upper= np.array([h_max,s_max,v_max])
#     mask=cv2.inRange(imgHSV,lower,upper)
#     cv2.imshow("HSV",imgHSV)
#     cv2.imshow("Mask",mask)
#
#     cv2.waitKey(1)



#
# cv2.imshow("output",img)
#
# imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray",imgGray)
# #cv2.waitKey(0)
#
# imgBlur=cv2.GaussianBlur(img,(7,7),0)
# cv2.imshow("Blur",imgBlur)
# #cv2.waitKey(0)
#
#
# imgCanny=cv2.Canny(img,150,200)
# cv2.imshow("Blur",imgCanny)
# #cv2.waitKey(0)
#
# # iteration =how many layer of dillation
# # kernel is matrix
# kernel=np.ones((5,5),np.uint8)
# imgDil=cv2.dilate(imgCanny,kernel,iterations=1)
# cv2.imshow("dil",imgDil)
# #cv2.waitKey(0)
#
#
# imgErode=cv2.erode(imgDil,kernel,iterations=1)
# cv2.imshow("erosion",imgErode)
# cv2.waitKey(0)b
for  x in range(100):


    img=cv2.imread("Img/t{num}.png".format(num=x+1))
    # if x+2 <=9:
    #     img = cv2.imread("dataset/Traffic signs/005_1_000{num}.png".format(num=x+2))
    #     temp_num = x+2
    #     print("dataset/Traffic signs/005_1_000{num}.png".format(num=x + 2))
    #
    # else:
    #     if x+2==37:
    #         img = cv2.imread("dataset/Traffic signs/005_1_00{num}.png".format(num=x + 2+1))
    #         temp_num = x+2+1
    #         print("dataset/Traffic signs/005_1_00{num}.png".format(num=x + 2+1))
    #
    #     else:
    #         img = cv2.imread("dataset/Traffic signs/005_1_00{num}.png".format(num=x + 2))
    #         temp_num = x+2
    #         print("dataset/Traffic signs/005_1_00{num}.png".format(num=x + 2))

    height, width, channels = img.shape



    ##################################################### initial lize empty pic  #############################
    results = np.zeros((height,width,1), np.uint8)
    cpy= np.zeros((height,width,1), np.uint8)
    cpy2= np.zeros((height,width,1), np.uint8)
    cpy3= np.zeros((height,width,1), np.uint8)
    cpy4= np.zeros((height,width,1), np.uint8)
    cpy5= np.zeros((height,width,1), np.uint8)
    cpy6= np.zeros((height,width,1), np.uint8)

    imgContour =np.zeros((height,width,1), np.uint8)
    imgContour2 =np.zeros((height,width,1), np.uint8)


    ######################################################  red blue normalization ############################
    for x in range(height):
        for y in range(width):
            r=img[x,y,2]
            g=img[x,y,1]
            b=img[x,y,0]
            r = int(r)
            g = int(g)
            b = int(b)
            red = r/(r+b+g+1)
            blue = b/(r+b+g+1)
            if red>blue :
                results[x,y]=red*255+10
            else:
                results[x, y] = blue*255+10


    ### results - image after red-blue normalization
    #cv2.imshow("results",results)


    imgBlur=cv2.GaussianBlur(results,(7,7),sigmaX=1,sigmaY=0)
    imgBlur=cv2.normalize(imgBlur,imgBlur,0,255,cv2.NORM_MINMAX)

    ### imgBlur - image after Gaussian blur
    #cv2.imshow("Blur",imgBlur)

    imgCanny=cv2.Canny(imgBlur,25,75)
    ## imgCanny- image after canny edge detection
    #cv2.imshow("Canny",imgCanny)


    ############################################# filtering  max size and max area countours ###########################
    contours ,hierarchy= cv2.findContours(imgCanny,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    max=0
    max2=0
    index2=0
    count=0
    index=0

    ###  find the max area and size
    for cnt in contours:

        area=cv2.contourArea(cnt)
        #print(area)
        size=(int(cnt.shape[0])*int(cnt.shape[1]))
        if max2<size:
            max2=size
            index2=count
        if max <area:
            max=area
            index=count
        count=count+1


    ######## CONVEX HULL- connecting the countours found to make them encolsed
    tep=[]
    tep.append(contours[index]) # area
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in tep]
    cv2.polylines(cpy, hulls, 1, (255, 255, 255))
    ## the results after connecting the  disconnected countour
    cv2.imshow("convex hull-area",cpy)

    ######## CONVEX HULL- connecting the countours found to make them encolsed
    tep2 = []
    tep2.append(contours[index2]) # size
    hulls2 = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in tep2]
    cv2.polylines(cpy5, hulls2, 1, (255, 255, 255))
    ## the results after connecting the  disconnected countour
    cv2.imshow("convex hull-size", cpy5)


    contours_cvx ,hierarchy_cvx= cv2.findContours(cpy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours_cvx_s ,hierarchy_cvx_s= cv2.findContours(cpy5,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


    ### draw countour on empty images
    cv2.drawContours(imgContour, contours, contourIdx=index, color=(255, 0, 0), thickness=2)
    cv2.drawContours(imgContour2, contours, contourIdx=index2, color=(255, 0, 0), thickness=2)
    cv2.drawContours(cpy2, contours_cvx, contourIdx=-1, color=(255, 0, 0), thickness=2)
    cv2.drawContours(cpy6, contours_cvx_s, contourIdx=-1, color=(255, 0, 0), thickness=2)




    ################################################################ MASKING ############################333
    #masking function-size

    im_floodfill = imgContour2.copy()
    h, w = imgContour.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img1_bg1 = cv2.bitwise_and(img,img,mask = im_floodfill_inv)
    cv2.imshow("size",img1_bg1)

    #masking function-area

    im_floodfill = imgContour.copy()
    h, w = imgContour.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img1_bg2 = cv2.bitwise_and(img,img,mask = im_floodfill_inv)
    cv2.imshow("area",img1_bg2)

    #masking function-convex

    im_floodfill = cpy2.copy()
    h, w = imgContour.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img1_bg3 = cv2.bitwise_and(img,img,mask = im_floodfill_inv)
    cv2.imshow("convex-area",img1_bg3)

    # masking function-convex

    im_floodfill = cpy6.copy()
    h, w = imgContour.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img1_bg4 = cv2.bitwise_and(img, img, mask=im_floodfill_inv)
    cv2.imshow("convex-size",img1_bg4)

    ############################################################# output ################################################3
    ##prepocessing
    #cv2.imshow("results",results)
    #cv2.imshow("Blur+normalize",imgBlur)
    #cv2.imshow("Canny",imgCanny)

    out_p=np.hstack((imgBlur,imgCanny))
    cv2.imshow("output_p",out_p)

    ######### boundary detection
    #cv2.imshow("Countors-area",imgContour)
    #cv2.imshow("Countors-size",imgContour2)
    #cv2.imshow("Countors-convex",cpy2)
    out_bd=np.hstack((imgContour2,imgContour,cpy6,cpy2))
    cv2.imshow("output_bd",out_bd)

    ## results
    #cv2.imshow("area",img1_bg1)
    #cv2.imshow("size",img1_bg2)
    #cv2.imshow("convex",img1_bg3)


    out2=np.hstack((img1_bg4,img1_bg3))
    out=np.hstack((img1_bg1,img1_bg2,out2))
    cv2.imshow("output",out)



    ########################################################################### MSER ###################################

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe= clahe.apply(results)
    cv2.imshow("CLAHE image", results)
    mser = cv2.MSER_create(delta=12)
    vis = imgBlur.copy()
    regions, boundingBoxes = mser.detectRegions(clahe)
    reglen=len(regions)
    max3=0
    index3=0
    print("regions ",reglen)
    # find max regions
    for x in range(reglen):
        size = regions[x].shape[0] * regions[x].shape[1]
        #print("MSER ", size)
        if max3 < size:
            max3 = size
            index3 = x

    arr=[]
    num=0
    #print("index 3",index3)

    #draw bouding box
    for box in boundingBoxes:
            x, y, w, h = box;
            if w*h<=0.7*height*width:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)
                temp = np.zeros((height, width, 1), np.uint8)
                temp = img[y:y + h, x:x + w]
                arr.append(temp.copy())
                # cv2.imshow('MSErs', temp)
                # cv2.waitKey(0)

    cv2.imshow('MSEr', vis)
    #cv2.waitKey(0)
    #print("shape ",len(arr))

###### generating  all detected  MSER

# for x in range(len(arr)):
#     cv2.imshow('filter{num}'.format(num=x), arr[x])
#
#
# cv2.waitKey(0)
#

# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
# #cv2.fillPoly(vis, hulls, (0, 255, 0))
#
# cv2.polylines(vis, hulls, 1, (255, 255, 255))
# cv2.imshow('MSEr', vis)
# cv2.waitKey(0)


################################################### generate HOG features ###################################33
# if index==index2:
#     resized_img = resize(imgContour2, (128*2, 64*2))
# else:
#     resized_img = resize(imgContour, (128*2, 64*2))
#
# cv2.imshow("resize",resized_img)
# cv2.waitKey(0)
# fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
# plt.axis("off")
# hog_image = cv2.normalize(hog_image,  hog_image, 0, 255, cv2.NORM_MINMAX)
#
# cv2.imshow("resizeds",hog_image)
# plt.imshow(hog_image, cmap="gray")
# plt.show()






    # Showing all the three images
    cv2.imshow("original", img)

    cv2.waitKey(0)





