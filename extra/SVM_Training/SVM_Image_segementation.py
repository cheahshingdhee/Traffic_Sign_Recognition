import cv2
import numpy as np
import glob


def getEmptyPic(height, width):
    return np.zeros((height,width,1), np.uint8)

def red_blue_normalize(height,width,img):
    results = getEmptyPic(height, width)  # to store the results of red/blue-normalization
    for x in range(height):
        for y in range(width):
            r=img[x,y,2]
            g=img[x,y,1]
            b=img[x,y,0]
            r = int(r)
            g = int(g)
            b = int(b)
            red = r/(r+b+(g*4)+8)
            blue = b/(r+b+(g*4)+8)
            if red>blue :
                results[x,y]=red*255+10
            else:
                results[x, y] = blue*255+10
    return results

def masking(img,imgContour):
    im_floodfill = imgContour.copy()
    h, w = imgContour.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img1_bg = cv2.bitwise_and(img, img, mask=im_floodfill_inv)
    return img1_bg

def Img_segmentation(img,height,width):
    ######################################################  red blue normalization ############################
    results = getEmptyPic(height, width)  # to store the results of red/blue-normalization
    results = red_blue_normalize(height, width, img)

    ######################################################  image-preprocessing ############################

    ### results - image after red-blue normalization
    #cv2.imshow("resultsssss", results)

    imgBlur = cv2.GaussianBlur(results, (7, 7), sigmaX=1, sigmaY=0)
    imgBlur = cv2.normalize(imgBlur, imgBlur, 0, 255, cv2.NORM_MINMAX)

    ### imgBlur - image after Gaussian blur
    # cv2.imshow("Blur",imgBlur)

    imgCanny = cv2.Canny(imgBlur, 25, 75)
    ## imgCanny- image after canny edge detection
    # cv2.imshow("Canny",imgCanny)

    ############################################# filtering  max size and max area countours ###########################
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    max_area = 0
    max_size = 0
    index_size = 0
    count = 0
    index_area = 0

    ###  find the max area and size
    for cnt in contours:

        area = cv2.contourArea(cnt)
        # print(area)
        size = (int(cnt.shape[0]) * int(cnt.shape[1]))
        if max_size < size:
            max_size = size
            index_size = count
        if max_area < area:
            max_area = area
            index_area = count
        count = count + 1

    ######## CONVEX HULL- connecting the countours found to make them encolsed
    convex_hull_area = getEmptyPic(height, width)  ## convex hull area found
    tep = []
    tep.append(contours[index_area])  # area
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in tep]
    cv2.polylines(convex_hull_area, hulls, 1, (255, 255, 255))
    ## the results after connecting the  disconnected countour
    #cv2.imshow("convex hull-area", convex_hull_area)

    ######## CONVEX HULL- connecting the countours found to make them encolsed
    convex_hull_size = getEmptyPic(height, width)  ## convex hull size found
    tep2 = []
    tep2.append(contours[index_size])  # size
    hulls2 = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in tep2]
    cv2.polylines(convex_hull_size, hulls2, 1, (255, 255, 255))
    ## the results after connecting the  disconnected countour
    #cv2.imshow("convex hull-size", convex_hull_size)

    contours_cvx_a, hierarchy_cvx = cv2.findContours(convex_hull_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_cvx_s, hierarchy_cvx_s = cv2.findContours(convex_hull_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ### draw countour on empty images
    # initialize empty pic
    imgContour_convex_area = getEmptyPic(height, width)
    imgContour_convex_size = getEmptyPic(height, width)
    imgContour_area = getEmptyPic(height, width)
    imgContour_size = getEmptyPic(height, width)
    # draw found countors on empty image
    cv2.drawContours(imgContour_area, contours, contourIdx=index_area, color=(255, 0, 0), thickness=2)
    cv2.drawContours(imgContour_size, contours, contourIdx=index_size, color=(255, 0, 0), thickness=2)
    cv2.drawContours(imgContour_convex_area, contours_cvx_a, contourIdx=-1, color=(255, 0, 0), thickness=2)
    cv2.drawContours(imgContour_convex_size, contours_cvx_s, contourIdx=-1, color=(255, 0, 0), thickness=2)

    ################################################################ MASKING ############################333
    # masking function-size

    # im_floodfill = imgContour_size.copy()
    # h, w = imgContour_size.shape[:2]
    # mask = np.zeros((h+2, w+2), np.uint8)
    # cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # img1_bg1 = cv2.bitwise_and(img,img,mask = im_floodfill_inv)

    img1_bg1 = masking(img, imgContour_size)
    #cv2.imshow("size", img1_bg1)

    # masking function-area

    img1_bg2 = masking(img, imgContour_area)
    #cv2.imshow("area", img1_bg2)

    # masking function-convex

    img1_bg3 = masking(img, imgContour_convex_area)
    #cv2.imshow("convex-area", img1_bg3)

    # masking function-convex

    img1_bg4 = masking(img, imgContour_convex_size)
    #cv2.imshow("convex-size", img1_bg4)

    ############################################################# output ################################################3
    ##prepocessing
    # cv2.imshow("results",results)
    # cv2.imshow("Blur+normalize",imgBlur)
    # cv2.imshow("Canny",imgCanny)

    out_p = np.hstack((imgBlur, imgCanny))
    #cv2.imshow("output_p", out_p)

    ######### boundary detection
    # cv2.imshow("Countors-area",imgContour_area)
    # cv2.imshow("Countors-size",imgContour_size)
    # cv2.imshow("Countors-convex",imgContour_convex_area)
    out_bd = np.hstack((imgContour_size, imgContour_area, imgContour_convex_size, imgContour_convex_area))
    #cv2.imshow("output_bd", out_bd)

    ## results
    # cv2.imshow("area",img1_bg1)
    # cv2.imshow("size",img1_bg2)
    # cv2.imshow("convex",img1_bg3)

    out2 = np.hstack((img1_bg4, img1_bg3))
    out = np.hstack((img1_bg1, img1_bg2, out2))
    cv2.imshow("output", out)
    return img1_bg1,img1_bg2,img1_bg3,img1_bg4


for x in (glob.glob("segmentation_dataset/*.*")):
    img = cv2.imread(x)
    cv2.imshow("orginal",img)
    height, width, channels = img.shape
    img1_bg1, img1_bg2, img1_bg3, img1_bg4=Img_segmentation(img,height,width)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



