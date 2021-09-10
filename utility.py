import cv2 as cv


def rescale_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv.resize(image, (width, height))


def preprocess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return img / 255
