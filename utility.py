import cv2 as cv


def rescale_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv.resize(image, (width, height))
