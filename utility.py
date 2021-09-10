import cv2 as cv
import numpy as np


def rescale_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv.resize(image, (width, height))


def preprocess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return img / 255


def get_empty_image(height, width):
    return np.zeros((height, width, 1), np.uint8)


def get_class_label(class_number):
    if class_number == 0:
        return "non-stopping"
    elif class_number == 1:
        return "60 speed limit"
    elif class_number == 2:
        return "no entry"
    elif class_number == 3:
        return "give way"
    elif class_number == 4:
        return "stop"
    elif class_number == 5:
        return "straight or right"
    elif class_number == 6:
        return "roundabout"
    elif class_number == 7:
        return "traffic light"
    elif class_number == 8:
        return "Invalid type"


def predict_image(img, model):
    img = cv.resize(img, (32, 32))
    img = preprocess(img)
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.array([img])
    probabilities = np.amax(model.predict(img))
    predicted_class = np.argmax(model.predict(img), axis=1)
    return predicted_class, probabilities * 100
