import cv2 as cv
from keras.models import load_model

from utility import predict_image, get_class_label

# Define constant
TEST_LIST_FILE = "inputSignNames.txt"
THRESHOLD = 0.5  # Set threshold as 0.5 to eliminate poor detection

# Load previous trained model
model = load_model('model/CNN_model.h5')

# Image Testing
with open(TEST_LIST_FILE, 'r') as f:  # Read image's path from text files
    file_paths = f.read().splitlines()
    for index, x in enumerate(file_paths):
        img = cv.imread(x)
        classIndex, prob = predict_image(img, model)
        cv.imshow("Traffic Signs", img)
        # Display classification result in console
        if prob < THRESHOLD * 100:
            print("Sign " + str(index + 1) + ": No matching case")
        else:
            print("Sign " + str(index + 1) + ": Class" + str(classIndex) + str(get_class_label(classIndex)) + " " + str(
                round(prob, 2)) + "%")
        cv.waitKey(0)
        cv.destroyAllWindows()
