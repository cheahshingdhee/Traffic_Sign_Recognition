import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from utility import preprocess

# Define constant
DATASET_PATH = "dataset"
CATEGORIES = ["non-stopping", "60_speed_limit", "no_entry", "give_way", "stop", "straight_or_right", "roundabout",
              "traffic_light", "invalid_type"]
IMAGE_DIMENSIONS = (32, 32, 3)
TESTING_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Load the images
print("[INFO] Start to load the images...")

data = []
labels = []
classList = os.listdir(DATASET_PATH)
noOfClasses = len(classList)
print("[INFO] Total class detected: ", noOfClasses)

for classNumber in range(0, noOfClasses):
    path = os.path.join(DATASET_PATH, str(classNumber))
    for img_file in os.listdir(path):
        img_path = os.path.join(path, img_file)
        image = cv.imread(img_path)
        image = cv.resize(image, (32, 32))
        data.append(image)
        labels.append(classNumber)
    print("[INFO] {} images are loaded.".format(CATEGORIES[classNumber]))

data = np.array(data)
labels = np.array(labels)
print("[INFO] Shape of data: ", data.shape)
print("[INFO] Shape of label: ", labels.shape)

# Split the training, testing and validation data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=TESTING_SIZE)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE)
print("[INFO] Data is split into training, testing and validation data.")
print("[INFO] Testing size: {}".format(TESTING_SIZE))
print("[INFO] Validation size: {}".format(VALIDATION_SIZE))

# Preprocess the data
X_train = np.array(list(map(preprocess, X_train)))
X_validation = np.array(list(map(preprocess, X_validation)))
X_test = np.array(list(map(preprocess, X_test)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print("[INFO] Data preprocess completed.")

############################### Data Augmentation########################
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


############## CNN Construction ######################
def CNNmodel():
    size_of_pool = (2, 2)
    model = Sequential()
    model.add((Conv2D(32, (3, 3), input_shape=(IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1], 1),
                      activation='relu')))  # 32 filters
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(64, (3, 3), input_shape=(IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1], 1),
                      activation='relu')))  # 64 filters
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))  # set dropout-rate as 0.5

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))  # set dropout-rate as 0.5
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(learning_rate=0.001), metrics=['accuracy'], loss='categorical_crossentropy')
    return model


########## TRAIN ##################

model = CNNmodel()
print(model.summary())
history = model.fit(dataGen.flow(X_train, y_train, batch_size=20), epochs=30,
                    validation_data=(X_validation, y_validation))

# Plot graphs
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print("Test score ", score[0])
print("Test Accuracy", score[1])

y_pred = []
Pred = model.predict(X_test, verbose=0)
print(Pred[1])
Pred_Label = np.argmax(Pred, axis=1)
Pred_Label_p = np.max(Pred, axis=1)

y_test = np.argmax(y_test, axis=1)

print(y_test.shape, Pred_Label.shape, Pred_Label_p.shape)
print(Pred_Label[1])
print(Pred_Label_p[1])

ConfusionM = confusion_matrix(np.array(y_test), Pred_Label)
class_report = classification_report(np.array(y_test), Pred_Label)

print(ConfusionM)
print(class_report)
accuracy = accuracy_score(y_test, Pred_Label)
print('Accuracy: %f' % accuracy)

n_class = 9
fpr = {}
tpr = {}
thresh = {}
for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, Pred[:, i], pos_label=i)

# plot ROC Curve
plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--', color='red', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--', color='yellow', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--', color='blue', label='Class 5 vs Rest')
plt.plot(fpr[6], tpr[6], linestyle='--', color='purple', label='Class 6 vs Rest')
plt.plot(fpr[7], tpr[7], linestyle='--', color='gray', label='Class 7 vs Rest')
plt.plot(fpr[8], tpr[8], linestyle='--', color='brown', label='Class 8 vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC', dpi=300)
plt.show()

# plot Precision vs Recall Curve
precision = dict()
recall = dict()
for i in range(n_class):
    precision[i], recall[i], _ = precision_recall_curve(y_test, Pred[:, i], pos_label=i)
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.savefig('Precision vs Recall', dpi=300)
plt.show()

# Save the model
model.save('model/CNN_model.h5')  # creates a HDF5 file 'CNN_model.h5'
