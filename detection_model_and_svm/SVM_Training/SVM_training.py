import os
import numpy as np
import cv2
import pickle
from skimage.feature import hog
from sklearn.metrics import roc_curve
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

print("done")


def HOG(img):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                        multichannel=True, )
    return hog_image, fd


path = "../../dataset"

count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total class detected : ", len(myList))

noOfClasses = len(myList)
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        curImg = cv2.resize(curImg, (32, 32))  # resize the images
        curImg, fd = HOG(curImg)  # generate HOG feature
        images.append(fd)
        classNo.append(count)
    print(count)
    count += 1

clf = svm.SVC(probability=True)

hog_feature = np.array(images)
hog_output = np.array(classNo).reshape(-1, 1)
print(hog_feature.shape)
print(hog_output.shape)

data_frame = np.hstack((hog_feature, hog_output))
print(data_frame.shape)
np.random.shuffle(data_frame)
# What percentage of data you want to keep for training
percentage = 70
partition = int(len(hog_feature) * percentage / 100)
x_train, x_test = data_frame[:partition, :-1], data_frame[partition:, :-1]
y_train, y_test = data_frame[:partition, -1:].ravel(), data_frame[partition:, -1:].ravel()

clf.fit(x_train, y_train)


y_pred = clf.predict(x_test)
# Testing accuracy
print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
print('\n')

# 5-fold Cross Validation
scores = cross_val_score(clf, x_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#########################################################################################
y_pred = []
Pred_Label = clf.predict(x_test)
y_prob_pred_cnb = clf.predict_proba(x_test)
Pred_Label_p = clf.predict_proba(x_test)
Pred_Label_p = np.max(Pred_Label_p, axis=1)

# Confusion Matrix
ConfusionM = confusion_matrix(np.array(y_test), Pred_Label)
# Precision and Recall for multiclass
class_report = classification_report(np.array(y_test), Pred_Label)

print(ConfusionM)
print(class_report)

fpr = {}
tpr = {}
thresh = {}

n_class = 9

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_prob_pred_cnb[:, i], pos_label=i)

# plotting ROC
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
plt.show()



# save the model#
with open('SVM_MODEL_FINAL', 'wb') as f:
    pickle.dump(clf, f)
