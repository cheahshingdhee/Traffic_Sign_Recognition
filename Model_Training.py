import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pickle
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D ,MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pandas
import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import  ImageDataGenerator
print("done")


path="traindataset"
batch_size_val=50
steps_per_epoch_val=2000
epochs_val=10
imageDimesions=(32,32,3)



count=0
images=[]
classNo=[]
myList = os.listdir(path)
print("Total class detected : ",len(myList))

noOfClasses=len(myList)
for x in range(0,len(myList)):
    myPicList=os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg=cv2.imread(path+"/"+str(count)+"/"+y)
        curImg=cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNo.append(count)
    print(count)
    count +=1
print(" ")
images=np.array(images)
classNo=np.array(classNo)

print("Shape of data ",images.shape)
print("Shape of label ",classNo.shape)


X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size=0.2)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.2)


print("done-spliting")
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



X_train=np.array(list(map(preprocessing,X_train)))
X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)



print("done-applying steps")


dataGen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)
dataGen.fit(X_train)
batches=dataGen.flow(X_train,y_train,batch_size=20)
X_batch,y_batch=next(batches)

fig,axs=plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0],32))
    axs[i].axis('off')
plt.show()


y_train=to_categorical(y_train,noOfClasses)
y_validation=to_categorical(y_validation,noOfClasses)
y_test=to_categorical(y_test,noOfClasses)

def CNNmodel():


    size_of_pool=(2,2)
    model=Sequential()
    model.add((Conv2D(32,(3,3),input_shape=(imageDimesions[0],imageDimesions[1],1),activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(64, (3, 3), input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))

    model.compile(Adam(learning_rate=0.001),metrics=['accuracy'],loss='categorical_crossentropy')
    return model


########## TRAIN

model= CNNmodel()
print(model.summary())
history=model.fit(dataGen.flow(X_train,y_train,batch_size=20),epochs=50,validation_data=(X_validation,y_validation))


## plot
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','valdation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','valdation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()

score=model.evaluate(X_test,y_test,verbose=0)
print("Test score ",score[0])
print("Test Accuaracy",score[1])

from keras.models import load_model

# save the model
#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'


# load the model
#model = load_model('my_model.h5')

img=cv2.imread("Img/testing.png")
img=cv2.resize(img,(32,32))
img=preprocessing(img)
img=img.reshape(img.shape[0],img.shape[1],1)
test=[img]
test=np.array(test)
print("Predicted = ",np.argmax(model.predict(test)))



