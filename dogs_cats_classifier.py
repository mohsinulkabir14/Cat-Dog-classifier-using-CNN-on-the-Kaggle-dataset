import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
img_size = 50
datadir= "F:\Deep Learning\kagglecatsanddogs_3367a\petimage"  #Your datadir here after you download the Kaggle Dataset

GROUPS= ["dog","cat"]


with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())


training_data = []


def creating_training_data():
    for i in GROUPS:
        path=os.path.join(datadir,i)
        class_num=GROUPS.index(i)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                img_array=cv2.resize(img_array,(img_size,img_size))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

creating_training_data()

random.shuffle(training_data)


print(len(training_data))


x=[]
y=[]


for features,labels in training_data:
    x.append(features)
    y.append(labels)
    
x  = np.array(x).reshape(-1,img_size,img_size,1)


x=x/255.0

model= Sequential()

model.add(Conv2D(256,(3,3),input_shape=(50,50,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])



model.fit(x,y,batch_size=16,epochs=5, validation_split=0.3, verbose=0)


######Predict a sample############3

import cv2

groups = ['Dog','Cat']

def prepare(filepath):
    try:
        img_array= cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array= cv2.resize(img_array,(50,50))
        return new_array.reshape(-1,50,50,1)
    except:
        pass

prediction = model.predict([prepare('1.jpg')])


print(groups[int(prediction[0][0])])