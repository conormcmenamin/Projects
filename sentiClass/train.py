#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:12:28 2019

@author: conor
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.utils import to_categorical
from PIL import Image as Img
DIRECTORY = "/home/conor/Downloads/sentiClass/train/"
CATEGORIES =["Angry","Fear", "Happiness","Neutral","Sadness","Surprise"]


training_set=[]


train_images=[]
train_labels=[]

test_images=[]
test_labels=[]

for ind, category in enumerate(CATEGORIES):
    path = os.path.join(DIRECTORY,category+"/")
    num_class=CATEGORIES.index(category)
    for img in os.listdir(path):
        myImage = Img.open(path +img)
        myImage = myImage.convert('L')
        myImage = myImage.resize((256,256))

        training_set.append([np.asarray(myImage),num_class])

    path = DIRECTORY

np.random.shuffle(training_set)


for element in training_set:
    element[0] = np.asarray(element[0])
    element[0]=element[0].astype('float32')/255

for index, element in enumerate(training_set):
    new_label=[0,0,0,0,0,0]
    new_label[element[1]-1]=1
    if(index<=245):
        train_labels.insert(index,new_label)
        train_images.append(element[0])
    else:
        test_labels.append(new_label)
        test_images.append(element[0])



network = models.Sequential()
network.add(layers.Conv2D(128, (3,3),  activation ='relu', input_shape=(256,256,1)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(256,(3,3), activation= 'relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(256,(3,3), activation= 'relu'))
network.add(layers.Flatten())
network.add(layers.Dense(128, activation  = 'relu'))
network.add(layers.Dense(64,activation="tanh"))
network.add(layers.Dense(6, activation = 'softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



train_images=np.asarray(train_images)
train_images = train_images.reshape(246,256,256,1)

train_labels=np.asarray(train_labels)

train_labels = train_labels.reshape(246,6)

network.fit(train_images,train_labels,epochs=3,batch_size=50)

network.save('myModel.h5')
