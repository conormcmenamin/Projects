'''

runtime accurracy: 98.98%
runtime output:

Epoch 1/3
60000/60000 [==============================] - 34s 564us/step - loss: 0.2278 - acc: 0.9273
Epoch 2/3
60000/60000 [==============================] - 33s 557us/step - loss: 0.0552 - acc: 0.9830
Epoch 3/3
60000/60000 [==============================] - 34s 562us/step - loss: 0.0373 - acc: 0.9880
10000/10000 [==============================] - 2s 240us/step
test_acc 0.9898
'''
from keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape =(28,28,1)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64,(3,3), activation= 'relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64,(3,3), activation= 'relu'))
network.add(layers.Flatten())
network.add(layers.Dense(64, activation  = 'relu',))
network.add(layers.Dense(10, activation = 'softmax'))

network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics =['accuracy'])
train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32') /255

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)  

network.fit(train_images,train_labels,epochs= 3,batch_size = 128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc', test_acc)
