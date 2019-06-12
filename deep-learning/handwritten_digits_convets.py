'''

runtime accurracy: 99.19%
runtime output:

Epoch 1/5
60000/60000 [==============================] - 34s 572us/step - loss: 0.2348 - acc: 0.9276
Epoch 2/5
60000/60000 [==============================] - 33s 553us/step - loss: 0.0552 - acc: 0.9828
Epoch 3/5
60000/60000 [==============================] - 32s 537us/step - loss: 0.0380 - acc: 0.9881
Epoch 4/5
60000/60000 [==============================] - 33s 552us/step - loss: 0.0273 - acc: 0.9912
Epoch 5/5
60000/60000 [==============================] - 33s 554us/step - loss: 0.0219 - acc: 0.9929
10000/10000 [==============================] - 2s 240us/step
test_acc 0.9919

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

network.fit(train_images,train_labels,epochs= 5,batch_size = 128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc', test_acc)
