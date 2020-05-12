# -*- coding: utf-8 -*-
"""cifar10 model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c-ptiJp_jVDSJ3e9YRgiriUxjTPplJVs
"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
print("TF VERSION %s" % tf.__version__)
import os, datetime
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray

seed = 1
tf.random.set_seed(seed)
print("Seed Set as:", seed )
# %load_ext tensorboard

labels = [
          "airplane",
          "automobile",
          "bird",
          "cat",
          "deer",
          "dog",
          "frog",
          "horse",
          "ship",
          "truck"
]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

"""
#exagerating edges
laplace_kernel = np.array([
    np.array([1,1,1]),
    np.array([1,-8,1]),
    np.array([1,1,1])
])

sobel_h = np.array([#horizontal edge corner
    np.array([1,2,1]),
    np.array([0,0,0]),
    np.array([-1,-2,-1])
])
"""
def mean_scale(X):
  for i, _ in enumerate(X):
    X[i] /= np.mean(X[i], axis=(0,1,2,))
  print("Finshed")
  return X

x_train /= 255
x_test /= 255

x_train = mean_scale(x_train)
x_test = mean_scale(x_test)
"""
img = 1
plt.imshow(x_test[img])
print(labels[np.max(y_test[img])])
plt.show()
"""


y_train = to_categorical(y_train, len(labels))
y_test = to_categorical(y_test, len(labels))

# Commented out IPython magic to ensure Python compatibility.
def model(input = (32,32,3), dropout = 0.5):
  model = Sequential()
  model.add(Conv2D(64, (3,3), input_shape=input, kernel_initializer='he_uniform', padding="same"))
  model.add(MaxPool2D())
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(192, (3,3), kernel_initializer="he_uniform", padding="same"))
  model.add(MaxPool2D())
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
  model.add(MaxPool2D())
  model.add(Activation('relu'))  
  model.add(Flatten())
  model.add(BatchNormalization())
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  model.add(Dense(256))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  model.add(Dense(10))
  model.add(Activation('softmax'))
  model.compile(
      optimizer = 'adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'],
  )
  return model

model = model(dropout=0.65)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[tensorboard_callback], batch_size=256)
# %tensorboard --logdir logs

