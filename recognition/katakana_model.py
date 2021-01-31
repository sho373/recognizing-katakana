import keras

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input

from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys
import pickle
import numpy as np

if not os.path.exists("./model"):
      os.mkdir("./model")

data_file = "./katakana.pickle"
im_size = 48
out_size = 72 # num of katakana characters
im_color = 1
in_shape = (im_size, im_size, im_color)

data = pickle.load(open(data_file, "rb"))


y = []
x = []
for d in data:
  (num, img) = d
  img = img.astype('float').reshape(im_size, im_size, im_color) / 255
  y.append(keras.utils.to_categorical(num, out_size))
  x.append(img)
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

#model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=in_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(out_size))
model.add(Activation('softmax'))

#compile
model.compile(
  loss='categorical_crossentropy',
  optimizer= RMSprop(),
  metrics=['accuracy'])

hist = model.fit(
  x_train, y_train,
  batch_size=64, epochs=50,verbose=1,
  validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy ", score[1], "loss ", score[0])

# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('Accuracy')
# plt.legend(['train', 'test'], loc = 'upper left')
# plt.show()

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('Loss')
# plt.legend(['train', 'test'], loc = 'upper left')
# plt.show()

model.save('model/model.h5')
model.save_weights('model/weights.h5')