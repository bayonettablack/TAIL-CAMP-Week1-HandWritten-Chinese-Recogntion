import os
import numpy as np
from keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from IPython.display import SVG
import matplotlib.pyplot as plt
%matplotlib inline

# create model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(64, 64, 3)))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten()) 
model.add(Dropout(0.5)) 
model.add(Dense(2048, activation='relu'))  
model.add(Dense(1877, activation='softmax'))

model.summary()

###################################################################
#                                                                 #
#                  the structure of network                       #
#                                                                 #
###################################################################
#_________________________________________________________________#
#Layer (type)                 Output Shape              Param #   #
#=================================================================#
#conv2d_1 (Conv2D)            (None, 64, 64, 32)        896       #
#_________________________________________________________________#
#conv2d_2 (Conv2D)            (None, 64, 64, 32)        9248      #
#_________________________________________________________________#
#max_pooling2d_1 (MaxPooling2 (None, 32, 32, 32)        0         #
#_________________________________________________________________#
#conv2d_3 (Conv2D)            (None, 32, 32, 64)        18496     #
#_________________________________________________________________#
#conv2d_4 (Conv2D)            (None, 32, 32, 64)        36928     #
#_________________________________________________________________#
#max_pooling2d_2 (MaxPooling2 (None, 16, 16, 64)        0         #
#_________________________________________________________________#
#flatten_1 (Flatten)          (None, 16384)             0         #
#_________________________________________________________________#
#dropout_1 (Dropout)          (None, 16384)             0         #
#_________________________________________________________________#
#dense_1 (Dense)              (None, 2048)              33556480  #
#_________________________________________________________________#
#dense_2 (Dense)              (None, 1877)              3845973   #
#=================================================================#
#Total params: 37,468,021                                         #
#Trainable params: 37,468,021                                     #
#Non-trainable params: 0                                          #
###################################################################

data_dir = '/home/kesci/work/week1-keras/'
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'val')

img_width, img_height = 64, 64
charset_size = 1877
nb_validation_samples = 200
nb_samples_per_epoch = 200
nb_nb_epoch = 5;

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 20,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size = (img_width, img_height),
                                                    batch_size = 1024,
                                                    color_mode = 'rgb',
                                                    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(test_data_dir,
                                                        target_size = (img_width, img_height),
                                                        batch_size = 1024, 
                                                        color_mode = 'rgb',
                                                        class_mode = 'categorical')

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

filepath = '/home/kesci/work/ConvNet-small-best.h5'

checkpoint = ModelCheckpoint(filepath,
                             monitor = 'val_acc', 
                             verbose = True, 
                             save_best_only = True, 
                             mode = 'auto')

reduce = ReduceLROnPlateau(monitor = 'val_loss', 
                         factor = 0.1, 
                         patience = 2, 
                         verbose = True, 
                         mode = 'auto', 
                         epsilon = 0.0001, 
                         cooldown = 0, 
                         min_lr = 0)

callbacks_list = [checkpoint, reduce]

history = model.fit_generator(train_generator,
                    steps_per_epoch = nb_samples_per_epoch,
                    epochs = nb_nb_epoch,
                    callbacks = callbacks_list,
                    validation_data = validation_generator,
                    validation_steps = nb_validation_samples)

model.save('/home/kesci/work/ConvNet-small-5iter.h5')

# plot the acc loss val_acc curve
val_loss = loss = history.history['val_loss']
loss = history.history['loss']
acc = history.history['acc']
lr = history.history['lr']

plt.plot(val_loss, label = 'val_loss')
plt.plot(loss, label = 'train loss')
plt.plot(acc, label = 'acc')
plt.plot(lr, label = 'lr')

plt.legend()
plt.show()