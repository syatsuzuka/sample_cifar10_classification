# coding: utf-8

'''
Execute training and create model file
'''

import os
import keras
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras.datasets import cifar10
from keras.optimizers import SGD, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np


class CIFAR10Dataset():

    '''
    Class to prepare cifar-10 dataset
    '''

    def __init__(self):
        self.image_shape = (32, 32, 3)
        self.num_classes = 10

    def get_batch(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, label_data=True) for d in [y_train, y_test]]

        return x_train, y_train, x_test, y_test
    
    def preprocess(self, data, label_data=False):

        if label_data:
            data = keras.utils.to_categorical(data, self.num_classes)
        
        else:
            data = data.astype("float32")
            data /= 255
            shape = (data.shape[0], ) + self.image_shape
            data = data.reshape(shape)

        return data


class Trainer():

    '''
    Class to run traning process
    '''

    def __init__(self, model, loss, optimizer):

        self._target = model
        self._target.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        self.verbose = 1
        self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        self.model_file_name = "model_file.hdf5"
    
    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)
        
        self._target.fit(
            x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            validation_split=validation_split,
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(os.path.join(self.model_dir, self.model_file_name),
                save_best_only=True)
            ],
            verbose=self.verbose            
        )


def network(input_shape, num_classes):

    '''
    Function to define neural network model
    '''
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, padding="same", input_shape=input_shape, activation="relu"))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model


if __name__ == '__main__':

    '''
    main function
    '''

    #======= Start training =======
    s_time = datetime.now()

    #======= Prepare Dataset =======
    dataset = CIFAR10Dataset()
    x_train, y_train, x_test, y_test = dataset.get_batch()

    #======= Prepare for the training =======
    model = network(dataset.image_shape, dataset.num_classes) 
#    optimizer = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    optimizer = RMSprop()
    trainer = Trainer(model, loss="categorical_crossentropy", optimizer=optimizer)

    #======= Run and evaluate the training process =======
    trainer.train(x_train, y_train, batch_size=128, epochs=50, validation_split=0.2)    
    score = model.evaluate(x_test, y_test, verbose=0)

    e_time = datetime.now()
    elapsed_time = e_time - s_time

    print("Test loss: {0}".format(score[0]))
    print("Test accuracy: {0}".format(score[1]))
    print("Elapsed Time: {0}".format(elapsed_time))