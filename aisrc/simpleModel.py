
from keras.models import Model
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Convolution1D
from keras.layers import Flatten
from keras.layers import MaxPool1D
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
from keras.layers import BatchNormalization


def makeModel(train_inputs, train_outputs):
    
    model = Sequential()

    im_shape=(train_inputs.shape[1],1)
    
    model.add(Convolution1D(64, (6), activation='relu', input_shape=im_shape))               #inputs_cnn
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=(3), strides=(2), padding="same"))

    model.add(Convolution1D(64, (3), activation='relu', input_shape=im_shape))               #pool1
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=(2), strides=(2), padding="same"))

    model.add(Convolution1D(64, (3), activation='relu', input_shape=im_shape))               #pool2
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=(2), strides=(2), padding="same"))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))                                             #flatten
    model.add(Dense(32, activation='relu'))                                             #dense_end1
    model.add(Dense(5, activation='softmax', name='main_output'))

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def trainModel(model, train_inputs, train_outputs, test_inputs, test_outputs):

    model.fit(train_inputs, train_outputs, epochs = 10, batch_size = 32, validation_data = (test_inputs, test_outputs))
    model.save("simplemodel")

    return model

def trainModelClassWeight(model, train_inputs, train_outputs):

    class_weights = class_weight.compute_class_weight('balanced',np.unique(train_outputs),train_outputs)
    model.fit(train_inputs, train_outputs, class_weight=class_weights)
    model.save("simplemodel")

    return model

