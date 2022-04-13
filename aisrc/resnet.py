from keras import layers
from keras.layers import Input, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Embedding, Add
from keras.layers import Convolution1D, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D, MaxPool1D, ZeroPadding1D, GlobalMaxPooling2D, GlobalAveragePooling2D, LSTM, SpatialDropout1D
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.utils import plot_model
from keras.applications.inception_v3 import InceptionV3
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization
import numpy as np
import tensorflow as tf
from tensorflow import keras

def identity_block(X, f, filters):
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Convolution1D(filters = F1, kernel_size = 1, activation='relu', strides = 1, padding = 'valid')(X)
    X = BatchNormalization()(X)
    
    X = Convolution1D(filters = F2, kernel_size = f, activation='relu', strides = 1, padding = 'same')(X)
    X = BatchNormalization()(X)

    X = Convolution1D(filters = F3, kernel_size = 1, activation='relu', strides = 1, padding = 'valid')(X)
    X = BatchNormalization()(X)

    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, s = 2):
    #retrieve filters
        F1, F2, F3 = filters
    #save input value  
        X_shortcut = X
    #main path
    #first componeent of main path
        X = Convolution1D(F1, 1, activation='relu', strides = s)(X)
        X = BatchNormalization()(X)
    #second component of main path  
        X = Convolution1D(F2, f, activation='relu', strides = 1,padding = 'same')(X)
        X = BatchNormalization()(X)
    #third component of main path
        X = Convolution1D(F3, 1, strides = 1)(X)
        X = BatchNormalization()(X)
    #shortcut path
        X_shortcut = Convolution1D(F3, 1, strides = s)(X_shortcut)
        X_shortcut = BatchNormalization()(X_shortcut)
    #add shortcut to main path, pass it through RELU activation  
        X = Add()([X,X_shortcut])
        X = Activation('relu')(X)

        return X

def ResNet50(input_shape = (187,1)):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    X = ZeroPadding1D(3)(X_input)
    #stage one
    X = Convolution1D(64, 7, activation='relu', strides = 2)(X)
    X = BatchNormalization()(X)
    X = MaxPool1D(pool_size=2, strides=2, padding='same')(X)
    #stage two
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])
    #stage three
    X = convolutional_block(X, f = 3, filters = [128,128,512], s = 2)
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])
    #stage four
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    #stage five
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])
    #max pooling
    X = MaxPool1D(pool_size=2, strides=2, padding='same')(X)
    #output layer
    X = Flatten()(X)
    X = Dense(5,activation='softmax')(X)
    #create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

def trainResNet50Model(model, train_inputs, train_outputs, test_inputs, test_outputs):

    model.fit(train_inputs, train_outputs, epochs=20, batch_size=100, verbose=2)
    model.save("ResNetmodel")

    return model


def evaluateModel(model, test_input, test_output):
    model.evaluate(test_input, test_output)

#given a model and test inputs, predicts the class each row of the input belongs to
def predictValues(model, test_inputs):

    prediction = model.predict(test_inputs)
    
    return prediction

def loadModel(name):

    new_model = tf.keras.models.load_model(name)

    return new_model

   


