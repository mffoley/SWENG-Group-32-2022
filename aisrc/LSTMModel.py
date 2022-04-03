
from keras.models import Model 
from keras.models import Sequential
from keras.layers import Dense, Dropout                                                               
from keras.layers import LSTM
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer


def makeModelLSTM(train_inputs):
    

    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(187,1)))
    lstm_model.add(Dense(128, activation = 'relu'))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(5, activation = 'softmax'))

    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return lstm_model


def trainModelLSTM(model, train_inputs, train_outputs, test_inputs, test_outputs):

    history = model.fit(train_inputs, train_outputs, epochs=20, batch_size=100, validation_data=(test_inputs, test_outputs))
    model.save("LSTMmodel")

    return model



#raw_train_outputs needs to be in the raw form before changed to binary in the to_categorical function
def trainModelClassWeightLSTM(model, train_inputs, train_outputs, test_inputs, test_outputs, raw_train_outputs):

    class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes = np.unique(raw_train_outputs), y = raw_train_outputs)
    
    class_weights = dict(enumerate(class_weights))#don't know whether this is necessary or not
    callbacks = [keras.callbacks.ModelCheckpoint("best model at epoch:{epoch}.h5", save_best_only=True)]#don't know whether this is necessary or not

    model.fit(train_inputs, train_outputs, epochs = 40, batch_size = 32, class_weight=class_weights, 
    callbacks = callbacks, validation_data=(test_inputs, test_outputs), shuffle=True)
    #shuffle and validation data can be removed

    model.save("LSTMmodelweighted")

    return model

