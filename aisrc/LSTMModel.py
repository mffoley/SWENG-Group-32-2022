
from keras.models import Model 
from keras.models import Sequential
from keras.layers import Dense                                                               
from keras.layers import CuDNNLSTM, Bidirectional
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer


def makeModel(train_inputs):
    
    model = Sequential()

    im_shape=(train_inputs.shape[1],1)

    t = Tokenizer()
    vocab_size = len(t.word_index) + 1

    model.add(Embedding(vocab_size, 64, input_shape=im_shape))                                              #vertion2:model.add(Embedding(2500, embed_dim,input_length = im_shape, dropout = 0.2))
    #input_length=max_length, maxlength=150. Maybe these two version are the same
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True, dropout_U = 0.2, dropout_W = 0.2)) )       #vertion2:model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
    model.add(Bidirectional(CuDNNLSTM(64, dropout_U = 0.2, dropout_W = 0.2)))

    #vertion3:in need of reshape the tran_inputs and train_outputs. And there is a look_back variable that I can't figure it out what's its usage
    ##   reshape into X=t and Y=t+1

    #look_back = 1
    #trainX, trainY = create_dataset(train, look_back)
    #testX, testY = create_dataset(test, look_back)

    ##  reshape input to be [samples, time steps, features]

    #trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    #testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    ##  create and fit the LSTM network

    #model = Sequential()
    #model.add(LSTM(4, input_shape=(1, look_back)))
    #......
    

    model.add(Dense(64, activation='relu')) 
    model.add(Dense(32, activation='relu'))  
    model.add(Dense(5, activation='softmax', name='main_output'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.summary()


    return model


def trainModel(model, train_inputs, train_outputs, test_inputs, test_outputs):

    model.fit(train_inputs, train_outputs, epochs=10, batch_size=32, validation_data=(test_inputs, test_outputs) )
    model.save("simplemodel")

    return model



#raw_train_outputs needs to be in the raw form before changed to binary in the to_categorical function
def trainModelClassWeight(model, train_inputs, train_outputs, test_inputs, test_outputs, raw_train_outputs):

    class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes = np.unique(raw_train_outputs), y = raw_train_outputs)
    
    class_weights = dict(enumerate(class_weights))#don't know whether this is necessary or not
    callbacks = [keras.callbacks.ModelCheckpoint("best model at epoch:{epoch}.h5", save_best_only=True)]#don't know whether this is necessary or not

    model.fit(train_inputs, train_outputs, epochs = 40, batch_size = 32, class_weight=class_weights, 
    callbacks = callbacks, validation_data=(test_inputs, test_outputs), shuffle=True)
    #shuffle and validation data can be removed

    model.save("simplemodelweighted")

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