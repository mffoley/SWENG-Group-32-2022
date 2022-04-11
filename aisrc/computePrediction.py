CNN_MODEL_ID = 1
RESNET_MODEL_ID = 2
LSTM_MODEL_ID = 3

from cgi import test
import keras
import numpy as np
from dataManipulation import formatInput


def computePrediction (model_id, testing_data):

    testing_data = testing_data.iloc[:,:187].values #ensure that there aren't any extra columns
    testing_data = testing_data.reshape(len(testing_data), testing_data.shape[1],1)

    if(model_id == CNN_MODEL_ID):
        
        print(testing_data.shape)

        model = keras.models.load_model("simplemodel")
        predictions = model.predict(testing_data)

        return predictions

        print(predictions[:,:50])

    elif (model_id == RESNET_MODEL_ID):

        print(testing_data.shape)

        model = keras.models.load_model("ResNetmodel")
        predictions = model.predict(testing_data) 

        return predictions

        print(predictions[:,:50])
        
    elif (model_id == LSTM_MODEL_ID):

        print(testing_data.shape)

        model = keras.models.load_model("LSTMmodel")
        predictions = model.predict(testing_data)

        return predictions

        print(predictions[:,:50])

