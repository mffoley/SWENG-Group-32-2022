

#importing a function from the other file

import keras
import pandas as pd 
from io import StringIO
from csv import reader
from simpleModel import makeModel, trainModel, trainModelClassWeight
from computePrediction import computePrediction
from keras.utils.np_utils import to_categorical
from dataManipulation import readECGData, readRawECGData, frequencyClasses, resampleData, formatOutputs, formatInput, calculateWeights, reshapeInputs, addGaussianNoise
from LSTMModel import makeModelLSTM, trainModelLSTM


def main(raw,num):

  #This is a temporary data read, will be done through UI instead
  data = stringToCSV(raw)

  #passes testing inputs and model_id
  return computePrediction (num, data)

def stringToCSV(data):
    test_data = pd.read_csv(StringIO(data), header=None)
    print("\n\n\nFORMATTED DATA\n\n")
    print(test_data)
    return test_data




# #print(train_outputs[84271])
# #print(train_inputs[84271])

# my_model = makeModelLSTM(train_inputs)

# trained_model = trainModelLSTM(my_model, train_inputs_LSTM, train_outputs, test_inputs_LSTM, test_outputs)




        