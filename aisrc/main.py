

#importing a function from the other file

import keras
from simpleModel import makeModel, trainModel, trainModelClassWeight
from computePrediction import computePrediction
from keras.utils.np_utils import to_categorical
from dataManipulation import readECGData, readRawECGData, frequencyClasses, resampleData, formatOutputs, formatInput, calculateWeights, reshapeInputs, addGaussianNoise
from LSTMModel import makeModelLSTM, trainModelLSTM


#This is a temporary data read, will be done through UI instead
test = readRawECGData()

#passes testing inputs and model_id
computePrediction (2, test)






# #print(train_outputs[84271])
# #print(train_inputs[84271])

# my_model = makeModelLSTM(train_inputs)

# trained_model = trainModelLSTM(my_model, train_inputs_LSTM, train_outputs, test_inputs_LSTM, test_outputs)




        