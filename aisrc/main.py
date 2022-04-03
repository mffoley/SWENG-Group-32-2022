

#importing a function from the other file

from keras.utils.np_utils import to_categorical
from dataManipulation import readECGData, frequencyClasses, resampleData, formatOutputs, formatInputs, calculateWeights, reshapeInputsLSTM
from LSTMModel import makeModelLSTM,trainModelClassWeight, trainModelLSTM, loadModel, evaluateModel

train, test = readECGData()

print(train.loc[[77188]])

train = resampleData(train)

train_inputs, test_inputs = formatInputs(train, test) 

train_outputs, test_outputs = formatOutputs(train, test)

train_inputs_LSTM, test_inputs_LSTM = reshapeInputsLSTM(train_inputs, test_inputs)




#print(train_outputs[84271])
#print(train_inputs[84271])

my_model = makeModelLSTM(train_inputs)

trained_model = trainModelLSTM(my_model, train_inputs_LSTM, train_outputs, test_inputs_LSTM, test_outputs)




        