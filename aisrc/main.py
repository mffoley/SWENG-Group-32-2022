

#importing a function from the other file

from keras.utils.np_utils import to_categorical
from dataManipulation import readECGData, readECGDataLSTM,  frequencyClasses, resampleData, formatOutputs, formatInputs, calculateWeights, reshapeInputs
from LSTMModel import makeModelLSTM,trainModelClassWeight, trainModelLSTM, loadModel, evaluateModel

train, test = readECGDataLSTM()

#raw_train_outputs = train[187]

#train = resampleData(train)


train_inputs, test_inputs, train_outputs, test_outputs = reshapeInputs(train, test)

# #print(train_outputs[84271])
# #print(train_inputs[84271])

my_model = makeModelLSTM(train_inputs)

trained_model = trainModelLSTM(my_model, train_inputs, train_outputs, test_inputs, test_outputs)




        