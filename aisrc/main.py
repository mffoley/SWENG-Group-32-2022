

#importing a function from the other file
from keras.utils.np_utils import to_categorical
from dataManipulation import readECGData,  frequencyClasses, resampleData, formatOutputs, formatInputs
from simpleModel import makeModel, trainModel

train, test = readECGData()

train_inputs, test_inputs = formatInputs(train, test)

train_outputs, test_outputs = formatOutputs(train, test)

print(train_outputs[84271])
print(train_inputs[84271])

my_model = makeModel(train_inputs, train_outputs)

trained_model = trainModel(my_model, train_inputs, train_outputs, test_inputs, test_outputs)

        