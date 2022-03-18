

#importing a function from the other file
from keras.utils.np_utils import to_categorical

from dataManipulation import readECGData 
from dataManipulation import frequencyClasses as getFrequency
from dataManipulation import resampleData

train, test = readECGData()

#format for accessing pandas dataframes
#dataframes = var[column][row]
for column in train:
    print(train[column][86789]) # prints all values in row 1pyth

#model([train_inputs, train_outputs],[test_inputs, test_outputs])
#separate training input and output for model
    # train_outputs = to_categorical(train[187])
    # train_inputs = train.iloc[:,:186].values

    # print(train_outputs)
    # print(train_inputs[85427])



#below does the resampling for imbalanced data and prints frequencies
    #resampleData(train)

    #getFrequency(train)



        