#Stage 1, preparing the data
import numpy as np 
import pandas as pd 
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
from keras.utils.np_utils import to_categorical 
import os


#
#Function that loads in training and testing data
#
def readECGData ():
    
    current_directory = os.getcwd()
    print(current_directory)    


    train = pd.read_csv(current_directory + '/data/mitbih_train.csv', header=None)
    test = pd.read_csv(current_directory + '/data/mitbih_test.csv', header=None)

    return train, test


def resampleData (data):
    
    data0 = (data[data[187]==0]).sample(n=20000,random_state=420)
    data1 = data[data[187]==1]
    data2 = data[data[187]==2]
    data3 = data[data[187]==3]
    data4 = data[data[187]==4]

    data1up = resample(data1,replace=True,n_samples=20000,random_state=10)
    data2up = resample(data2,replace=True,n_samples=20000,random_state=11)
    data3up = resample(data3,replace=True,n_samples=20000,random_state=12)
    data4up = resample(data4,replace=True,n_samples=20000,random_state=13)

    result = pd.concat([data0,data1up,data2up,data3up,data4up])
    
    return result

#function that calculates the class weighting for the weighted model, takes in imbalanced training data
def calculateWeights(trainingData):

    totalSize = trainingData.shape[0]
    freq = frequencyClasses(trainingData)

    weight0 = totalSize/(5*freq[0])
    weight1 = totalSize/(5*freq[1])
    weight2 = totalSize/(5*freq[2])
    weight3 = totalSize/(5*freq[3])
    weight4 = totalSize/(5*freq[4])

    classWeights = {"0":weight0,"1":weight1,"2":weight2,"3":weight3,"4":weight4}

    print(classWeights)

    return classWeights




#function that returns frequencies of all classes from a dataset
def frequencyClasses (data):
    data[187]=data[187].astype(int)
    classcounts=data[187].value_counts()
    
    return classcounts


#function that returns formatted training and testing outputs
def formatOutputs (train, test):
    train_outputs = to_categorical(train[187])
    test_outputs = to_categorical(test[187])

    return train_outputs, test_outputs
    #target_train=result[187]
    #target_test=result[187]
    #y_train=to_categorical(target_train)
    #y_test=to_categorical(target_test)
    

#function that returns formatted training and testing inputs
def formatInputs (train, test):
    train_inputs = train.iloc[:,:186].values
    test_inputs = test.iloc[:,:186].values
    #X_train=result.iloc[:,:186].values
    #X_test=result.iloc[:,:186].values

    return train_inputs, test_inputs