#Stage 1, preparing the data
import numpy as np 
import pandas as pd

from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.layers import Convolution1D, Model
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import BatchNormalization, MaxPool1D
from tensorflow.python.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical 
from sklearn.utils import resample
import os


#
#Function that loads in training and testing data
#
def readECGData ():
    
    current_directory = os.getcwd()
    print(current_directory)    


    train = pd.read_csv(current_directory + '/data/archive/mitbih_train.csv', header=None)
    test = pd.read_csv(current_directory + '/data/archive/mitbih_test.csv', header=None)

    return train, test

train, test = readECGData()

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

    result = pd.concat([data1up,data2up,data3up,data4up])

    return result


def network(X_train,y_train,X_test,y_test):
    

    im_shape=(X_train.shape[1],1)
    #inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    #conv1_1=Convolution1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    #conv1_1=BatchNormalization()(conv1_1)
    #pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    #
    #conv2_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
    #conv2_1=BatchNormalization()(conv2_1)
    #pool2=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
    #
    #conv3_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool2)
    #conv3_1=BatchNormalization()(conv3_1)
    #pool3=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
    #
    #flatten=Flatten()(pool3)
    #dense_end1 = Dense(64, activation='relu')(flatten)
    #dense_end2 = Dense(32, activation='relu')(dense_end1)
    #main_output = Dense(5, activation='softmax', name='main_output')(dense_end2)
    #
    #
    #model = Model(inputs= inputs_cnn, outputs=main_output)
    #model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    #
    #
    #callbacks = [EarlyStopping(monitor='val_loss', patience=8), ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    #
    #history=model.fit(X_train, y_train,epochs=40,callbacks=callbacks, batch_size=32,validation_data=(X_test,y_test))
    #model.load_weights('best_model.h5')
    #return(model,history)

    #create model
    model = Sequential()

    #add model layers
    model.add(Convolution1D(64, activation='relu', input_shape=im_shape))               #inputs_cnn
    model.add(BatchNormalization(input_shape=im_shape))
    model.add(MaxPool1D(pool_size=(3), strides=(2), padding="same"))

    model.add(Convolution1D(64, activation='relu', input_shape=im_shape))               #pool1
    model.add(BatchNormalization(input_shape=im_shape))
    model.add(MaxPool1D(pool_size=(2), strides=(2), padding="same"))

    model.add(Convolution1D(64, activation='relu', input_shape=im_shape))               #pool2
    model.add(BatchNormalization(input_shape=im_shape))
    model.add(MaxPool1D(pool_size=(2), strides=(2), padding="same"))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))                                             #flatten
    model.add(Dense(32, activation='relu'))                                             #dense_end1
    model.add(Dense(5, activation='softmax', name='main_output'))

    ##Defining the model by specifying the input and output layers
    model = Model(inputs=X_train, outputs=y_train)

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    #train the model
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))

    #predict first 4 images in the test set
    model.predict(X_test[:4])




#function that returns frequencies of all classes from a dataset
def frequencyClasses (data):
    data[187]=data[187].astype(int)
    equilibre=data[187].value_counts()
    print(equilibre)


#function that print train outputs
def trainOutputs (data):
    train_outputs = to_categorical(train[187])
    print(train_outputs)


#function that print train inputs
def trainInputs (data):
    train_inputs = train.iloc[:,:186].values
    print(train_inputs[85427])    

