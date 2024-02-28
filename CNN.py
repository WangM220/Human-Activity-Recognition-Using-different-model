
"""import libraries"""
from tensorflow import keras

"""subclass the model class"""
class HAR_ModelCNN(keras.Model):
    def __init__(self):
        super(HAR_ModelCNN,self).__init__();
        self.first_cnn= keras.layers.Conv1D(64,3,activation='relu',name="First_CNN_Layer")
        self.first_pooling = keras.layers.AveragePooling1D(2,name="First_AveragePool_Layer")
        self.second_cnn = keras.layers.Conv1D(64,3,activation='relu',name="Second_CNN_Layer")
        self.second_pooling= keras.layers.AveragePooling1D(2,name ="Second_AveragePool_Layer")
        self.flatten = keras.layers.Flatten()
        self.first_dense = keras.layers.Dense(100, activation='relu', name="First_Dense_Layer")
        self.second_dense = keras.layers.Dense(6, activation= 'softmax', name = "Second_Dense_Layer")

    def call(self,inputs, training = False):
        x = self.first_cnn(inputs,training = training)
        x = self.first_pooling(x)
        x = self.second_cnn(x,training = training)
        x = self.second_pooling(x)
        x = self.flatten(x)
        x = self.first_dense(x,training = training)
        x = self.second_dense(x,training = training)

        return x
        
"""import libraries"""
from pandas import read_csv
from numpy import dstack
import numpy as np
from sklearn.preprocessing import LabelBinarizer

"""define Load_data"""
class Load_data:
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path

    def load_file(self, filepath):
        load_file= read_csv(filepath, header=None, sep='\s+') 
        return load_file.values 

    def load_tsdata(self,filename,filepath):
        data = list()
        for name in filename:
            newfilepath = filepath + name
            loaded_data = self.load_file(newfilepath)
            data.append(loaded_data)
        data = np.dstack(data)
        return data

    def load_groupdata(self, group):
        filenames = ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt',
                  'body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt',
                  'body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
        filepath = self.dataset_path + group + '/Inertial Signals/'
        X = self.load_tsdata(filenames,filepath)
        y = self.load_file(self.dataset_path + group + '/y_' + group+'.txt')
        return X,y

"""load train dataset and test dataset"""
import os
### get current work directory
current_dir = os.path.dirname(os.path.abspath(__file__))

data_loader = Load_data(current_dir+'/HARDataset/')
trainX, trainy = data_loader.load_groupdata('train')
testX, testy = data_loader.load_groupdata('test')

"""transform multiclass labels to binary labels"""
encoderdata= LabelBinarizer()
testy_encode = encoderdata.fit_transform(testy)
trainy_encode = encoderdata.fit_transform(trainy)
print(trainX.shape,trainy_encode.shape,testX.shape, testy_encode.shape)

"""fit the model and evaluate"""
model=HAR_ModelCNN()
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(trainX,trainy_encode,epochs=10, batch_size=32, verbose=1)
_, accuracy_train = model.evaluate(trainX, trainy_encode, batch_size=32, verbose=0)
_, accuracy_test = model.evaluate(testX, testy_encode, batch_size=32, verbose=0)

print(accuracy_train)
print(accuracy_test)



