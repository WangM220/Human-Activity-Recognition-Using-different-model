"""import libraries"""
import numpy as np
import pandas as pd

"""load train dataset and test dataset"""
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

train = pd.read_csv(current_dir +'/HARDataset02/train.csv')
test = pd.read_csv(current_dir +'/HARDataset02/test.csv')

trainX = pd.DataFrame(train.drop(['Activity','subject'],axis=1))
trainy = pd.DataFrame(train.Activity)

testX = pd.DataFrame(test.drop(['Activity','subject'],axis=1))
testy = pd.DataFrame(test.Activity)

print(trainX.shape, trainy.shape, testX.shape, testy.shape)

"""transform multiclass labels to binary labels"""
# from sklearn.preprocessing import LabelBinarizer
# encoder = LabelBinarizer()
# trainy_encode = encoder.fit_transform(trainy)
# testy_encode =encoder.fit_transform(testy)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
trainy_encode = encoder.fit_transform(trainy)
testy_encode =encoder.fit_transform(testy)



"""fit SVM model-OneVsOne"""
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

model_onevsone = OneVsOneClassifier(LinearSVC(dual=True,max_iter=1000))
model_onevsone.fit(trainX, trainy_encode)
testy_pred = model_onevsone.predict(testX)
trainy_pred= model_onevsone.predict(trainX)

"""evaluate the model"""
trainy_pred_decode = pd.DataFrame(encoder.inverse_transform(trainy_pred))
testy_pred_decode= pd.DataFrame(encoder.inverse_transform(testy_pred))

results = confusion_matrix(testy_pred_decode,testy)
print(results)

print(classification_report(testy_pred_decode,testy))
print(classification_report(trainy_pred_decode,trainy))


"""fit SVM model-OneVsRest"""
from sklearn.multiclass import OneVsRestClassifier

model_onevsrest = OneVsRestClassifier(LinearSVC()).fit(trainX, trainy_encode) #.predict(X_train)

testy_pred_onevsrest = model_onevsrest.predict(testX)
trainy_pred_onevsrest= model_onevsrest.predict(trainX)

"""evaluate the model"""
trainy_pred_decode_onevsrest = pd.DataFrame(encoder.inverse_transform(trainy_pred_onevsrest))
testy_pred_decode_onevsrest= pd.DataFrame(encoder.inverse_transform(testy_pred_onevsrest))

results_onevsrest = confusion_matrix(testy_pred_decode_onevsrest,testy)
print(results_onevsrest)

print(classification_report(trainy_pred_decode_onevsrest,trainy))
print(classification_report(testy_pred_decode_onevsrest,testy))



