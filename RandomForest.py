"""Random Forest Model"""


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

"""scale date set"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
trainX_scaled = scaler.fit_transform(trainX)
testX_scaled = scaler.fit_transform(testX)

"""encode the labels"""
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
trainy_encode = encoder.fit_transform(trainy)
testy_encode =encoder.fit_transform(testy)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


model_rf = RandomForestClassifier()
model_rf.fit(trainX_scaled, trainy_encode)
trainX_pred = model_rf.predict(trainX_scaled)

print(confusion_matrix(trainX_pred, trainy_encode))
print(classification_report(trainX_pred, trainy_encode))

testX_pred = model_rf.predict(testX_scaled)
print(confusion_matrix(testX_pred, testy_encode))
print(classification_report(testX_pred, testy_encode))

model_rf.get_params

"""Cross-Validation"""

from sklearn.model_selection import GridSearchCV

rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [100, 200, 500],
    'max_depth' : [2,3,4,5]}

model_rf_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
model_rf_cv.fit(trainX_scaled, trainy_encode)

model_rf_cv.best_params_
model_rf_cv.get_params

final_model = model_rf_cv.best_estimator_
testy_pred = final_model.predict(testX_scaled)

print(confusion_matrix(testy_pred,testy_encode))
print("\n")
print(classification_report(testy_pred,testy_encode))