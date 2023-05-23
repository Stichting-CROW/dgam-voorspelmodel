#Apply trained models and send updates to the webapp

import warnings
import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime

from packages.feature_selector import FeatureSelector
from packages.transformpositive import transformPositive

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

import pickle

import requests
import json

query = "select * from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME='train_data'"
sqldf = pd.read_sql(query, cnxn)

#Choose input file
folderPath = dbutils.widgets.get("sourceFolder")
fileName = dbutils.widgets.get("sourceFile")

folder, rhs = folderPath.split("-", 1)
url = "https://voorspelmodel-webapp.azurewebsites.net/progress/{0}"

req_prog10 = requests.post(url.format(folder), json={"prog": "10% ||  Choose input file"})
input_path = "/dbfs/mnt/"+folderPath+"/"+fileName
df = pd.read_csv(input_path)

#Assign to X and y
req_prog26 = requests.post(url.format(folder), json={"prog": "26% ||  Assign to X and y"})
X = df.copy()

#drop columns
column_list = sqldf["COLUMN_NAME"].tolist()
for column in X.columns:
    if column not in column_list:
        X.drop(column, axis=1, inplace=True)
        
#add missing columns
missing_columns = []
for column in column_list:
    if column not in X.columns:
        missing_columns.append(column)

req_missing_features = requests.post(url.format(folder), json=missing_columns)

add = X.columns.tolist() + missing_columns
X = X.reindex(columns=add, fill_value="")

X = X.drop('Kwaliteitsniveau', axis=1, errors='ignore')
X = X.drop('KWALITEITSNIVEAU', axis=1, errors='ignore')

if 'Oppervlakte' in missing_columns:
    X['Oppervlakte'] = pd.to_numeric(X['Oppervlakte'], downcast='float')
    X = X.assign(Oppervlakte=0)
if 'Aanlegjaar' in missing_columns:
    X['Aanlegjaar'] = pd.to_numeric(X['Aanlegjaar'], downcast='integer')
    X = X.assign(Aanlegjaar=0)
if X['Inspectiedatum'].dtypes != 'object':
    X['Inspectiedatum'] = X['Inspectiedatum'].astype(str, errors='ignore')

#Select columns to make dummies
req_prog48 = requests.post(url.format(folder), json={"prog": "48% ||  Select columns to make dummies"})
columns_to_make_dummies = list()

for lab, val in X.dtypes[X.dtypes == 'object'].iteritems():
       columns_to_make_dummies.append(lab)

#Create dummies
req_prog62= requests.post(url.format(folder), json={"prog": "62% ||  Encoding dummies"})
modelpath = "/dbfs/mnt/ml-models/one_hot_encode_model.sav"
enc = pickle.load(open(modelpath, 'rb'))

enc.transform(X[columns_to_make_dummies].astype(str))
dummies = pd.DataFrame(enc.transform(X[columns_to_make_dummies]).toarray())
dummies.columns = enc.get_feature_names(columns_to_make_dummies)

X = X.drop(columns_to_make_dummies,axis=1)
X = pd.concat([X.reset_index(drop=True),dummies.reset_index(drop=True)],axis=1)

#Create an instance of the feature selector for removal of missing
req_prog69= requests.post(url.format(folder), json={"prog": "69% ||  Create an instance of the feature selector for removal of missing"})
labels = X.columns.values
fs = FeatureSelector(data = X, labels = labels)
fs.identify_missing(missing_threshold=0.5)
X = fs.remove(methods = ['missing'])

req_prog73= requests.post(url.format(folder), json={"prog": "73% ||  Replace the NaNs"})

#Create an instance to identify and remove columns with single unique value
print("Replace the NaNs...")
X.fillna(value=X.mean(), inplace=True)
 
X = transformPositive(X)

#Feature importance
req_prog79= requests.post(url.format(folder), json={"prog": "79% ||  Feature importance"})
modelpath = "/dbfs/mnt/ml-models/feature_importance_model.sav"
model = pickle.load(open(modelpath, 'rb'))
model.predict(X)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

#Feature selection
req_prog85= requests.post(url.format(folder), json={"prog": "85% ||  Feature selection"})

#Filter the features with zero importance
req_prog88= requests.post(url.format(folder), json={"prog": "88% ||  Filter the features with zero importance"})
df_features = pd.Series.to_frame(feat_importances)
df_features['id'] = list(df_features.index)

#Filter rows with zero importance
req_prog90= requests.post(url.format(folder), json={"prog": "90% ||  Filter rows with zero importance"})
df_features_drop = df_features[df_features[0] == 0]
df_features_drop = df_features_drop['id']
feature_list = df_features_drop.values.tolist()
X = X.drop(feature_list, axis=1)

#Scale the data
req_prog93= requests.post(url.format(folder), json={"prog": "93% ||  Scale the data"})
modelpath = "/dbfs/mnt/ml-models/scaler_model.sav"
scaler = pickle.load(open(modelpath, 'rb'))

# Now apply the transformations to the data:
X_train = scaler.transform(X)

#Training the model
req_prog95= requests.post(url.format(folder), json={"prog": "95% || Load the model"})
modelpath = "/dbfs/mnt/ml-models/prediction_model.sav"
mlp = pickle.load(open(modelpath, 'rb'))

#Predictions and evaluation
req_prog98= requests.post(url.format(folder), json={"prog": "98% || Generate predictions"})
predictions = mlp.predict(X_train)
    
req_prog100= requests.post(url.format(folder), json={"prog": "100% || Succesful run"})

#Make add prediction to original df
modelpath = "/dbfs/mnt/ml-models/label_encoder_model.sav"
le = pickle.load(open(modelpath, 'rb'))
df['Voorspelling'] = le.inverse_transform(predictions)

#Save df as csv
date = datetime.today().strftime('%d-%m-%Y')
output_path = "/dbfs/mnt/"+folder+"-output/"+"prediction_"+date+"_"+fileName
df.to_csv(output_path, index=False)
