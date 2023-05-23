import warnings
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

import mlflow
import mlflow.sklearn

#Choose input file
folderPath = dbutils.widgets.get("sourceFolder")
fileName = dbutils.widgets.get("sourceFile")
input_path = "/dbfs/mnt/"+folderPath+"/"+fileName
df = pd.read_csv(input_path)

#Remove rows which do not have an outcome value
df = df.loc[~df['KWALITEITSNIVEAU'].isna(), :]

#Assign to X and y
y = df['KWALITEITSNIVEAU']
X = df.drop('KWALITEITSNIVEAU', axis=1)

#Automatically encode y to numeric values
le = LabelEncoder()
y = le.fit_transform(y)
y = pd.Series(y)

#Remove geometry columns
X.drop(["Shape_Length","Shape_Area","geometry"], inplace=True, axis=1, errors='ignore')

#drop ID column
X.drop('ID', inplace=True, axis=1, errors='ignore')

#Select columns to make dummies
columns_to_make_dummies = list()

for lab, val in X.dtypes[X.dtypes == 'object'].iteritems():
       columns_to_make_dummies.append(lab)

#drop columns with more than m unique values
print("Columns removed from columns_to_make_dummies:")
m = len(X.index) / 10 # 10%
for column in columns_to_make_dummies:
    if X[column].nunique() > m:
        columns_to_make_dummies.remove(column)
        X.drop(column, axis=1, inplace=True)
        print(column)

#Create dummies
print("Creating dummies...")
dummies = pd.get_dummies(X[columns_to_make_dummies])
X = X.drop(columns_to_make_dummies,axis=1)
X = pd.concat([X,dummies],axis=1)

#Create an instance of the feature selector for removal of missing
labels = X.columns.values
fs = FeatureSelector(data = X, labels = labels)
fs.identify_missing(missing_threshold=0.5)
X = fs.remove(methods = ['missing'])

#Create an instance to identify and remove columns with single unique value
labels = X.columns.values
fs = FeatureSelector(data = X, labels = labels)
fs.identify_single_unique()
single_unique = fs.ops['single_unique']
print(single_unique)
X = fs.remove(methods = ['single_unique'])

print("Replace the NaNs...")
X.fillna(value=X.mean(), inplace=True)

X = transformPositive(X)

#Feature importance model
print("Feature importance...")
model = ExtraTreesClassifier()
model.fit(X,y)

n = 20 # show x features in plot
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(n).plot(kind='barh') 
plt.show()

#Feature selection
#Filter the features with zero importance
df_features = pd.Series.to_frame(feat_importances)
df_features['id'] = list(df_features.index)

#Filter rows with zero importance
df_features_drop = df_features[df_features[0] == 0]
df_features_drop = df_features_drop['id']
feature_list = df_features_drop.values.tolist()
X = X.drop(feature_list, axis=1)

#Show shape of remaining features
print(X.shape)
