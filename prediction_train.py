#Train models and save them

import warnings
import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime

from packages.feature_selector import FeatureSelector
from packages.transformpositive import transformPositive

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

import pickle

#Select input data
query = "SELECT * FROM train_data;"
df = pd.read_sql(query, cnxn)
df.shape

#Remove rows which do not have an outcome value
df = df.loc[~df['Kwaliteitsniveau'].isna(), :]
df = df.loc[df['Kwaliteitsniveau'] != '-']

#Assign to X and y
y = df['Kwaliteitsniveau']
X = df.drop('Kwaliteitsniveau', axis=1)

#Automatically encode y to numeric values
le = LabelEncoder()
y = le.fit_transform(y)
y = pd.Series(y)

#Save trained label encoder
modelpath = "/dbfs/mnt/ml-models/label_encoder_model.sav"
pickle.dump(le, open(modelpath, 'wb'))

#Select columns to make dummies
columns_to_make_dummies = list()

for lab, val in X.dtypes[X.dtypes == 'object'].iteritems():
       columns_to_make_dummies.append(lab)

#Create dummies
print("Creating dummies...")
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X[columns_to_make_dummies].astype(str))

#Save trained dummies encoder
modelpath = "/dbfs/mnt/ml-models/one_hot_encode_model.sav"
pickle.dump(enc, open(modelpath, 'wb'))

dummies = pd.DataFrame(enc.transform(X[columns_to_make_dummies]).toarray())
dummies.columns = enc.get_feature_names(columns_to_make_dummies)

X = X.drop(columns_to_make_dummies,axis=1)
X = pd.concat([X.reset_index(drop=True),dummies.reset_index(drop=True)],axis=1)

#Create an instance to identify and remove columns with single unique value
print("Replace the NaNs...")
X.fillna(value=X.mean(), inplace=True)

X = transformPositive(X)

#Feature importance
print("Feature importance...")
model = ExtraTreesClassifier()
model.fit(X,y)

#Save trained feature importance model
modelpath = "/dbfs/mnt/ml-models/feature_importance_model.sav"
pickle.dump(model, open(modelpath, 'wb'))

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
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

#Partition the data
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Scale the data
print("Scale the data...")
scaler = StandardScaler()
scaler.fit(X_train)

#Save trained scaler model
modelpath = "/dbfs/mnt/ml-models/scaler_model.sav"
pickle.dump(scaler, open(modelpath, 'wb'))

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Training the model
print("Train the model...")
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)

#Save the trained prediction model
modelpath = "/dbfs/mnt/ml-models/prediction_model.sav"
pickle.dump(mlp, open(modelpath, 'wb'))

#Predictions and evaluation
print("Generate predictions...")
y_pred = mlp.predict(X_test_scaled)

print(confusion_matrix(y_test, y_pred))

warnings.filterwarnings('ignore')
print(classification_report(y_test, y_pred))
