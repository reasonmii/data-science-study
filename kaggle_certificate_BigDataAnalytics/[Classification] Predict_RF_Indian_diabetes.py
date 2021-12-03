
'''
당뇨병 여부 판단
이상치 처리 (Glucose, BloodPressure, SkinThickness, Insulin, BMI가 0인 값)

dataset : diabetes.csv
https://www.kaggle.com/uciml/pima-indians-diabetes-database
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6770
'''

import pandas as pd
import numpy as np

df = pd.read_csv('diabetes.csv')
df.head()
df.info()
df.describe()


# 이상치 처리 ==================================================================
print(df.isnull().sum())

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols:
    print(col, " : ", len(df[df[col] == 0]))

for col in cols:
    df[col] = df[col].replace(0, df[col].mean())


# Split data ==================================================================
X = df.drop(columns=['Outcome'])
y = df['Outcome']

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# print(X_tr.shape, y_tr.shape)
# print(X_te.shape, y_te.shape)


# Min Max Scaler ==============================================================
from sklearn.preprocessing import MinMaxScaler

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

for col in cols:
    scaler = MinMaxScaler()
    scaler.fit(X_tr[[col]])
    X_tr[[col]] = scaler.transform(X_tr[[col]])
    X_te[[col]] = scaler.transform(X_te[[col]])

# print(X_tr.head())


# Modeling ====================================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)
nb = GaussianNB()

models = [dt, rf, svc, nb]

for model in models:
    name = model.__class__.__name__
    model.fit(X_tr, y_tr)
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    print(name)
    print('Train : ', model.score(X_tr, y_tr))
    print('Test : ', model.score(X_te, y_te))

# DecisionTreeClassifier
# Train :  1.0
# Test :  0.6753246753246753
# RandomForestClassifier
# Train :  1.0
# Test :  0.7402597402597403
# SVC
# Train :  0.8224755700325733
# Test :  0.7337662337662337
# GaussianNB
# Train :  0.752442996742671
# Test :  0.7012987012987013


# Tuning ======================================================================
# print(help(RandomForestClassifier))

params = {'n_estimators':[30, 50, 100],
         'max_features':['auto', 'sqrt', 'log2']}

from sklearn.model_selection import GridSearchCV
rf_search = GridSearchCV(rf, param_grid = params, cv=5, n_jobs=-1)
rf_search.fit(X_tr, y_tr)
print('Best Param : ', rf_search.best_params_)
print('Best Score : ', rf_search.best_score_)
# Best Param :  {'max_features': 'auto', 'n_estimators': 30}
# Best Score :  0.7720378515260563


# Final =======================================================================

model = RandomForestClassifier(random_state=42, n_estimators=30, max_features='auto')
model.fit(X_tr, y_tr)
print(model.score(X_tr, y_tr), model.score(X_te, y_te))
# 0.998371335504886 0.7532467532467533

from sklearn.metrics import confusion_matrix
pred_tr = model.predict(X_tr)
pred_te = model.predict(X_te)
cf_tr = confusion_matrix(y_tr, pred_tr)
cf_te = confusion_matrix(y_te, pred_te)
print(cf_tr)
print(cf_te)
# [[400   0]
#  [  1 213]]
# [[84 16]
#  [22 32]]


# 결과제출 =====================================================================

output = pd.DataFrame({'idx':y_te.index, 'Outcome':pred_te})
output.head()

output.to_csv('diabetes_rst.csv', index=False)




