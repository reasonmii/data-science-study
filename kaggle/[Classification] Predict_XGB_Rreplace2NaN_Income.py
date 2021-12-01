'''
성인 인구조사 소득 예측
age: 나이
workclass: 고용 형태
fnlwgt: 사람의 대표성을 나타내는 가중치(final weight)
education: 교육 수준
education.num: 교육 수준 수치
marital.status: 결혼 상태
occupation: 업종
relationship: 가족 관계
race: 인종
sex: 성별
capital.gain: 양도 소득
capital.loss: 양도 손실
hours.per.week: 주당 근무 시간
native.country: 국적
income: 수익 (예측해야 하는 값)

dataset : adult.csv
https://www.kaggle.com/uciml/adult-census-income
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6858
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/adult-census-income/adult.csv')

df.head()


# 이상치 처리 ==================================================================

# ★ 물음표를 nan 값으로 변경
df[df == '?'] = np.nan

# print(df.isnull().sum())
# print(df['workclass'].value_counts())
# print(df['occupation'].value_counts())
# print(df['native.country'].value_counts())

# 결측치는 최빈값과 차이가 크면 최빈값으로 값이 비슷하면 별도의 값으로 대체
df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
df['occupation'] = df['occupation'].fillna('null')
df['native.country'] = df['native.country'].fillna(df['native.country'].mode()[0])


# Scale data ==================================================================
print(df.head())
print(df.info())

X = df.drop('income', axis=1)
y = df[['income']]

from sklearn.preprocessing import MinMaxScaler
cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

for col in cols:
    scaler = MinMaxScaler()
    scaler.fit(X[[col]])
    X[[col]] = scaler.transform(X[[col]])


# Label Encoder ===============================================================
from sklearn.preprocessing import LabelEncoder
cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

for col in cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])


# Split data ==================================================================
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# Modeling ====================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

rf = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)
xgb = XGBClassifier(random_state=42)

models = [rf, svc, xgb]

for model in models:
    name = model.__class__.__name__
    model.fit(X_tr, y_tr)
    print(name, " : ", model.score(X_tr, y_tr), model.score(X_te, y_te))

# RandomForestClassifier  :  1.0 0.8541378780899739
# SVC  :  0.7591753685503686 0.7592507293106096
# XGBClassifier  :  0.9079007985257985 0.8654997696913864


# Tuning ======================================================================

params = {'random_state': [42],
          'max_depth': [3, 5, 7],
          'n_estimators': [50, 100]}

from sklearn.model_selection import GridSearchCV
search = GridSearchCV(xgb, param_grid = params, cv=5)
search.fit(X_tr, y_tr)
print("Best Param : {}".format(search.best_params_))
print("Best Score : {}".format(search.best_score_))
# Best Param : {'max_depth': 5, 'n_estimators': 50, 'random_state': 42}
# Best Score : 0.8726963261946233


# Final =======================================================================
final = XGBClassifier(random_state=42, max_depth=5, n_estimators=50)
final.fit(X_tr, y_tr)
print(final.score(X_tr, y_tr), final.score(X_te, y_te))
# 0.887323402948403 0.8673422385997236

pred = final.predict(X_te)
output = pd.DataFrame({'age': X_te['age'], 'income': pred})
output.to_csv('adult_rst.csv', index=False)





