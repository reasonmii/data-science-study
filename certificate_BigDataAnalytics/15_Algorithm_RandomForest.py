
# =================================================================
# Random Forest
# - 앙상블(Ensemble) 기법
#   의사결정나무 수십~수백 개가 예측한 분류/회귀값 평균낸 모델
# - 학습데이터를 "무작위로 샘플링"해서 다수의 의사결정 트리 분석
#   1) 데이터에서 부트스트래핑 과정을 통해 N개의 샘플링 데이터셋 생성
#   2) 각 샘플링된 데이터 셋에서 임의의 변수 선택
#      M개의 총 변수들 중에서 sqrt(M) 또는 M/3개
#   3) 의사결정트리 종합하여 앙상블 모델 생성
#      OOB error를 통해 오분류율 평가
#
# sklearn.ensemble
# 1) 분류 : RandomForestClassifier
#    - 기준 : gini
# 2) 회귀 : RandomForestRegressor
#    - 기준 : mse
#
# 핵심 파라미터
# - n_estimators : 나무의 수 (deafult 100)
#   의사결정나무 모델을 몇 개 구성할 것인지
# - max_features : 선택 변수(특성) 수
#   특성치를 얼마나 반영할 것인지
#   "auto" / "sqrt" : sqrt(n_features)
#   "log2" : log2(n_features)
#   none : n_features
# =================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# Classification
# =================================================================

data = pd.read_csv('breast-cancer-wisconsin.csv', encoding='utf-8')

data.info()
data.describe()
data.head()

X = data[data.columns[1:10]]
y = data[['Class']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 1.0

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.9649122807017544

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# 훈련데이터 오차행렬:
#  [[333   0]
#  [  0 179]]
 
from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00       333
#            1       1.00      1.00      1.00       179
#
#     accuracy                           1.00       512
#    macro avg       1.00      1.00      1.00       512
# weighted avg       1.00      1.00      1.00       512

confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)
# 테스트데이터 오차행렬:
#  [[106   5]
#  [  1  59]]
 
cfreport_test = classification_report(y_test, pred_test)
print("분류예측 레포트:\n", cfreport_test)
# 분류예측 레포트:
#                precision    recall  f1-score   support

#            0       0.99      0.95      0.97       111
#            1       0.92      0.98      0.95        60

#     accuracy                           0.96       171
#    macro avg       0.96      0.97      0.96       171
# weighted avg       0.97      0.96      0.97       171

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators':range(100,1000,100),
              'max_features':['auto', 'sqrt', 'log2']}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'max_features': 'auto', 'n_estimators': 200}
# Best Score: 0.9746
# Test set Score: 0.9649

# Random Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'n_estimators': randint(low=100,high=1000),
                  'max_features': ['auto', 'sqrt', 'log2']}
random_search = RandomizedSearchCV(RandomForestClassifier(),
                                   param_distributions=param_distribs,
                                   n_iter=20, cv=5)
random_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'max_features': 'sqrt', 'n_estimators': 839}
# Best Score: 0.9746
# Test set Score: 0.9649


# =================================================================
# Regression
# =================================================================

data = pd.read_csv('house_price.csv', encoding='utf-8')

data.info()
data.describe()
data.head()

X = data[data.columns[1:5]]
y = data[['house_value']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.9381138978714235

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.5834146378295427

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  23743.656942069807
# 테스트 데이터 RMSE:  61704.16459034068

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': range(100,500,100),
              'max_features': ['auto','sqrt','log2']}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'max_features': 'sqrt', 'n_estimators': 300}
# Best Score: 0.5684
# Test set Score: 0.5929

# Random Search
from sklearn.model_selection import RandomizedSearchCV
param_distribs = {'n_estimators': randint(low=100, high=500),
                  'max_features': ['auto','sqrt','log2']}
random_search = RandomizedSearchCV(RandomForestRegressor(),
                                   param_distributions = param_distribs,
                                   n_iter=20,
                                   cv=5)
random_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'max_features': 'sqrt', 'n_estimators': 414}
# Best Score: 0.5686
# Test set Score: 0.5934
