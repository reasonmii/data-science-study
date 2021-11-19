
# =================================================================
# Support Vector Machine
# - 뛰어난 성능으로 활용도가 높은 분류 모델
# - 데이터 크기가 중간 이하, 여러 변수를 기준으로 분류하는 복잡한 과제에 적합
# - 레이블 범주를 선형적으로 구분하거나 비선형적으로 분류하는
#   선/초평면을 찾는 것이 핵심
# - 완벽한 분류는 어렵기 때문에 어느 정도 오류 허용
#
# 핵심문제 : "어떻게 집단을 분류하는 선을 긋는 것이 최선일까"
# 두 집단을 가장 멀리 떨어뜨리는 (확실하게 분리하는) 선
# => 여분(margin)을 최대화 해야 함
#    margin : 두 집단의 떨어짐 정도
#
# Support Vector
# 경계선에 가장 가까이 있는 각 카테고리(class)의 데이터(점)
# 
# soft margin
# - 잘못 분류된 데이터를 본래 속하는 카테고리로 비용을 들어 이동시킴
# - 파라미터 값 조절
#   값이 클수록 margin 폭이 좁아져 margin error가 작아지나 overfitting
#   값이 작을수록 margin 폭이 커져 margin error가 커짐
# 
# sklearn.svm : Support Vector Machine
# 다른 알고리즘에 비해 kernel 종류, C, gamma 등 하이퍼파라미터 다양
# 그만큼 모델의 "유연성"이 뛰어남
# 다만 이에 대한 이해가 깊어야 모델을 유연하게 다룰 수 있음
# 단점 : 대규모 데이터에는 느린 학습속도
# 1) 분류 : SVC (Support Vector Classification)
# 2) 회귀 : SVR (Support Vector Regression)
#           - kernel에 민감 (5가지 중 잘 맞는 것 선택이 중요)
#
# 주요 하이퍼파리미터
# - C : default = 1
#   분류에만 존재하는 하이퍼파라미터
#   작을수록 모델이 단순, 클수록 모델이 복잡
#   로그스케일(0.01, 0.1, 1, 10, 100) 단위로 최적치 탐색 권고
# - kernel : default = rbf
#   집단을 분류하거나 값을 예측(회귀)하는 함수에 대한 정의
#   Other options : linear, poly, sigmoid, precomputed
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

from sklearn.svm import SVC
model = SVC()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.984375

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# 훈련데이터 오차행렬:
#  [[329   4]
#  [  4 175]]

from sklearn.metrics import classification_report
cfreprot_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreprot_train)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.99      0.99      0.99       333
#            1       0.98      0.98      0.98       179
#
#     accuracy                           0.98       512
#    macro avg       0.98      0.98      0.98       512
# weighted avg       0.98      0.98      0.98       512

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)

confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)
# 테스트데이터 오차행렬:
#  [[106   5]
#  [  1  59]]
 
cfreport_test = classification_report(y_test, pred_test)
print("분류예측 레포트:\n", cfreport_test)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.99      0.95      0.97       111
#            1       0.92      0.98      0.95        60
#
#     accuracy                           0.96       171
#    macro avg       0.96      0.97      0.96       171
# weighted avg       0.97      0.96      0.97       171

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = [{'kernel': ['rbf'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel': ['linear'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}]
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
# Best Score: 0.9746
# Test set Score: 0.9591

# Random Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = [{'kernel':['rbf'],
                   'C': randint(low=0.001, high=100),
                   'gamma': randint(low=0.001, high=100)},
                  {'kernel':['linear'],
                   'C': randint(low=0.001, high=100),
                   'gamma': randint(low=0.001, high=100)}]
random_search = RandomizedSearchCV(SVC(),
                                   param_distributions = param_distribs,
                                   n_iter=100,
                                   cv=5)
random_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'C': 5, 'gamma': 91, 'kernel': 'linear'}
# Best Score: 0.9765
# Test set Score: 0.9591


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

from sklearn.svm import SVR
model = SVR(kernel='poly')
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.45177025582209707

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.45177025582209707

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련데이터 RSME: ", np.sqrt(MSE_train))
print("테스트데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련데이터 RSME:  70669.55248802518
# 테스트데이터 RMSE:  69600.08964461758

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'kernel': ['poly'],
              'C': [0.01, 0.1, 1, 10],
              'gamma': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(SVR(kernel='poly'), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'C': 10, 'gamma': 10, 'kernel': 'poly'}
# Best Score: 0.4888
# Test set Score: 0.5092

# Random Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'kernel': ['poly'],
                  'C': randint(low=0.01, high=10),
                  'gamma': randint(low=0.01, high=10)}
random_search = RandomizedSearchCV(SVR(kernel='poly'),
                                   param_distributions=param_distribs,
                                   n_iter=20,
                                   cv=5)
random_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'C': 7, 'gamma': 9, 'kernel': 'poly'}
# Best Score: 0.4682
# Test set Score: 0.4922
