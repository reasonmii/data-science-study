
# =================================================================
# Naive Bayes
# 사건 B가 주어졌을 때, 사건 A가 발생할 확률 P(A|B)
# 조건부 확률과 베이즈 정리를 이용한 알고리즘
#
# Naive : 예측에 사용되는 X가 상호 독립적이라는 가정 하에 확률 계산
# 즉, 모든 특성치(X)가 레이블 분류/예측에 동등한 역할을 한다는 의미
#
# sklearn.naive_bayes : Naive_Bayes
# 분류 문제)
# 주로 가우시안 나이브베이즈 (GaussianNB) 알고리즘 사용
# - 정규분포상에서 발생확률 계산
# - 특성치 중 연속형 자료일 경우 발생확률을 정규분포상에서의
#   확률(likelihood, 우도)을 구해서 계산하기 때문
#
# GaussianNB의 주요 하이퍼파라미터 : var_smoothing
# default 0.000000001
# 안정적 연산을 위해 분산에 더해지는 모든 특성치의 최대 분산 비율
#
# sklearn.linear_model.BayesianRidge
# 회귀 문제)
# linear_model의 Baysian regressors 사용
# 그 중 BayesianRidge 모델이 가장 적합
# ※ 회귀 문제는 sklearn의 naive_bayes 알고리즘과 잘 맞지 않음
# ※ BayesianRidge는 회귀모델에서 자주 사용하는 알고리즘은 아님
#
# BayesianRidge의 하이퍼파라미터
# 감마분포에서 사전 파라미터값 결정
# - alpha_1 : default 1e-6
#             감마분포의 alpha parameter 사전 설정
# - lambda_1 : default 1e-6
#              감마분포의 lambda parameter 사전 설정
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

data.head()
data.info()
data.describe()

X = data[data.columns[1:10]]
y = data[['Class']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬: \n", confusion_train)
# 훈련데이터 오차행렬: 
#  [[319  14]
#  [  3 176]]

from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트: \n", cfreport_train)
# 분류예측 레포트: 
#                precision    recall  f1-score   support
#
#            0       0.99      0.96      0.97       333
#            1       0.93      0.98      0.95       179
#
#     accuracy                           0.97       512
#    macro avg       0.96      0.97      0.96       512
# weighted avg       0.97      0.97      0.97       512

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.9590643274853801

confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬: \n", confusion_test)
# 테스트데이터 오차행렬: 
#  [[106   5]
#  [  2  58]]

cfreport_test = classification_report(y_test, pred_test)
print("분류예측 레포트: \n", cfreport_test)
# 분류예측 레포트: 
#                precision    recall  f1-score   support
#
#            0       0.98      0.95      0.97       111
#            1       0.92      0.97      0.94        60
#
#     accuracy                           0.96       171
#    macro avg       0.95      0.96      0.96       171
# weighted avg       0.96      0.96      0.96       171

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'var_smoothing': [0,1,2,3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'var_smoothing': 0}
# Best Score: 0.9649
# Test set Score: 0.9591

# Random Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'var_smoothing': randint(low=0, high=20)}
random_search = RandomizedSearchCV(GaussianNB(), param_distribs, n_iter=100, cv=5)
random_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'var_smoothing': 0}
# Best Score: 0.9649
# Test set Score: 0.9591


# =================================================================
# Regression
# =================================================================

data = pd.read_csv('house_price.csv', encoding='utf-8')

data.head()
data.info()
data.describe()

X = data[data.columns[1:5]]
y = data[['house_value']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

from sklearn.linear_model import BayesianRidge
model = BayesianRidge()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.5455724466331762

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.5626859871488648

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  64340.343029485426
# 테스트 데이터 RMSE:  63220.68115643447

# Grid Search
param_grid = {'alpha_1': [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 2, 3, 4],
              'lambda_1': [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 2, 3, 4]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(BayesianRidge(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'alpha_1': 4, 'lambda_1': 1e-06}
# Best Score: 0.5452
# Test set Score: 0.5627

# Random Search
param_distribs = {'alpha_1': randint(low=1e-06, high=10),
                  'lambda_1': randint(low=1e-06, high=10)}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(BayesianRidge(),
                                   param_distributions=param_distribs,
                                   n_iter=50,
                                   cv=5)
random_search.fit(X_scaled_train, y_train)

print("Best Parmaeter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parmaeter: {'alpha_1': 6, 'lambda_1': 0}
# Best Score: 0.5452
# Test set Score: 0.5627

