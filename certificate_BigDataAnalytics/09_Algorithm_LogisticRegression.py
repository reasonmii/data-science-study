
# =================================================================
# sklearn.linear_model : Linear Models
#
# Logistic Regression
# 종속변수가 범주형 자료일 때 사용
# 원자료를 확률 -> odds -> log 변환하여 선형회귀모델을 적용한 모델
#
# 확률 : 3차식 모형 (복잡함)
# odd : 모델 범주가 0~무한대로 나타남
# log : 직선의 선형모형 -> 이해하기도 쉽고 분석도 명쾌함
#
# 해석 : 지수(exp)로 변환하여 원데이터로 전환
#
# Hyper Parameter
# - C (default : C=1)
#   작을수록 모델이 단순해지고, 클수록 모델이 복잡해짐
#   로그스케일 (0.01, 0.1, 1, 10, 100 등) 단위로 최적치 탐색 권고
# - solver
#   데이터 양에 따른 연산속도와 관련된 주요 하이퍼파라미터
#   데이터양이 수백 ~ 수십만 건인데 full-batch로 할 경우 시간 오래 걸림
#   'solver=seg' 설정
#    -> 평균경사하강법 (Stochastic Average Gradient Descent) 적용
#       빠른 속도 가능
# =================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('breast-cancer-wisconsin.csv', encoding='utf-8')

data.head()
data.describe()
data.info()

X = data[data.columns[1:10]]
y = data[['Class']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.97265625

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# 훈련데이터 오차행렬:
#  [[328   5]
#  [  9 170]]

from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.97      0.98      0.98       333
#            1       0.97      0.95      0.96       179
#
#     accuracy                           0.97       512
#    macro avg       0.97      0.97      0.97       512
# weighted avg       0.97      0.97      0.97       512

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.9590643274853801

confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)
# 테스트데이터 오차행렬:
#  [[106   5]
#  [  2  58]]

from sklearn.metrics import classification_report
cfreport_test = classification_report(y_test, pred_test)
print("분류예측 레포트:\n", cfreport_test)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.98      0.95      0.97       111
#            1       0.92      0.97      0.94        60
#
#     accuracy                           0.96       171
#    macro avg       0.95      0.96      0.96       171
# weighted avg       0.96      0.96      0.96       171

from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'C': 10}
# Best Score: 0.9726
# Test set Score: 0.9591

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'C':randint(low=0.01, high=100)}
random_search = RandomizedSearchCV(LogisticRegression(),
                                   param_distributions=param_distribs,
                                   n_iter=100,
                                   cv=5)

random_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'C': 11}
# Best Score: 0.9745
# Test set Score: 0.9591
