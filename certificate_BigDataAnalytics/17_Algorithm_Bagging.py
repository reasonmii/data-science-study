
# =================================================================
# Bagging
# - 여러 개의 부트스트랩(bootstrap) 데이터 생성
#   각 부트스트랩 데이터에 하나 이상 알고리즘 학습
#   산출된 결과 중 투표(Voting) 방식에 의해 최종 결과 선정
#
# sklearn.ensemble
# 1) 분류 : BaggingClassifier
# 2) 회귀 : BaggingRegressor
#
# 핵심 하이퍼파라미터
# n_estimator : 부트스트랩 데이터셋 수
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
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(base_estimator=SVC(), n_estimators=10)

model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.98046875

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# 훈련데이터 오차행렬:
#  [[327   6]
#  [  4 175]]

from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.99      0.98      0.98       333
#            1       0.97      0.98      0.97       179
#
#     accuracy                           0.98       512
#    macro avg       0.98      0.98      0.98       512
# weighted avg       0.98      0.98      0.98       512

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.9649122807017544

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

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor(base_estimator=KNeighborsRegressor(),
                         n_estimators=10, random_state=0)
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.6928982134381334

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.5612676280708411

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  52892.27111989147
# 테스트 데이터 RMSE:  63323.12131927774

# 훈련데이터에 과대적합되는 경향이 있음
# 기저모델의 개별 최적 하이퍼파라미터를 찾고 이를 배깅에 적재하면
# 더 좋은 결과 가능
