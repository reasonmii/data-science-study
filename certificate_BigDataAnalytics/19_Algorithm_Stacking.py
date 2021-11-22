
# =================================================================
# Stacking
# - 데이터 셋이 아니라, 여러 학습기에서 예측한
#   예측값(predict value)으로 다시 학습 데이터를 만들어 
#   일반화(generalization)된 최종 모델을 구성하는 방법
# - 모델을 어떻게 쌓는가에 따라 결과가 달라짐
#   모델 순서를 변경하거나 다른 알고리즘을 구성하면서 충분히
#   개선된 성능 결과 얻을 수 있음
#
# 1) 분류 : StackingClassifier
# 2) 회귀 : StackingRegressor
#
# 각 개별 알고리즘이 예측한 값들이 마지막 StackingClassifier와
# StackingRegressor의 학습 데이터가 됨
#
# 주요 하이퍼파라미터
# - estimators
#   여러 알고리즘을 쌓는 옵션
#   Voting 방법과 유사하게 여러 개별 알고리즘으로 estimator 설정
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

# 랜덤포레스트, 서포트벡터머신 2가지를 개별 모델로 설정
# 이를 StackingClassifier의 estimators로 설정
# 방식은 Voting과 유사
# 마지막 최종 모델로 로지스틱회귀모델 설정
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
              ('svr', SVC(random_state=42))]
model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.986328125

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.9649122807017544

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# 훈련데이터 오차행렬:
#  [[330   3]
#  [  4 175]]

confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)
# 테스트데이터 오차행렬:
#  [[106   5]
#  [  1  59]]
 
from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.99      0.99      0.99       333
#            1       0.98      0.98      0.98       179
#
#     accuracy                           0.99       512
#    macro avg       0.99      0.98      0.98       512
# weighted avg       0.99      0.99      0.99       512

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

# 선형회귀분석, KNN의 Regressor 2가지를 개별 모델로 설정
# 이를 StackingRegressor estimators로 설정
# 마지막 최종 모델로 랜덤포레스트 설정
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
estimators = [('lr', LinearRegression()),
              ('knn', KNeighborsRegressor())]
model = StackingRegressor(estimators=estimators,
                          final_estimator=RandomForestRegressor(n_estimators=10, random_state=42))
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.543404932348004

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.4781188528801523

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  64493.60476580374
# 테스트 데이터 RMSE:  69063.45138802647
