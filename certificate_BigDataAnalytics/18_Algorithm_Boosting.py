
# =================================================================
# Boosting
# - 순차적 직렬식 앙상블
#   여러 개의 약한 학습기 weak learner 순차적으로 학습
#   ※ Bagging : 한 번에 여러 개 데이터셋에서 학습한 결과 종합
# - 잘못 예측한 데이터에 가중치 부여하여 오류 개선
# - 오류를 찾아 해결하는 방식이므로 훈련데이터에 과대적합되는 경향
# - Adaboosting과 GradientBoosting 모두 과대적합 문제가 나타나지만
#   테스트 데이터에서는 일반적 수준의 정확도를 보임
# 
# sklearn.ensemble
# 1) AdaBoosting
#    핵심 파라미터 : base_estimator, n_estimator
#    - AdaBoostClassifier : 분류
#    - AdaBoostRegressor : 회귀
# 2) GradientBoosting
#    학습률(learning_rate) 설정하여 오차(loss) 줄이는 방향으로 찾아나가는 방식
#    옵션 설정이 다소 복잡함
#    - GradientBoostingClassifier : 분류
#    - GradientBoostingRegressor : 회귀
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

# AdaBoosting 앙상블 모델 적용 =====================================
# n_estimators : 모델 수행횟수
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100, random_state=0)
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 1.0

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.9532163742690059

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# 훈련데이터 오차행렬:
#  [[333   0]
#  [  0 179]]

confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)
# 테스트데이터 오차행렬:
#  [[106   5]
#  [  3  57]]

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

cfreport_test = classification_report(y_test, pred_test)
print("분류예측 레포트:\n", cfreport_test)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.97      0.95      0.96       111
#            1       0.92      0.95      0.93        60
#
#     accuracy                           0.95       171
#    macro avg       0.95      0.95      0.95       171
# weighted avg       0.95      0.95      0.95       171

# GridentBoosting 앙상블 모델 적용 =================================
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)
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
#            0       1.00      1.00      1.00       333
#            1       1.00      1.00      1.00       179
#
#     accuracy                           1.00       512
#    macro avg       1.00      1.00      1.00       512
# weighted avg       1.00      1.00      1.00       512

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
# 아래 예시에서는 GradientBoosting 결과가 더 좋게 나왔지만
# 실제로는 AdaBoosting 성능이 더 좋은 경우가 많음
# 데이터 특성, 파리미터 조정을 통해 AdaBoosting 모델 튜닝 필요
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

# AdaBoosting 앙상블 모델 적용 =====================================
from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.47675870153901745

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.4762228581403902

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  74780.72847662865
# 테스트 데이터 RMSE:  74891.44537057991

# GridentBoosting 앙상블 모델 적용 =================================
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=0)
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.6178724780500952

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.5974112241813845

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  59000.433545962376
# 테스트 데이터 RMSE:  60658.72886338227
