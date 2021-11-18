
# =================================================================
# sklearn.neighbors : KNeighborsClassifier (범주)
# sklearn.neighbors : KNeighborsRegressor (회귀)
#
# K-Nearest Neighbor
# 가까운 K개의 다른 데이터의 레이블을 참조하여 분류
# 거리계산 : 유클리디안, minkowski,
# 주요 하이퍼파라미터 : 몇 개의 케이스들을 기준으로
#                      동일 범주, 동일 값을 분류 혹은 예측할 것인가
#
# 최적의 K수는 일반적으로 3~10 범위 내에서 사용
# 1) K값이 작을수록 정교한 분류, 예측
#    ex. 주변에 가장 가까운 1개의 케이스를 참조
#    overfitting
# 2) K값이 클수록 주변 많은 케이스들의 평균적 군집과 평균값으로 분류, 예측
#    underfitting
#    
#
# 적합한 K값 찾기
# 1) 데이터 수의 제곱근 값 대입
# 2) 직접 탐색 : Grid Search, Random Search
#
# Hyper Parameter
# - n_neighbors : K값 (default : 5)
#                 1에 가까울수록 overfitting, 클수록 underfitting
# - metric : 거리측정 (default : minkowski)
# =================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# KNeighborsClassifier : Classification
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

# KNeighborsClassifier ============================================

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# 훈련데이터 오차행렬:
#  [[331   2]
#  [  6 173]]

from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.98      0.99      0.99       333
#            1       0.99      0.97      0.98       179
#
#     accuracy                           0.98       512
#    macro avg       0.99      0.98      0.98       512
# weighted avg       0.98      0.98      0.98       512

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)

confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)
# 테스트데이터 오차행렬:
#  [[106   5]
#  [  3  57]]
 
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

# Grid Search =====================================================

from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'n_neighbors': 3}
# Best Score: 0.9824
# Test set Score: 0.9532

# Random Search ===================================================

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'n_neighbors': randint(low=1, high=20)}
random_search = RandomizedSearchCV(KNeighborsClassifier(),
                                   param_distributions=param_distribs,
                                   n_iter=20,
                                   cv=5)

random_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'n_neighbors': 3}
# Best Score: 0.9824
# Test set Score: 0.9532


# =================================================================
# KNeighborsClassifier : Regression
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

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.6804607237174459

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.5541889571372401

# RMSE : Root Mean Squared Error

from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  53952.69804097723
# 테스트 데이터 RMSE:  63831.91662964773

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [1,3,4,7,9,11]}
grid_search = GridSearchCV(KNeighborsRegressor(),
                           param_grid,
                           cv=5)

grid_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'n_neighbors': 11}
# Best Score: 0.5638
# Test set Score: 0.5880

from sklearn.model_selection import RandomizedSearchCV
param_distribs = {'n_neighbors': randint(low=1, high=20)}
random_search = RandomizedSearchCV(KNeighborsRegressor(),
                                   param_distributions=param_distribs,
                                   n_iter=20,
                                   cv=5)

random_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'n_neighbors': 19}
# Best Score: 0.5777
# Test set Score: 0.6004

# 주택가격 데이터는 16천여 개로 양이 많은 편
# 이 경우 K값이 크면 성능 및 일반화에 모두 도움
# -> 1~20의 범위를 좀 더 늘려서 그리드탐색, 랜덤탐색을 시도해 볼 수 있음
# 즉, KNN은 데이터의 수, 특성치의 수 등에 따라 K값을 작게 혹은 크게 늘려야 하는 case 기반 모델

