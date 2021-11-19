
# =================================================================
# Decision Tree
# - 의사결정 규칙(Decision rule)을 나무구조로 도표화하여
#   관심대상이 되는 집단을 몇 개의 소집단으로 분류(Classification), 예측
# - root node 뿌리마디 : 레이블(y)
#   intermediate node 중간마디 : 분류되는 과정
#   terminal node 끝마디 : 최종 분류 기준
# - 장점
#   직관적으로 결과 도식화, 어떻게 분류되는지 알기 좋음
#   특성치(X)가 많지 않고 최종 알고리즘 도출 전에 탐색적으로
#   주요 분류 변수가 어떤 것인지 확인하는 차원에서는 활용하기 좋음 (초기 분석모델)      
# - 단점
#   분류단계가 많을수록 이해가 어려움
#   결과가 안정적이지 못해, 실제 머신러닝으로 많이 활용하지 않음
#   학습데이터에 과적합되는 경향이 있음
#   하이퍼파라미터의 설정에 대한 기준이 어려움
# - 실제 분석에서는 의사결정나무의 앙상블인 랜덤포레스트 선호
# 
# sklearn.tree: Decision Trees
# 1) 분류 : DecisionTreeClassifier
#    - 기준 : gini
# 2) 회귀 : DecisionTreeRegressor
#    - 기준 : mse
#
# 하이퍼파라미터
# - max_depth : 최대 가지치기 수
#   얼마나 많은 단계로 분류할 것인가
# - max_leaf_node : 리프 노드의 최대 개수
#   얼마나 많은 노드를 만들 것인가
# - max_sample_leaf : 리프 노드가 되기 위한 최소 샘플 수
#   한 노드에서 최소 표본수는 몇 개일 때 별개 집단으로 분류할 것인가
# -> 현실적으로 min_sample_leaf만 가늠할 수 있는 기준
#    다른 하이퍼파라미터는 결정하기가 어려움
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

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
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

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth':range(2,20,2),
              'min_samples_leaf':range(1,50,2)}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'max_depth': 14, 'min_samples_leaf': 1}
# Best Score: 0.9628
# Test set Score: 0.9591

# Random Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'max_depth': randint(low=1,high=20),
                  'min_samples_leaf': randint(low=1, high=50)}
random_search = RandomizedSearchCV(DecisionTreeClassifier(),
                                   param_distributions=param_distribs,
                                   n_iter=20, cv=5)
random_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'max_depth': 2, 'min_samples_leaf': 4}
# Best Score: 0.9453
# Test set Score: 0.9415


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

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 1.0

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.20167193315833565

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  0.0
# 테스트 데이터 RMSE:  85418.74236057274

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': range(2,20,2),
              'min_samples_leaf': range(1,50,2)}
grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'max_depth': 8, 'min_samples_leaf': 49}
# Best Score: 0.5592
# Test set Score: 0.5770

# Random Search
from sklearn.model_selection import RandomizedSearchCV
param_distribs = {'max_depth': randint(low=1, high=20),
                  'min_samples_leaf': randint(low=1, high=50)}
random_search = RandomizedSearchCV(DecisionTreeRegressor(),
                                   param_distributions = param_distribs,
                                   n_iter=20,
                                   cv=5)
random_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'max_depth': 8, 'min_samples_leaf': 48}
# Best Score: 0.5591
# Test set Score: 0.5764
