
# =================================================================
# Neural Network
# - 인간의 뉴런구조와 활성화 작동원리를 근간으로
#   input(자극)과 output(반응)과의 연관 구현한 알고리즘
# - 전통적 알고리즘과 가장 큰 차이
#   중간에 은닉층(hidden layers)과 노드(nodes)들을 깊고(deep) 넓게(wide) 두어
#   특성치로부터 분류와 회귀를 더 잘할 수 있도록
#   특징추출 및 분류 단계를 확장하는 역할을 할 수 있도록 한 모델
# - 인공신경망의 기초모델 : 다층퍼셉트론 Multi-Layer Perceptron
# - 입력층과 출력층은 각각 특성치(X)와 레이블(y) 의미
# 
# sklearn.neural_network: Neural Network Models
# 1) MLPClassifier : 분류 알고리즘
# 2) MLPRegressor : 회귀 알고리즘
# 
# 다층퍼셉트론 모델
# 딥러닝의 구조와 동일하지만 하이퍼파라미터가 매우 많고 모델에 대한 깊은 이해 필요
# 기본설정모델로는 좋은 결과를 얻기 힘듦
# 은닉층과 노드의 수, 활성화함수 등을 다양하게 조합할 때 좋은 성능
#
# 핵심 파라미터
# 중간 은닉층 사용 개수, 각 은닉층 내 노드 개수
#
# 기타 파라미터
# - 학습률 learning rate
#   어느 정도의 간격으로 가중치를 조정하면서 오차를 줄일 것인지 결정
# - 활성화함수 activation function
#   분류/회귀를 선형(linear) 혹은 로지스틱(logistic, sigmoid), relu 사용
# - sgd, adam
#   오차를 줄이기 위해 해를 찾아가는 고속옵티마이저
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

from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.97265625

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# 훈련데이터 오차행렬:
#  [[327   6]
#  [  8 171]]

from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.98      0.98      0.98       333
#            1       0.97      0.96      0.96       179
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

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'hidden_layer_sizes':[10,30,50,100],
              'solver':['sgd','adam'],
              'activation':['tanh','relu']}
grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'activation': 'tanh', 'hidden_layer_sizes': 30, 'solver': 'adam'}
# Best Score: 0.9765
# Test set Score: 0.9591

# Random Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'hidden_layer_sizes':randint(low=10, high=100),
                  'solver':['sgd','adam'],
                  'activation':['tanh','relu']}
random_search = RandomizedSearchCV(MLPClassifier(),
                                   param_distributions=param_distribs,
                                   n_iter=10,
                                   cv=5)
random_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'activation': 'relu', 'hidden_layer_sizes': 59, 'solver': 'adam'}
# Best Score: 0.9765
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

from sklearn.neural_network import MLPRegressor
model = MLPRegressor()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# -2.9072398258655396

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# -2.864103559507669

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  188662.71432770626
# 테스트 데이터 RMSE:  187925.99245858338

# Tuning Model
# 이 모델은 하이퍼파리미터가 매우 다양하여 최적 조합을 찾는 것이 어려움
# - 은닉층을 3개로 두어 각각 64개의 노드를 구성하는 깊은 모델을 만들고자 함
#   hidden_layer_sizes=(64, 64, 64)
# - 활성화함수 : relu 설정
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(64,64,64),
                     activation='relu',
                     random_state=1,
                     max_iter=2000)
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.566197903746314

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.584086684313508

# RMSE (Roote Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  62863.255358058195
# 테스트 데이터 RMSE:  61654.37310884089
