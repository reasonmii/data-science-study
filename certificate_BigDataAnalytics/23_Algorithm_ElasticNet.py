
# =================================================================
# Elasticnet
# - Ridge 회귀와 Lasso 회귀 절충
# - 규제항의 혼합정도를 혼합비율 r을 사용해 조절
# - r=0 : ridge 회귀
#   r=1 : lasso 회귀
# - 규제식
#   j(theta) = MSE(theta) + ra*sum(|theta|) + (1-r)a/2*sum(theta^2)
#   ra*sum(|theta|) : Lasso
#   (1-r)a/2*sum(theta^2) : Ridge
#
# sklearn.linear_model.ElasticNet
# - alpha (default = 1)
#   값이 크면 계수를 0에 가깝게 제약하여 훈련데이터의 정확도는
#   낮아지지만 일반화에 기여
#   0에 가까울수록 회귀계수를 아무런 제약을 하지 않은 선형회귀와
#   유사하게 적용
# =================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from sklearn.linear_model import ElasticNet
model = ElasticNet()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.050029698219161034

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.051683303919568435

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  93026.36648194955
# 테스트 데이터 RMSE:  93097.74727682666

# Grid Search
param_grid = {'alpha': [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 3.0]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(ElasticNet(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'alpha': 1e-05}
# Best Score: 0.5452
# Test set Score: 0.5627

# Random Search
from scipy.stats import randint
param_distribs = {'alpha': randint(low=0.00001, high=10)}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(ElasticNet(),
                                   param_distributions=param_distribs,
                                   n_iter=100,
                                   cv=5)
random_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'alpha': 0}
# Best Score: 0.5452
# Test set Score: 0.5627

