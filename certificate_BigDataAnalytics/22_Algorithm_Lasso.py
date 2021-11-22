
# =================================================================
# Lasso
# - Ridge 회귀모델과 유사하게 특성의 계수값을 0에 가깝게 함
#   차이점은 실제 중요하지 않은 변수의 계수를 아예 0으로 만들어
#   불필요한 변수 제거
#
# sklearn.linear_model.Lasso
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

from sklearn.linear_model import Lasso
model = Lasso()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.5455724679313863

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.5626850497564577

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  64340.34152172676
# 테스트 데이터 RMSE:  63220.748913873045

# Grid Search
param_grid = {'alpha': [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 3.0]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(Lasso(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'alpha': 0.5}
# Best Score: 0.5452
# Test set Score: 0.5627

# Random Search
from scipy.stats import randint
param_distribs = {'alpha': randint(low=0.00001, high=10)}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(Lasso(),
                                   param_distributions=param_distribs,
                                   n_iter=100,
                                   cv=5)
random_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'alpha': 1}
# Best Score: 0.5452
# Test set Score: 0.5627
