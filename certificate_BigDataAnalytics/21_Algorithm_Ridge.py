
# =================================================================
# Ridge
# - 선형회귀분석의 기본원리를 따르나, 가중치(회귀계수) 값을 최대한
#   작게 만들어, 즉 0에 가깝게 만들어 모든 독립변수가 종속변수에 미치는
#   영향을 최소화하는 제약(regularization)을 반영한 회귀모델
# - 훈련데이터에 과대적합되지 않도록 제약한 모델
# - 선형관계뿐 아니라 다항곡선 추정도 가능
#
# sklearn.linear_model.Ridge
# - alpha (default = 1)
#   규제의 정도 결정
#   값이 클수록 규제가 강하여 회귀계수가 0에 근접함
#   값이 0에 가까울수록 규제를 하지 않아 선형회귀와 유사한 결과
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

from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.5455487773718164

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.5626954941458684

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  64342.018619526265
# 테스트 데이터 RMSE:  63219.99395904853

# Grid Search
param_grid = {'alpha': [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("Test set Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))
# Best Parameter: {'alpha': 0.1}
# Best Score: 0.5452
# Test set Score: 0.5627

# Random Search
from scipy.stats import randint
param_distribs = {'alpha': randint(low=0.0001, high=100)}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(Ridge(),
                                   param_distributions=param_distribs,
                                   n_iter=100,
                                   cv=5)
random_search.fit(X_scaled_train, y_train)
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("Test set Score: {:.4f}".format(random_search.score(X_scaled_test, y_test)))
# Best Parameter: {'alpha': 1}
# Best Score: 0.5451
# Test set Score: 0.5627

