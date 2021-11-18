
# 다중분류는 이진분류와 달리 y 레이블의 범주 수가 3개 이상

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Fvote.csv', encoding='utf-8')

X = data[data.columns[1:13]]
y = data[['parties']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
model.score(X_train, y_train)
# 0.6139240506329114

pred_test = model.predict(X_test)
model.score(X_test, y_test)
# 0.5283018867924528

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# 훈련데이터 오차행렬:
#  [[21  2  3 11]
#  [ 1 25  2 12]
#  [ 6  2  5  6]
#  [ 7  8  1 46]]

confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)
# 테스트데이터 오차행렬:
#  [[ 6  1  2  4]
#  [ 1  9  1  2]
#  [ 1  2  1  2]
#  [ 2  5  2 12]]
 

# =================================================================
# Grid Search
# =================================================================

from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(grid_search.best_score_))
# Best Parameter: {'C': 0.1}
# Best Cross-validity Score: 0.544

print("Test set Score: {:.3f}".format(grid_search.score(X_test, y_test)))


# =================================================================
# Random Search
# =================================================================

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'C': randint(low=0.001, high=100)}

random_search = RandomizedSearchCV(LogisticRegression(),
                                   param_distributions=param_distribs,
                                   cv=5,
                                   n_iter=100,
                                   return_train_score=True)

random_search.fit(X_train, y_train)

print("Best Parameter: {}".format(random_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(random_search.best_score_))
# Best Parameter: {'C': 4}
# Best Cross-validity Score: 0.544

print("Test set Score: {:.3f}".format(random_search.score(X_test, y_test)))
# Test set Score: 0.509
