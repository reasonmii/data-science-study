
# 하이퍼파라미터를 찾는 두 가지 방법
# 1. 그리드탐색 (Grid Search)
#    분석자가 모델과 하이퍼파라미터에 대한 경험과 직관이 있어야 좋은 결과
# 2. 랜덤탐색 (Random Search)
#    무작위로 많이 찾아 돌려보는 탐욕적(greedy) 방법이지만, 매우 효과적
#    단, 물리적 연산장치가 충분히 뒷받침되어야 함


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# 1. 데이터 불러오기 및 데이터셋 분할
# =================================================================

import warnings
warnings = warnings.filterwarnings("ignore")

data = pd.read_csv('Fvote.csv', encoding='utf-8')

X = data[data.columns[1:13]]
y = data[['vote']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


# =================================================================
# 2. Grid Search
# =================================================================

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

from sklearn.linear_model import LogisticRegression

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Cross-Validity Score: {:.3f}".format(grid_search.best_score_))
# Best Parameter: {'C': 10}
# Best Cross-Validity Score: 0.727

print("Test set Score: {:.3f}".format(grid_search.score(X_test, y_test)))
# Test set Score: 0.679

result_grid = pd.DataFrame(grid_search.cv_results_)
result_grid

# Plot
plt.plot(result_grid['param_C'], result_grid['mean_train_score'], label='Train')
plt.plot(result_grid['param_C'], result_grid['mean_test_score'], label='Test')
plt.legend()


# =================================================================
# 3. Random Search
# =================================================================

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'C': randint(low=0.001, high=100)}

from sklearn.linear_model import LogisticRegression

random_search = RandomizedSearchCV(LogisticRegression(),
                                   param_distributions=param_distribs,
                                   cv=5,
                                   n_iter=100,  # default : 10
                                   return_train_score=True)

random_search.fit(X_train, y_train)

print("Best Parameter: {}".format(random_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(random_search.best_score_))
# Best Parameter: {'C': 18}
# Best Cross-validity Score: 0.727

print("Test set Scroe: {:.3f}".format(random_search.score(X_test, y_test)))
# Test set Scroe: 0.679

result_random = random_search.cv_results_
pd.DataFrame(result_random)

# Plot
plt.plot(result_grid['param_C'], result_grid['mean_train_score'])
plt.plot(result_grid['param_C'], result_grid['mean_test_score'])
plt.legend()
