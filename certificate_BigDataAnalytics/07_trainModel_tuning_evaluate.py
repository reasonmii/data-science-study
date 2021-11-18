
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


# =================================================================
# 4. 모델평가
# =================================================================

Final_model = LogisticRegression(C=10)
Final_model.fit(X_train, y_train)

pred_train = Final_model.predict(X_train)
Final_model.score(X_train, y_train)
# 0.740506329113924

pred_test = Final_model.predict(X_test)
Final_model.score(X_test, y_test)
# 0.6792452830188679

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# 훈련데이터 오차행렬:
#  [[ 12  34]
#  [  7 105]]

from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.63      0.26      0.37        46
#            1       0.76      0.94      0.84       112
#
#     accuracy                           0.74       158
#    macro avg       0.69      0.60      0.60       158
# weighted avg       0.72      0.74      0.70       158

confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)
# 테스트데이터 오차행렬:
#  [[ 3 12]
#  [ 5 33]]

from sklearn.metrics import classification_report
cfreport_test = classification_report(y_test, pred_test)
print("분류예측 레포트:\n", cfreport_test)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.38      0.20      0.26        15
#            1       0.73      0.87      0.80        38
#
#     accuracy                           0.68        53
#    macro avg       0.55      0.53      0.53        53
# weighted avg       0.63      0.68      0.64        53

from sklearn.metrics import roc_curve, auc
from sklearn import metrics
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, Final_model.decision_function(X_test))
roc_auc = metrics.roc_auc_score(y_test, Final_model.decision_function(X_test))
roc_auc
# 0.6350877192982456

plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)'% roc_auc)
plt.plot([0,1], [1,1], 'y--')
plt.plot([0,1], [0,1], 'r--')
plt.legend(loc='lower right')
plt.show()

# 전체적으로 좋은 결과는 아님
# 좋은 결과는 좋은 모델보다 좋은 예측자(X)에 의해 결정
