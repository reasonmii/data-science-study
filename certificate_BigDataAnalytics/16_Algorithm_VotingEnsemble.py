
# =================================================================
# Voting Ensemble 투표기반 앙상블
# - 여러 분류기를 학습시킨 후 각각의 분류기가 예측하는 레이블 범주가
#   가장 많이 나오는 범주를 예측
# - 다수결 원리
# - 개별 분류기의 최적 하이퍼파라미터를 찾고, 투표기반 앙상블 모델 생성
# - 좋은 개별 알고리즘을 조합하면 더 나은 결과를 보임
#
# sklearn.ensemble
# 1) 분류 : VotingClassifier
#    - 범주 Class : Hard Learner
#    - 확률 Probability : Soft Learner
#    - ★ 범주(hard)보다는 확률(soft) 방식이 다소 정확도가 높은 것으로 알려져 있음
# 2) 회귀 : VotingRegressor
#
# Hyper Parameter
# - Classifier : hard, sort
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

# voting = 'hard' =================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

logit_model = LogisticRegression(random_state=42)
rnf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(random_state=42)

voting_hard = VotingClassifier(estimators=[('lr', logit_model),
                                           ('rf', rnf_model),
                                           ('svc', svm_model)],
                               voting='hard')

voting_hard.fit(X_scaled_train, y_train)

# 3개의 개별 모델과 1개의 투표 앙상블 모델 결과를 모두 제시하기 위해 for문 구성
from sklearn.metrics import accuracy_score
for clf in (logit_model, rnf_model, svm_model):
    clf.fit(X_scaled_train, y_train)
    y_pred = clf.predict(X_scaled_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
# LogisticRegression 0.9590643274853801
# RandomForestClassifier 0.9649122807017544
# SVC 0.9649122807017544

from sklearn.metrics import confusion_matrix

# Logistic Regression
log_pred_train = logit_model.predict(X_scaled_train)
log_confusion_train = confusion_matrix(y_train, log_pred_train)
print("로지스틱 분류기 훈련데이터 오차행렬:\n", log_confusion_train)
# 로지스틱 분류기 훈련데이터 오차행렬:
#  [[328   5]
#  [  9 170]]
log_pred_test = logit_model.predict(X_scaled_test)
log_confusion_test = confusion_matrix(y_test, log_pred_test)
print("로지스틱 분류기 테스트데이터 오차행렬:\n", log_confusion_test)
# 로지스틱 분류기 테스트데이터 오차행렬:
#  [[106   5]
#  [  2  58]]

# RandomForest Classifier
rnf_pred_train = rnf_model.predict(X_scaled_train)
rnf_confusion_train = confusion_matrix(y_train, rnf_pred_train)
print("랜덤포레스트 분류기 훈련데이터 오차행렬:\n", rnf_confusion_train)
# 랜덤포레스트 분류기 훈련데이터 오차행렬:
#  [[333   0]
#  [  0 179]]

rnf_pred_test = rnf_model.predict(X_scaled_test)
rnf_confusion_test = confusion_matrix(y_test, rnf_pred_test)
print("랜덤포레스트 분류기 테스트데이터 오차행렬:\n", rnf_confusion_test)
# 랜덤포레스트 분류기 테스트데이터 오차행렬:
#  [[106   5]
#  [  1  59]]
 
# SVC
svm_pred_train = svm_model.predict(X_scaled_train)
svm_confusion_train = confusion_matrix(y_train, svm_pred_train)
print("서포트벡터머신 분류기 훈련데이터 오차행렬:\n", svm_confusion_train)
# 서포트벡터머신 분류기 훈련데이터 오차행렬:
#  [[329   4]
#  [  4 175]]

svm_pred_test = svm_model.predict(X_scaled_test)
svm_confusion_test = confusion_matrix(y_test, svm_pred_test)
print("서포트벡터머신 분류기 테스트데이터 오차행렬:\n", svm_confusion_test)
# 서포트벡터머신 분류기 테스트데이터 오차행렬:
#  [[106   5]
#  [  1  59]]

# 투표기반 앙상블 모델
voting_pred_train = voting_hard.predict(X_scaled_train)
voting_confusion_train = confusion_matrix(y_train, voting_pred_train)
print("투표분류기 훈련데이터 오차행렬:\n", voting_confusion_train)
# 투표분류기 훈련데이터 오차행렬:
#  [[329   4]
#  [  4 175]]

voting_pred_test = voting_hard.predict(X_scaled_test)
voting_confusion_test = confusion_matrix(y_test, voting_pred_test)
print("투표분류기 테스트데이터 오차행렬:\n", voting_confusion_test)
# 투표분류기 테스트데이터 오차행렬:
#  [[106   5]
#  [  1  59]]

# voting = 'soft' =================================================

logit_model = LogisticRegression(random_state=42)
rnf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True, random_state=42)
voting_soft = VotingClassifier(estimators=[('lr', logit_model),
                                           ('rf', rnf_model),
                                           ('svc', svm_model)],
                               voting='soft')

voting_soft.fit(X_scaled_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (logit_model, rnf_model, svm_model):
    clf.fit(X_scaled_train, y_train)
    y_pred = clf.predict(X_scaled_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
# LogisticRegression 0.9590643274853801
# RandomForestClassifier 0.9649122807017544
# SVC 0.9649122807017544
    
voting_pred_train = voting_soft.predict(X_scaled_train)
voting_confusion_train = confusion_matrix(y_train, voting_pred_train)
print("투표분류기 훈련데이터 오차행렬:\n", voting_confusion_train)
# 투표분류기 훈련데이터 오차행렬:
#  [[330   3]
#  [  3 176]]

voting_pred_test = voting_soft.predict(X_scaled_test)
voting_confision_test = confusion_matrix(y_test, voting_pred_test)
print("투표분류기 테스트데이터 오차행렬:\n", voting_confusion_test)
# 투표분류기 테스트데이터 오차행렬:
#  [[106   5]
#  [  1  59]]


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

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

linear_model = LinearRegression()
rnf_model = RandomForestRegressor(random_state=42)
voting_regressor = VotingRegressor(estimators=[('lr', linear_model),
                                               ('rf', rnf_model)])
voting_regressor.fit(X_scaled_train, y_train)

pred_train = voting_regressor.predict(X_scaled_train)
voting_regressor.score(X_scaled_train, y_train)
# 0.7962532705428835

pred_test = voting_regressor.predict(X_scaled_test)
voting_regressor.score(X_scaled_test, y_test)
# 0.5936371957936409

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  43082.050654857834
# 테스트 데이터 RMSE:  60942.38524353489

# 2개 알고리즘만 조합했음에도 정확도가 개별 알고리즘 대비 2~3% 높아짐
# 개별 알고리즘에서 가장 좋은 하이퍼파라미터를 찾아 설정하면 더 좋은 결과 얻을 수 있음
