
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# 1. 데이터 범주-연속-레이블로 나누기
# =================================================================

data = pd.read_csv('vote.csv', encoding='utf-8')

data.head()
data.describe()
data.info()

X1 = data[['gender', 'region']]
XY = data[['edu','income', 'age', 'score_gov', 'score_progress', 'score_intention', 'vote', 'parties']]


# =================================================================
# 2. 범주형 변수의 One-hot-encoding 변환
# =================================================================

X1['gender'] = X1['gender'].replace([1,2], ['male', 'female'])
X1['region'] = X1['region'].replace([1,2,3,4,5],['Sudo', 'Chungcheung', 'Honam', 'Youngnam', 'Others'])

X1.head()

X1_dum = pd.get_dummies(X1)
X1_dum.head()
#    gender_female  gender_male  ...  region_Sudo  region_Youngnam
# 0              0            1  ...            0                1
# 1              0            1  ...            0                0
# 2              0            1  ...            0                0
# 3              1            0  ...            1                0
# 4              0            1  ...            1                0


# =================================================================
# 3. 자료 통합 및 저장하기
# =================================================================

Fvote = pd.concat([X1_dum, XY], axis=1)
Fvote.head()

Fvote.to_csv('Fvote.csv', index=False, sep=',', encoding='utf-8')


# =================================================================
# 4. 특성치(X), 레이블(y) 나누기
# =================================================================

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Fvote.csv', encoding='utf-8')

data.describe()
data.info()
data.head()

# 방법 세 가지
X = data[['gender_female', 'gender_male',
          'region_Chungcheung', 'region_Honam', 'region_Others', 'region_Sudo', 'region_Youngnam',
          'edu', 'income', 'age', 'score_gov', 'score_progress', 'score_intention']]
X = data[data.columns[1:14]]
X = data.loc[:,'gender_female':'score_intention']

y = data[['vote']]


# =================================================================
# 5. train-test 데이터셋 나누기
# =================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

print(X_train.shape)   # (158, 13)
print(X_test.shape)    # (53, 13)


# =================================================================
# 6. 모델 적용
# =================================================================

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# 1) 랜덤 없는 교차검증 : cross_val_score
# 기존 데이터셋의 순서가 그대로 적용됨
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)
print("5개 테스트 셋 정확도: ", scores)
print("정확도 평균: ", scores.mean())
# 5개 테스트 셋 정확도:  [0.71875    0.6875     0.8125     0.58064516 0.80645161]
# 정확도 평균:  0.7211693548387096

# 2) 랜덤 있는 교차검증 : K-Fold
# 데이터셋의 순서가 랜덤으로 섞여서 사용됨
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kfold)
print("5개 폴드의 정확도: ", scores)
print("정확도 평균: ", scores.mean())
# 5개 폴드의 정확도:  [0.71875    0.6875     0.625      0.70967742 0.77419355]
# 정확도 평균:  0.7030241935483872

# 3) 임의분할 교차검증
# 훈련데이터와 테스트데이터 구성 시 다른 교차검증에서 사용된 데이터도 랜덤으로 선택
# 단, 특정 데이터가 어디에서도 사용되지 않을 수 있음
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=0.5, train_size=0.5, random_state=42)
score = cross_val_score(model, X_train, y_train, cv=shuffle_split)
print("교차검증 정확도: ", scores)
print("정확도 평균: ", scores.mean())
# 교차검증 정확도:  [0.71875    0.6875     0.625      0.70967742 0.77419355]
# 정확도 평균:  0.7030241935483872


# =================================================================
# 7. train-validity-test 분할과 교차검증
# =================================================================

from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, random_state=2)

model.fit(X_train, y_train)
scores = cross_val_score(model, X_train, y_train, cv=5)
print("교차검증 정확도: ", scores)
print("정확도 평균: ", scores.mean())
# 교차검증 정확도:  [0.70833333 0.66666667 0.66666667 0.69565217 0.69565217]
# 정확도 평균:  0.6865942028985507

scores = cross_val_score(model, X_valid, y_valid, cv=5)
print("교차검증 정확도: ", scores)
print("정확도 평균: ", scores.mean())
# 교차검증 정확도:  [0.875 1.    0.875 0.625 0.75 ]
# 정확도 평균:  0.825

scores = cross_val_score(model, X_test, y_test, cv=5)
print("교차검증 정확도: ", scores)
print("정확도 평균: ", scores.mean())
# 교차검증 정확도:  [0.72727273 0.54545455 0.72727273 0.5        0.7       ]
# 정확도 평균:  0.64
