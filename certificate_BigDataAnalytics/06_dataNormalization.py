
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. 데이터 불러오기 및 확인
# =================================================================

data = pd.read_csv('Fvote.csv', encoding='utf-8')

data.head()
data.describe()
data.info()

data.hist(figsize=(20, 15))


# =================================================================
# 2. 특성치(X), 레이블(y) 나누기
# =================================================================

X = data.loc[:, 'gender_female':'score_intention']
y = data[['vote']]

print(X.shape)  # (211, 13)
print(y.shape)  # (211, 1)


# =================================================================
# 3. train-test 데이터셋 나누기
# =================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

print(y_train.mean())
print(y_test.mean())
# vote    0.708861
# dtype: float64
# vote    0.716981
# dtype: float64


# =================================================================
# 4. 연속형 특성의 Scaling
# =================================================================

# 1) Min-Max Scaling

from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()

scaler_minmax.fit(X_train)
X_scaled_minmax_train = scaler_minmax.transform(X_train)

pd.DataFrame(X_scaled_minmax_train).describe()

X_scaled_minmax_test = scaler_minmax.transform(X_test)
pd.DataFrame(X_scaled_minmax_test).describe()

# 2) Standardization Scaling

from sklearn.preprocessing import StandardScaler
scaler_standard = StandardScaler()

scaler_standard.fit(X_train)
X_scaled_standard_train = scaler_standard.transform(X_train)

# 평균 0, 표준편차 1
pd.DataFrame(X_scaled_standard_train).describe()

X_scaled_standard_test = scaler_standard.transform(X_test)

# X_train 기준으로 model을 fit 했기 때문에 test에서는 평균 0, 표준편차 1이 나타나지 않음
pd.DataFrame(X_scaled_standard_test).describe()


# =================================================================
# 5. 모델 학습
# =================================================================

# 1) Min-Max 정규화 데이터 적용 결과 ===============================
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_scaled_minmax_train, y_train)
pred_train = model.predict(X_scaled_minmax_train)
model.score(X_scaled_minmax_train, y_train)
# 0.7278481012658228

pred_test = model.predict(X_scaled_minmax_test)
model.score(X_scaled_minmax_test, y_test)
# 0.7169811320754716

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬: \n", confusion_train)
# 훈련데이터 오차행렬: 
#  [[  9  37]
#  [  6 106]]

confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬: \n", confusion_test)
# 테스트데이터 오차행렬: 
#  [[ 2 13]
#  [ 2 36]]

# 2) Standardize 정규화 데이터 적용 결과 ===========================

model.fit(X_scaled_standard_train, y_train)
pred_train = model.predict(X_scaled_standard_train)
model.score(X_scaled_standard_train, y_train)
# 0.740506329113924

pred_test = model.predict(X_scaled_standard_test)
model.score(X_scaled_standard_test, y_test)
# 0.6792452830188679

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬: \n", confusion_train)
# 훈련데이터 오차행렬: 
#  [[ 12  34]
#  [  7 105]]
 
confusion_test = confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬: \n", confusion_test)
# 테스트데이터 오차행렬: 
#  [[ 3 12]
#  [ 5 33]]
