
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# 1. 분석 데이터 검토
# =================================================================

data = pd.read_csv('house_price.csv', encoding='utf-8')

data.head()
data.describe()

print(data.shape)
# (17689, 6)

# Histogram
data.hist(bins=50, figsize=(20,15))


# =================================================================
# 2. 특성(X)과 레이블(y) 나누기
# =================================================================

# 방법1) 특성 이름으로 특성 데이터셋(X) 나누기
X1 = data[['housing_age', 'income', 'bedrooms', 'households', 'rooms']]

# 방법2) 특성 위치값으로 특성 데이터셋(X) 나누기
X2 = data[data.columns[0:5]]

# 방법3) loc 함수로 특성 데이터셋(X) 나누기
X3 = data.loc[:, 'housing_age':'rooms']

# (17689, 5)
print(X1.shape)
print(X2.shape)
print(X3.shape)

y = data[['house_value']]

# (17689, 1)
print(y.shape)


# =================================================================
# 3. train-test 데이터셋 나누기
# =================================================================

# ★ stratify = y : y레이블의 범주비율에 맞게 분류
# -> 회귀모델에서는 옵션 설정 불가능
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, random_state = 42)

print(y_train.mean())
print(y_test.mean())
# house_value    189260.967812
# dtype: float64
# house_value    188391.001357
# dtype: float64


# =================================================================
# 4. 정규화
# =================================================================

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

# train data 정규화 ===============================================

scaler_minmax.fit(X_train)
X_scaled_minmax_train = scaler_minmax.transform(X_train)

scaler_standard.fit(X_train)
X_scaled_standard_train = scaler_standard.transform(X_train)

scaler_minmax.fit(X_test)
X_scaled_minmax_test = scaler_minmax.transform(X_test)

scaler_standard.fit(X_test)
X_scaled_standard_test = scaler_standard.transform(X_test)

# 기술통계량 확인 : min 0, max 1
pd.DataFrame(X_scaled_minmax_train).describe()

# 기술통계량 확인 : 평균 0, 표준편차 1
pd.DataFrame(X_scaled_standard_train).describe()

pd.DataFrame(X_scaled_minmax_test).describe()
pd.DataFrame(X_scaled_standard_test).describe()

# 특정 컬럼 정규화 ================================================
from sklearn.preprocessing import minmax_scale
a['mtcars'] = minmax_scale(a['mtcars'])

# 정규화 후 0.5보다 큰 데이터 개수
print(data[len(a['mtcars'] > 0.5]))  # 데이터 수
print(sum(a['mtcars'] > 0.5))  # True 출력 개수


# =================================================================
# 5. 모델 학습
# =================================================================

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_scaled_minmax_train, y_train)

# 모델평가지표 1 ===================================================
# R-squared =======================================================

pred_train = model.predict(X_scaled_minmax_train)
model.score(X_scaled_minmax_train, y_train)
# 0.5706921210926263

pred_test = model.predict(X_scaled_minmax_test)
model.score(X_scaled_minmax_test, y_test)
# 0.5817349363867992

# 모델평가지표 2 ===================================================
# RMSE (Root Mean Squared Error) ==================================
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test, pred_test)
RMSE = np.sqrt(MSE)
print(RMSE)
# 61828.43721035346

# 모델평가지표 3 ===================================================
# MAPE (Mean Absolute Percentage Error) ===========================
def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

MAPE(y_test, pred_test)
# house_value    30.671845
# dtype: float64

# 모델평가지표 4 ===================================================
# MPE (Mean Percentage Error) =====================================
def MAE(y_test, y_pred):
    return np.mean((y_test - pred_test) / y_test) * 100

MAE(y_test, pred_test)
# house_value   -12.864669
# dtype: float64


# =================================================================
# 6. 예측값 병합 및 저장
# =================================================================

prob_train = model.predict(X_scaled_minmax_train)
y_train[['y_pred']] = pred_train
y_train

prob_test = model.predict(X_scaled_minmax_test)
y_test[['y_pred']] = pred_test
y_test

# pd.concat([데이터셋1, 데이터셋2, ..., 데이터셋n], option)
# option : axis=1 변수 병합 (열추가)
#          axis=0 케이스 병합 (행추가)
Total_test = pd.concat([X_test, y_test], axis=1)
Total_test

# CSV 파일로 내보내기
Total_test.to_csv('regression_test.csv')

