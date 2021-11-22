
# =================================================================
# Linear Regression Model
# - 연속형 원인변수가 연속형 결과변수에 영향을 미치는지 분석하여
#   레이블 변수를 예측하기 위한 목적으로 활용
#
# 모델성능지표
# 실제값과 예측값 간에 얼마나 일치하는지, 얼마나 차이가 나는지를 계산
# - RMSE : 평균제곱근 오차
# - MAPE : 평균절대백분율 오차
# - SSE : 오차제곱합
# - AE : 평균오차
# - MSE : 평균제곱오차
# - MAE : 평균절대오차
#
# sklearn.linear_model : Linear Models
# 선형회귀모델
#
# 특별한 하이퍼파라미터는 없음
# - normalize : 특성치(X)의 정규화
#   보통 이미 정규화한 데이터를 모델에 투입하기 때문에 기본 False 설정
# - intercept : X가 0일 때 Y의 기본값인 상수를 모델에 반영할지 여부
#   모델에서 상수는 대부분 필수
# => 결론적으로 기본 옵션 그대로 분석하면 됨
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


# =================================================================
# statmodel
# sklearn 머신러닝 모델 이용 전 파이썬의 통계분석 모듈인 statmodel 사용
# 회귀분석에서 각종 통계량과 지표의 의미를 아는 것은 분석에 더 도움이 됨
# =================================================================

# const 열 추가 : 1.0
# 이 변수가 상수(intercept)를 추정하는 역할
# 통계적 회귀분석에서는 일반적으로 X데이터의 정규화를 하지 않음
import statsmodels.api as sm
x_train_new = sm.add_constant(X_train)
x_test_new = sm.add_constant(X_test)
x_train_new.head()
#      const  Clump_Thickness  ...  Normal_Nucleoli  Mitoses
# 131    1.0                3  ...                2        1
# 6      1.0                1  ...                1        1
# 0      1.0                5  ...                1        1
# 269    1.0                1  ...                1        1
# 56     1.0                5  ...                1        1

# sm.OLS를 적용하고 바로 훈련
# 주의점 : y dataset, x dataset과 같이 레이블 데이터셋을 먼저 지정
multi_model = sm.OLS(y_train, x_train_new).fit()

# 결과 요약 : multi_model.summary()
# - R-squared : 예측값과 실제값이 일치하는 정도 (설명력)
#   가장 먼저 확인할 것
# - coef : 기울기
#   X변수가 1증가할 때 Y가 변화하는 정도
# - p < |t| : 통계적으로 유의한가
#   일반적으로 0.05보다 작으면 유의한 영향을 미치는 변수
print(multi_model.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:            house_value   R-squared:                       0.546
# Model:                            OLS   Adj. R-squared:                  0.545
# Method:                 Least Squares   F-statistic:                     3980.
# Date:                Mon, 22 Nov 2021   Prob (F-statistic):               0.00
# Time:                        09:25:19   Log-Likelihood:            -1.6570e+05
# No. Observations:               13266   AIC:                         3.314e+05
# Df Residuals:                   13261   BIC:                         3.315e+05
# Df Model:                           4                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const      -2.849e+04   8884.093     -3.206      0.001   -4.59e+04   -1.11e+04
# income      5.588e+04    500.997    111.538      0.000    5.49e+04    5.69e+04
# bedrooms    5.586e+05   2.02e+04     27.637      0.000    5.19e+05    5.98e+05
# households -2.586e+04    775.128    -33.356      0.000   -2.74e+04   -2.43e+04
# rooms      -5810.6069    834.780     -6.961      0.000   -7446.896   -4174.318
# ==============================================================================
# Omnibus:                     1975.541   Durbin-Watson:                   2.016
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4568.878
# Skew:                           0.866   Prob(JB):                         0.00
# Kurtosis:                       5.294   Cond. No.                         284.
# ==============================================================================

# test dataset
multi_model2 = sm.OLS(y_test, x_test_new).fit()
print(multi_model2.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:            house_value   R-squared:                       0.563
# Model:                            OLS   Adj. R-squared:                  0.562
# Method:                 Least Squares   F-statistic:                     1421.
# Date:                Mon, 22 Nov 2021   Prob (F-statistic):               0.00
# Time:                        09:26:24   Log-Likelihood:                -55169.
# No. Observations:                4423   AIC:                         1.103e+05
# Df Residuals:                    4418   BIC:                         1.104e+05
# Df Model:                           4                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const      -2.196e+04   1.48e+04     -1.483      0.138    -5.1e+04    7075.709
# income       5.57e+04    838.452     66.426      0.000    5.41e+04    5.73e+04
# bedrooms    5.402e+05   3.44e+04     15.713      0.000    4.73e+05    6.08e+05
# households -2.603e+04   1270.717    -20.484      0.000   -2.85e+04   -2.35e+04
# rooms      -6039.8888   1344.918     -4.491      0.000   -8676.601   -3403.177
# ==============================================================================
# Omnibus:                      688.606   Durbin-Watson:                   1.968
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1499.714
# Skew:                           0.915   Prob(JB):                         0.00
# Kurtosis:                       5.188   Cond. No.                         284.
# ==============================================================================


# =================================================================
# Scikit-learn
# =================================================================

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)
# 0.5455724996358273

pred_test = model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)
# 0.5626843883587158

# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE: ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE: ", np.sqrt(MSE_test))
# 훈련 데이터 RMSE:  64340.33927728243
# 테스트 데이터 RMSE:  63220.79672157403

# 기타 선형 모델평가지표 : MAE (Mean Absolute Error)
# 실제값과 예측값 차이에 절대값을 씌어 평균
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, pred_test)
# 47230.87470163738

# 기타 선형 모델평가지표 : MSE (Mean Squared Error)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, pred_test)
# 3996869138.1105857

# 기타 선형 모델평가지표 : MAPE (Mean Absolute Percentage Error)
# 평균 절대 오차비율
# 실제값 대비 오차(실제값-예측값) 정도를 백분률로 나타낸 지표
# 일반적 선형회귀모델보다는 시계열 데이터에서 주로 사용
def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - pred_test) / y_test)) * 100
MAPE(y_test, pred_test)
# house_value    30.571439

# 기타 선형 모델평가지표 : MPE (Mean Percentage Error)
def MAE(y_test, y_pred):
    return np.mean((y_test - pred_test) / y_test) * 100
MAE(y_test, pred_test)
# house_value   -12.37266
