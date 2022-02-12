
#===============================================================================================================
# 시계열분석
#
# ARIMA
# 1) ★ 정상성 시계열만 사용 가능
# 2) 비정상 시계열 자료는 차분해 정상성으로 만족하는 시계열로 변경 필요
#===============================================================================================================

install.packages("tseries") 
install.packages("forecast")
install.packages("TTR")

library(tseries)
library(forecast)
library(TTR)

# skip=3 : 앞에 세 줄 건너 뛰고 그 다음부터 읽기
king <- scan("http://robjhyndman.com/tsdldata/misc/kings.dat", skip=3)

king.ts <- ts(king)
# Time Series:
# Start = 1 
# End = 42 
# Frequency = 1 
# [1] 60 43 67 50 56 42 50 65 68 43 65 34 47 34 49 41 13 35 53 56 16 43 69 59 48 59 86 55 68 51 33 49 67 77 81 67 71 81 68 70 77 56

plot.ts(king.ts)

# SMA : Simple Moving Average
# 3년마다 평균내서 그래프 그리기
king.sma3 <- SMA(king.ts, n=3)
plot.ts(king.sam3)

# 8년마다 평균내서 그래프 그리기
king.sma8 <- SMA(king.ts, n=8)
plot.ts(king.sma8)

# 해석
# 평균이 시간에 따라 일정하지 않음 = 비정상시계열
# -> ARIMA 분석을 하려면 차분 필요

# ARIMA(p,1,q) 모델
# 1차 차분 결과 평균과 분산이 시간에 의존하지 않음을 확인
# = 차분을 한 번 해야 정상성 만족
king.ff1 <- diff(king.ts, differences=1)
plot.ts(king.ff1)

# ACF와 PACF를 통해 적합한 ARIMA 모델 결정

# 1) ACF
# lag는 0부터 값을 갖지만 너무 많은 구간 설정하면 그래프 보고 판단하기 어려움 -> 20 정도로 설정
acf(king.ff1, lag.max=20)
# 그래프 : ACF 값이 lag1인 지점 빼고는 모두 점선 구간 안에 있음

acf(king.ff1, lag.max=20, plot=FALSE)
# Autocorrelations of series ‘king.ff1’, by lag
# 
# 0      1      2      3      4      5      6      7      8      9     10    
# 1.000 -0.360 -0.162 -0.050  0.227 -0.042 -0.181  0.095  0.064 -0.116 -0.071
# 11     12     13     14     15     16     17     18     19     20 
# 0.206 -0.017 -0.212  0.130  0.114 -0.009 -0.192  0.072  0.113  -0.093 

# 2) PACF
# PACF 값이 lag1,2,3에서 점선 구간을 초과하고 음의 값을 가짐
# 절단점 : lag4
pacf(king.ff1, lag.max=20)

pacf(king.ff1, lag.max=20, plot=FALSE)
# Partial autocorrelations of series ‘king.ff1’, by lag
# 
# 1      2      3      4      5      6      7      8      9     10      
# -0.360 -0.335 -0.321  0.005  0.025 -0.144 -0.022 -0.007 -0.143 -0.167 
# 11     12     13     14     15     16     17     18     19     20
# 0.065  0.034 -0.161  0.036  0.066  0.081 -0.005 -0.027 -0.006 -0.037

# ARMA 후보 생성
# 1) ARMA(3,0) 모델 : PACF 값이 lag4에서 절단점 가짐 - AR(3)모형
# 2) ARMA(0,1) 모델 : ACF 값이 lag2에서 절단점을 가짐 - MA(1)모형
# 3) ARMA(p,q) 모델 : AR 모형과 MA 모형 혼합

# 적절한 ARIMA 모형 찾기
# forecast package - auto.arima() 함수 사용
# 결과 : 영국 왕 사망 나이 데이터의 적절한 ARIMA 모형은 ARIMA(0,1,1)
auto.arima(king)
# Series: king 
# ARIMA(0,1,1) 
# 
# Coefficients:
#   ma1
# -0.7218
# s.e.   0.1208
# 
# sigma^2 = 236.2:  log likelihood = -170.06
# AIC=344.13   AICc=344.44   BIC=347.56

# 예측
# 42명의 영국왕 중 마지막 왕의 사망 나이 : 56세
# 43 ~ 52번째 10명의 왕 나이 예측 (h=10) -> 67.75살로 추정
# 신뢰구간 : 80~95%
king.arima <- arima(king, order=c(0,1,1))
king.forecasts <- forecast(king.arima)
king.forecasts
# Point    Forecast    Lo 80    Hi 80    Lo 95     Hi 95
# 43       67.75063   48.29647 87.20479 37.99806  97.50319
# 44       67.75063   47.55748 87.94377 36.86788  98.63338
# 45       67.75063   46.84460 88.65665 35.77762  99.72363
# 46       67.75063   46.15524 89.34601 34.72333 100.77792
# 47       67.75063   45.48722 90.01404 33.70168 101.79958
# 48       67.75063   44.83866 90.66260 32.70979 102.79146
# 49       67.75063   44.20796 91.29330 31.74523 103.75603
# 50       67.75063   43.59372 91.90753 30.80583 104.69543
# 51       67.75063   42.99472 92.50653 29.88974 105.61152
# 52       67.75063   42.40988 93.09138 28.99529 106.50596
