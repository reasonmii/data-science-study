
#===============================================================================================================
# 단순선형 회귀분석
#===============================================================================================================

# 10년간 에어컨 예약대수와 판매대수 (단위 : 1천대)

x <- c(19,23,26,29,30,38,39,46,49)
y <- c(33,51,40,49,50,69,70,64,89)

lm(y~x)
# Call:
#   lm(formula = y ~ x)
# 
# Coefficients:
#   (Intercept)            x  
# 6.409        1.529  

summary(lm(y~x))
# Call:
#   lm(formula = y ~ x)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -12.766  -2.470  -1.764   4.470   9.412 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   6.4095     8.9272   0.718 0.496033    
# x             1.5295     0.2578   5.932 0.000581 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 7.542 on 7 degrees of freedom
# Multiple R-squared:  0.8341,	Adjusted R-squared:  0.8104 
# F-statistic: 35.19 on 1 and 7 DF,  p-value: 0.0005805

# 해석
# p-value < 0.05 : 통계적으로 유의함
# 결정계수 : 0.8341 : 회귀식이 데이터를 잘 설명함
# -> 결론 : 에어컨 판매대수를 에어컨 예약대수로 추정 가능하다



#===============================================================================================================
# 회귀분석 사례
#===============================================================================================================

library(MASS)
head(Cars93)

# ★ attach : dataFrame의 column에 직접 접근 가능
# Cars93$Origin -> Origin
attach(Cars93)

lm(Price~EngineSize+RPM+Weight, data=Cars93)
# Call:
#   lm(formula = Price ~ EngineSize + RPM + Weight, data = Cars93)
# 
# Coefficients:
#   (Intercept)   EngineSize          RPM       Weight  
# -51.793292     4.305387     0.007096     0.007271  

summary(lm(Price~EngineSize+RPM+Weight, data=Cars93))
# Call:
#   lm(formula = Price ~ EngineSize + RPM + Weight, data = Cars93)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -10.511  -3.806  -0.300   1.447  35.255 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -51.793292   9.106309  -5.688 1.62e-07 ***
#   EngineSize    4.305387   1.324961   3.249  0.00163 ** 
#   RPM           0.007096   0.001363   5.208 1.22e-06 ***
#   Weight        0.007271   0.002157   3.372  0.00111 ** 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 6.504 on 89 degrees of freedom
# Multiple R-squared:  0.5614,	Adjusted R-squared:  0.5467 
# F-statistic: 37.98 on 3 and 89 DF,  p-value: 6.746e-16

# 해석
# F-통계량 : 37.98, p-value < 0.05 -> 통계적으로 매우 유의함
# 결정계수 : 0.5614
# -> 결론 : 결정계수가 낮아 데이터의 설명력은 낮지만
#           통계적으로 유의하여 자동차의 가격을 엔진의 크기와 RPM, 무게로 추정 가능하다


library(boot)
data(nodal)

a <- c(2,4,6,7)
data <- nodal[,a]
glmModel <- glm(r~., data=data, family="binomial")
summary(glmModel)
# Call:
#   glm(formula = r ~ ., family = "binomial", data = data)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.1231  -0.6620  -0.3039   0.4710   2.4892  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  -3.0518     0.8420  -3.624  0.00029 ***
#   stage         1.6453     0.7297   2.255  0.02414 *  
#   xray          1.9116     0.7771   2.460  0.01390 *  
#   acid          1.6378     0.7539   2.172  0.02983 *  
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 70.252  on 52  degrees of freedom
# Residual deviance: 49.180  on 49  degrees of freedom
# AIC: 57.18
# 
# Number of Fisher Scoring iterations: 5

# 해석
# age, grade는 유의수준 5% 하에서 유의하지 않아 제외
# stage, xray, acid를 활용해서 모형 개발 필요



#===============================================================================================================
# 최적회귀방정식
#
# step(lm(출력변수~입력변수, dataset), scope=list(lower=~1, upper=~입력변수),direction="변수선택방법")
# scope : 변수선택 과정에서 설정 가능한 가장 큰/작은 모형 설정
# scope가 없는 경우 - 전진선택법에서는 현재 선택한 모형을 가장 큰 모형으로,
#                     후진제거법에서는 상수항만 있는 모형을 가장 작은 모형으로 설정
# direction : forward (전진선택법) / backward (후진제거법) / stepwise (단계적선택법)
# k : 모형선택 기준에서 AIC, BIC 등 옵션 사용 - k=2 : AIC / k=log(자료의수) : BIC
#===============================================================================================================

### 후진제거법 (backward elimination)

### 1) dataFrame 생성
x1 <- c(7,1,11,11,7,11,3,1,2,21,1,11,10)
x2 <- c(26,29,56,31,52,55,71,31,54,47,40,66,68)
x3 <- c(6,15,8,8,6,9,17,22,18,4,23,9,8)
x4 <- c(60,52,20,47,33,22,6,44,22,26,34,12,12)
y <- c(78.5,74.3,104.3,87.6,95.9,109.2,102.7,72.5,93.1,115.9,83.8,113.3,109.4)

df <- data.frame(x1,x2,x3,x4,y)
head(df)
# x1 x2 x3 x4     y
# 1  7 26  6 60  78.5
# 2  1 29 15 52  74.3
# 3 11 56  8 20 104.3
# 4 11 31  8 47  87.6
# 5  7 52  6 33  95.9
# 6 11 55  9 22 109.2

### 2) 회귀모형 생성
a <- lm(y~x1+x2+x3+x4, data=df)
summary(a)
# Call:
#   lm(formula = y ~ x1 + x2 + x3 + x4, data = df)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.1750 -1.6709  0.2508  1.3783  3.9254 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)  
# (Intercept)  62.4054    70.0710   0.891   0.3991  
# x1            1.5511     0.7448   2.083   0.0708 .
# x2            0.5102     0.7238   0.705   0.5009  
# x3            0.1019     0.7547   0.135   0.8959  
# x4           -0.1441     0.7091  -0.203   0.8441  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.446 on 8 degrees of freedom
# Multiple R-squared:  0.9824,	Adjusted R-squared:  0.9736 
# F-statistic: 111.5 on 4 and 8 DF,  p-value: 4.756e-07

# 해석
# t 통계량을 통한 유의확률이 0.05보다 작은 변수 없음 -> 모형 활용 불가능
# 유의확률이 가장 높은 x3 제외 후 회귀모형 재생성 필요

b <- lm(y~x1+x2+x4, data=df)
summary(b)
# Call:
#   lm(formula = y ~ x1 + x2 + x4, data = df)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.0919 -1.8016  0.2562  1.2818  3.8982 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  71.6483    14.1424   5.066 0.000675 ***
#   x1            1.4519     0.1170  12.410 5.78e-07 ***
#   x2            0.4161     0.1856   2.242 0.051687 .  
# x4           -0.2365     0.1733  -1.365 0.205395    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.309 on 9 degrees of freedom
# Multiple R-squared:  0.9823,	Adjusted R-squared:  0.9764 
# F-statistic: 166.8 on 3 and 9 DF,  p-value: 3.323e-08

# 해석
# p-value는 통계적으로 유의하게 나옴
# 그러나 상세 변수를 보면 x2, x4는 유의하지 않음
# 유의확률이 가장 높은 x4 제외 후 회귀모형 재생성

c <- lm(y~x1+x2, data=df)
summary(c)
# Call:
#   lm(formula = y ~ x1 + x2, data = df)
# 
# Residuals:
#   Min     1Q Median     3Q    Max 
# -2.893 -1.574 -1.302  1.363  4.048 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 52.57735    2.28617   23.00 5.46e-10 ***
#   x1           1.46831    0.12130   12.11 2.69e-07 ***
#   x2           0.66225    0.04585   14.44 5.03e-08 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.406 on 10 degrees of freedom
# Multiple R-squared:  0.9787,	Adjusted R-squared:  0.9744 
# F-statistic: 229.5 on 2 and 10 DF,  p-value: 4.407e-09

# 해석
# Adjusted R-squared : 0.9744
# 선정된 다변량회귀식이 전체 데이터의 97.44% 설명하고 있음


### step 함수를 이용한 전진선택법
step(lm(y~1, data=df), scope=list(lower=~1, upper=~x1+x2+x3+x4), direction="forward")
# Start:  AIC=71.44
# y ~ 1
# 
#       Df  Sum of Sq     RSS    AIC
# + x4    1   1831.90  883.87 58.852
# + x2    1   1809.43  906.34 59.178
# + x1    1   1450.08 1265.69 63.519
# + x3    1    776.36 1939.40 69.067
# <none>              2715.76 71.444
# 
# Step:  AIC=58.85
# y ~ x4
# 
#       Df   Sum of Sq    RSS    AIC
# + x1    1    809.10  74.76 28.742
# + x3    1    708.13 175.74 39.853
# <none>              883.87 58.852
# + x2    1     14.99 868.88 60.629
# 
# Step:  AIC=28.74
# y ~ x4 + x1
# 
#       Df  Sum of Sq    RSS    AIC
# + x2    1    26.789 47.973 24.974
# + x3    1    23.926 50.836 25.728
# <none>              74.762 28.742
# 
# Step:  AIC=24.97
# y ~ x4 + x1 + x2
# 
#       Df  Sum of Sq    RSS    AIC
# <none>              47.973 24.974
# + x3    1   0.10909 47.864 26.944
# 
# Call:
#   lm(formula = y ~ x4 + x1 + x2, data = df)
# 
# Coefficients:
#   (Intercept)           x4           x1           x2  
#       71.6483      -0.2365       1.4519       0.4161 

# 해석
# 가장 먼저 선택된 변수 : AIC=58.852로 가장 낮은 x4
# x4에 x1을 추가 : AIC=28.742로 최소화됨
#     -> 더이상 AIC 낮추기 불가능 -> 변수선택 종료
# 최종 회귀식 : y = 71.6483 - 0.2365x4 + 1.4519x1 + 0.4161x2


### step 함수를 이용한 후진제거법
# 책 예시 : ElemStatLearn
# 전립선암 자료 : lcavol(종양부피로그), lweight(전립선무게로그), age, lbph(양성전립선증식량로그),
#                 svi(암이 정낭을 침범할 확률), lcp(capsular penetration의 로그값),
#                 gleason(Gleason 점수), pgg45(Gleason 점수 4 or 5 비율), lpsa (전립선 수치의 로그)
# 원래 'CRAN' package에 있는 dataset인데, 현재 package에서 삭제 됨
# -> 수동설치 필요
# Download Link : https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/

# tar.gz 파일 다운받아서 수동 설치하기
install.packages("C:\\Users\\User\\Documents\\R\\win-library\\4.0\\ElemStatLearn_2015.6.26.2.tar.gz", repos=NULL, type="source") 
library(ElemStatLearn)

Data = prostate
data.use = Data[,-ncol(Data)]  # 마지막 컬럼 (lpsa) 제거
lm.full.Model = lm(lpsa~., data=data.use)

backward.aic = step(lm.full.Model, lpsa~1, direction="backward")
# Start:  AIC=-60.78
# lpsa ~ lcavol + lweight + age + lbph + svi + lcp + gleason + 
#   pgg45
# 
#           Df Sum of Sq    RSS     AIC
# - gleason  1    0.0491 43.108 -62.668
# - pgg45    1    0.5102 43.569 -61.636
# - lcp      1    0.6814 43.740 -61.256
# <none>                 43.058 -60.779
# - lbph     1    1.3646 44.423 -59.753
# - age      1    1.7981 44.857 -58.810
# - lweight  1    4.6907 47.749 -52.749
# - svi      1    4.8803 47.939 -52.364
# - lcavol   1   20.1994 63.258 -25.467
# 
# Step:  AIC=-62.67
# lpsa ~ lcavol + lweight + age + lbph + svi + lcp + pgg45
# 
#           Df Sum of Sq    RSS     AIC
# - lcp      1    0.6684 43.776 -63.176
# <none>                 43.108 -62.668
# - pgg45    1    1.1987 44.306 -62.008
# - lbph     1    1.3844 44.492 -61.602
# - age      1    1.7579 44.865 -60.791
# - lweight  1    4.6429 47.751 -54.746
# - svi      1    4.8333 47.941 -54.360
# - lcavol   1   21.3191 64.427 -25.691
# 
# Step:  AIC=-63.18
# lpsa ~ lcavol + lweight + age + lbph + svi + pgg45
# 
#           Df Sum of Sq    RSS     AIC
# - pgg45    1    0.6607 44.437 -63.723
# <none>                 43.776 -63.176
# - lbph     1    1.3329 45.109 -62.266
# - age      1    1.4878 45.264 -61.934
# - svi      1    4.1766 47.953 -56.336
# - lweight  1    4.6553 48.431 -55.373
# - lcavol   1   22.7555 66.531 -24.572
# 
# Step:  AIC=-63.72
# lpsa ~ lcavol + lweight + age + lbph + svi
# 
#           Df Sum of Sq    RSS     AIC
# <none>                 44.437 -63.723
# - age      1    1.1588 45.595 -63.226
# - lbph     1    1.5087 45.945 -62.484
# - lweight  1    4.3140 48.751 -56.735
# - svi      1    5.8509 50.288 -53.724
# - lcavol   1   25.9427 70.379 -21.119

# 해석
# 맨처음 AIC는 -62.67로 gleason 제거하고 회귀분석 실시
# 다음으로 lcp, pgg45 순서로 제거
