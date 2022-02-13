
#===============================================================================================================
# 로지스틱 회귀분석 Logistic Regression
# 반응변수 : 범주형
#
# 사후확률 : 모형의 적합을 통해 추정된 확률
# exp(B1) : x1이 한 단위 증가할 때마다 성공(Y=1)의 odds가 몇 배 증가하는지를 나타내는 값
#
# odds : 성공할 확률이 실패할 확률의 몇 배인지를 나타내는 확률 (성공/실패)
# odds ratio : odds의 비율
#===============================================================================================================

# glm(종속변수~독립변수1+...+독립변수k, data=dataset, family=binomial)

a <- iris[iris$Species=="setosa"|iris$Species=="versicolor",]
b <- glm(Species ~ Sepal.Length, data=a, family=binomial)

summary(b)
# Call:
#   glm(formula = Species ~ Sepal.Length, family = binomial, data = a)
# 
# Deviance Residuals: 
#   Min        1Q    Median        3Q       Max  
# -2.05501  -0.47395  -0.02829   0.39788   2.32915  
# 
# Coefficients:
#               Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)   -27.831      5.434  -5.122 3.02e-07 ***
#   Sepal.Length    5.140      1.007   5.107 3.28e-07 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 138.629  on 99  degrees of freedom
# Residual deviance:  64.211  on 98  degrees of freedom
# AIC: 68.211
# 
# Number of Fisher Scoring iterations: 6

# 해석
#
# 종속변수 : Species, 독립변수 : Sepal.Length
# Sepal.Length 한 단위 증가하면, Species(y)가 1에서 2로 바뀔 때 Odds가 exp(5.140)=170배 증가
#
# NULL deviance : 절편만 포함하는 모형의 완전 모형으로부터의 이탈도(deviance) 나타냄
# p-value = P(x^2(99)>138.629)=0.005 < 0.05 : 통계적으로 유의하므로 적합결여 나타냄
#
# Residual deviance : 예측변수 Sepal.Length가 추가된 적합 모형의 이탈도 나타냄
# 자유도 1 기준에 이탈도 감소 74.4로 큰 감소를 보임
# p-value = P(x^2(98)>64.211) = 0.997 > 0.05 : 통계적으로 유의하지 못함 -> 귀무가설 기각 불가
#
# 결론 : 적합값이 관측된 자료를 잘 적합한다고 말할 수 있음
