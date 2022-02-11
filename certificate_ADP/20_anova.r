
#===============================================================================================================
# 분산분석 ANOVA
# 두 개 이상 집단 간 평균 차이
#
# R에서 분산분석 수행 시 주의점
# ★ 그룹 구분 기준 변수는 반드시 factor형이어야 함
#
# 1. 단일변량 분산분석
# 1) 일원배치 분산분석 One-way ANOVA
#     독립변수 1개, 종속변수 1개
# 2) 이원배치 분산분석 Two-way ANOVA
#     독립변수 2개, 종속변수 1개
# 3) 다원배치 분산분석 Multi-way ANOVA
#     독립변수 3개 이상, 종속변수 1개
#
# 2. 다변량 분산분석 MANOVA
#    독립변수 1개 이상, 종속변수 2개 이상
#  
# aov(formula, data)
# formula : 반응변수 ~ 그룹변수
# data : 분석하고자 하는 데이터명
#
# 사후검정
# 적어도 한 집단에서 평균 차이가 있을 때, 어떤 집단 간 차이인지 알아보기 위한 분석
# 종류
# - Duncan의 MRT (Multiple Range Test)
# - Fisher의 최소유의치 (LSD)
# - Tukey의 HSD
# - Scheffe의 방법
#
# TukeyHSD(x, conf.level=0.95, ...)
# x : 분산분석의 결과
# conf.level : 신뢰수준 (default=0.95)
#===============================================================================================================

# 1. 일원배치 분산분석
# H0 귀무가설 : 세 가지 종에 대해 Sepal.Width의 평균은 모두 같다
# H1 대립가설 : 적어도 하나의 종에 대한 Sepal.Width의 평균값에는 차이가 있다

str(iris)
rst <- aov(Sepal.Width~Species, data=iris)

# 분산분석표 확인
# SSA 자유도 : 2 (집단수 - 1 = 3-1 = 2)
# SSE 자유도 : 147 (관측값수 - 집단수 = 150-3 = 147)
# p-value < 0.05 -> 모두 동일하지는 않다
summary(rst)
#               Df Sum Sq Mean Sq F value Pr(>F)    
# Species       2  11.35   5.672   49.16 <2e-16 ***
# Residuals   147  16.96   0.115                   
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# 사후분석
# 모든 집단 수준에 대해 두 집단씩 짝을 지어 각각 다중 비교 수행
# 수정된 p-value (p adj) < 0.05
# -> 모든 종들에 대해 유의한 차이가 있음
# ex) versicolor-setosa : 음수
#     -> versicolor보다 setosa일 때 통계적으로 유의하게 큰 값을 가짐
TukeyHSD(aov(Sepal.Width~Species, data=iris))
# Tukey multiple comparisons of means
# 95% family-wise confidence level
# 
# Fit: aov(formula = Sepal.Width ~ Species, data = iris)
# 
# $Species
# diff         lwr        upr     p adj
# versicolor-setosa    -0.658 -0.81885528 -0.4971447 0.0000000
# virginica-setosa     -0.454 -0.61485528 -0.2931447 0.0000000
# virginica-versicolor  0.204  0.04314472  0.3648553 0.0087802

