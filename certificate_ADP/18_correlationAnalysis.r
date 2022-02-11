
#=====================================
# 상관관계 특성
# 0.7 < r <= 1 : 강한 +
# 0.3 < r <= 0.7 : 약한 +
# 0 < r <= 0.3 : 거의 상관X
# r = 0 : 상관관계가 없다
# -0.3 <= r < 0 : 거의 상관X
# -0.7 <= r < -0.3 : 약한 -
# -1 <= r < -0.7 : 강한 -
#
# 피어슨
# 등간척도 이상으로 측정된 변수 간 상관관계
# 연속형 변수, 정규성 가정
# 대부분 많이 사용
# 피어슨 r (적률상관계수)
#
# 스피어만
# 서열척도 변수 간 상관관계
# 순서형 변수, 비모수적 방법
# 순위를 기준으로 상관관계 측정
# 순위상관계수 (p, 로우)
#
# 분산
# var(x, y=NULL, na.rm=FASLE)
#
# 공분산
# cov(x, y=NULL, use="everything", method=c("pearson","kendall","spearman"))
#
# 상관관계
# 1) cor(x, y=NULL, use="everything", method=c("pearson","kendall","spearman"))
# 2) Hmisc 패키지의 rcorr 사용
#    rcorr(matrix(data명), type=c("pearson","kendall","spearman"))
#
# x : 숫자형 변수
# y : NULL(default) or 변수
# na.rm : 결측값 처리
#=====================================

# 'mtcars' dataset에서 mpg, hp 상관관계 구하기

data(mtcars)
a <- mtcars$mpg
b <- mtcars$hp

# 해석 : mpg와 hp는 음의 상관관계
cov(a,b)    # 공분산 : -320.7321
cor(a,b)    # 상관계수 : -0.7761684 -> 강한 음의 상관관계

# 해석 : p-value가 유의수준 0.05보다 작으므로 두 변수 간 상관관계 있다고 판단 가능
cor.test(a,b,method="pearson")
# Pearson's product-moment correlation
# 
# data:  a and b
# t = -6.7424, df = 30, p-value = 1.788e-07
# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
#  -0.8852686 -0.5860994
# sample estimates:
#        cor 
# -0.7761684

