
#=====================================
# 일표본 t-검정 : one sample t-test
# 단일모집단에서 관심이 있는 연속형 변수의 평균 값을 특정 기준값과 비교할 때
# 모집단의 구성요소들이 정규분포를 이룬다는 가정 필요
#
# 귀무가설 H0 : 모평균의 값은 u0이다 (u=u0)
# 대립가설 H1
# 1) 모평균의 값은 u0이 아니다 (u <> u0) - 양측검정
# 2) 모평균의 값은 u0보다 크다 (u > u0) - 단측검정
# 3) 모평균의 값은 u0보다 작다 (u < u0) - 단측검정
#
# t.test(x, alternative=c("two.sided","less","greater),mu=0)
# x : 표본으로부터 관측한 값 (수치형 벡터)
# alternative : 양측검정시 "two.sided", 단측검정시 "less", "greater" 입력
# mu : 검정 시 기준이 되는 값
#=====================================

# H0 귀무가설 : A 과수원에서 생산되는 사과무게의 평균값은 200g이다
# H1 대립가설 : A 과수원에서 생산되는 사과무게의 평균값은 200g이 아니다

data <- c(200,210,180,190,185,170,180,
          180,210,180,183,191,204,201,186)

# 표본 크기가 충분히 크지 않아, 중심극한정리를 따른다고 보기 어려움
# 정규성 검정 실시 필요
# 방법 : 샤피로-윌크 검정, 콜모고로프 스미르노프 검정, Q-Q도

# 샤피로-윌크 검정 Shapiro-Wilk normality test
# -> p-value가 0.2047로 유의수준 0.05보다 높으므로
#    이 데이터가 정규분포를 따르고 있다고 할 수 있음
shapiro.test(data)
# Shapiro-Wilk normality test
# 
# data:  data
# W = 0.92173, p-value = 0.2047

# 검정통계량(t값) = -3.1563
# 자유도(df) = 14
# 유의확률 (p-value) < 0.05
# -> 대립가설 채택
#    A 과수원에서 생산되는 사과의 평균무게는 200g이 아니다
t.test(data, alternative="two.sided", mu=200)
# One Sample t-test
# 
# data:  data
# t = -3.1563, df = 14, p-value = 0.007004
# alternative hypothesis: true mean is not equal to 200
# 95 percent confidence interval:
#   183.2047 196.7953
# sample estimates:
#   mean of x 
# 190 

