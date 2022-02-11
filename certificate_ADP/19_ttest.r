
#===============================================================================================================
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
#===============================================================================================================

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



#===============================================================================================================
# 대응표본 t-검정 : paired sample t-test = matched pair t-test
# 단일모집단에 대해 두 번의 처리를 가했을 때, 두 개의 처리에 따른 평균 차이 비교
# 관측값들은 서로 독립적이지 않고 쌍(pair)으로 이루어짐
#
# 모수 : 두 개의 모평균 차이 (D)
# 귀무가설 H0 : 두 모평균 간 차이가 없다 (ux-uy=D=0)
# 대립가설 H1
# 1) 두 모평균 간 차이가 있다 (ux-uy=D <> 0) - 양측검정
# 2) 두 모평균 간 차이가 0보다 크다 (ux-uy=D > 0) - 단측검정
# 3) 두 모평균 간 차이가 0보다 작다 (ux-uy=D < 0) - 단측검정
#
# t.test(x, y, alternative=c("two.sided","less","greater),paired=FALSE,m=0)
# x : 처리방법이 x일 때의 관측값 (수치형 벡터)
# y : 처리방법이 y일 때의 관측값 (수치형 벡터)
# alternative : 양측검정시 "two.sided", 단측검정시 "less", "greater" 입력
# paired : 대응표본 t-검정을 수행할지에 대한 여부 (★인자값을 TRUE로 지정)
# m : 검정 시 기준이 되는 값 (default=0)
#     대응표본 t-검정에서는 모평균의 차이가 0인지를 검정하기 때문에
#     m 인자는 적지 않아도 됨
#===============================================================================================================

# H0 귀무가설 : 수면영양제를 복용하기 전과 후의 평균 수면시간에는 차이가 없다 (D=0)
# H1 대립가설 : 수면영양제를 복용하기 전과 후의 평균 수면시간 차이는 0보다 작다 (D<0)

# 10명의 환자에 대해 영양제 복용 전과 후 수면시간 data
data <- data.frame(before = c(7,3,4,5,2,1,6,6,5,4),
                   after = c(8,4,5,6,2,3,6,8,6,5))

# 검정통계량(t값) = -4.7434
# 자유도(df) = 9
# 유의확률 (p-value) < 0.05
# -> 대립가설 채택 : 영양제 복용 후 수면시간이 늘었다
t.test(data$before, data$after, althernative="less", paired=TRUE)
# Paired t-test
# 
# data:  data$before and data$after
# t = -4.7434, df = 9, p-value = 0.001054
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -1.4769046 -0.5230954
# sample estimates:
#   mean of the differences 
# -1 



#===============================================================================================================
# 독립표본 t-검정 : independent sample t-test
# 두 개의 독립된 모집단의 평균 비교
# ex) 성별에 따른 출근 준비시간 차이
# 1) 두 모집단은 정규성을 만족해야 함
#    표본의 크기가 충분히 크다면 중심극한정리에 따라 정규성 만족
# 2) 두 모집단은 서로 독립적이어야 함
# 3) 두 독립 집단의 모분산이 동일해야 함
#    -> ★ t-검정 전 등분산 검정을 통해 모분산이 동일한지 반드시 확인 필요
#
# 모수 : 서로 독립된 두 모집단의 평군 (u1, u2)
# 귀무가설 H0 : 두 모평균 간 차이가 없다 (u1=u2)
# 대립가설 H1
# 1) 두 모평균 간 차이가 있다 (u1 <> u2) - 양측검정
# 2) 집단1 모평균이 집단2 모평균보다 크다 (u1 > u2) - 단측검정
# 3) 집단1 모평균이 집단2 모평균보다 작다 (u1 < u2) - 단측검정
#
# 등분산검정
# 귀무가설 H0 : 두 집단의 분산이 동일하다
# 대립가설 H1 : 두 집단의 분산이 동일하지 않다
# var.test(x, y, alternative)
# var.test(formula, data, alternative)
# x : 모집단1로부터 측정한 관측값 (수치형 벡터)
# y : 모집단2로부터 측정한 관측값 (수치형 벡터)
# formula : 수치형 벡터(종속변수)~집단분류(독립변수)
#           데이터프레임을 var.test 함수에 적용시킬 때 사용
# data : 등분산 검정을 수행할 데이터명
# alternative : 양측검정시 "two.sided", 단측검정시 "less", "greater" 입력
#
# t.test(x, y, alternative, var.equal=FALSE)
# t.test(formula, data, alternative, var.equal=FALSE)
# x : 모집단1로부터 측정한 관측값 (수치형 벡터)
# y : 모집단2로부터 측정한 관측값 (수치형 벡터)
# formula : 수치형 벡터(종속변수)~집단분류(독립변수)
#           데이터프레임을 var.test 함수에 적용시킬 때 사용
# data : t-검정을 수행할 데이터명
# alternative : 양측검정시 "two.sided", 단측검정시 "less", "greater" 입력
# var.equal : 등분산성을 만족하는지의 여부 (TRUE/FALSE)
#===============================================================================================================

# H0 귀무가설 : A, B 두 지역에 따른 겨울 낮 최고기온은 차이가 없다 (u1 = u2)
# H1 대립가설 : A, B 두 지역에 따른 겨울 낮 최고기온은 차이가 있다 (u1 <> u2)

group <- factor(rep(c("A","B"),each=10))          # 집단구분을 위한 변수
A <- c(-1,0,3,4,1,3,3,1,1,3)                      # A지역의 온도
B <- c(6,6,8,8,11,11,10,8,8,9)                    # B지역의 온도
weather <- data.frame(group=group, temp=c(A,B))   # dataFrame 생성

# p-value > 0.05 : 등분산 가정을 만족함
var.test(temp~group, data=weather)
# F test to compare two variances
# 
# data:  temp by group
# F = 0.82807, num df = 9, denom df = 9, p-value = 0.7833
# alternative hypothesis: true ratio of variances is not equal to 1
# 95 percent confidence interval:
#   0.2056809 3.3338057
# sample estimates:
#   ratio of variances 
# 0.8280702 

# 검정통계량(t값) = -8.806
# 자유도(df) = 18
# 유의확률 (p-value) < 0.05
# -> 대립가설 채택 :  A, B 두 지역에 따른 겨울 낮 최고기온은 차이가 있다
t.test(temp~group, data=weather, alternative="two.sided", var.equal=TRUE)
# Two Sample t-test
# 
# data:  temp by group
# t = -8.806, df = 18, p-value = 6.085e-08
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -8.298481 -5.101519
# sample estimates:
#   mean in group A mean in group B 
# 1.8             8.5 


