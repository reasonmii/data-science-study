
#===============================================================================================================
# 데이터 시각화
#===============================================================================================================

install.packages("ggplot2")
library(ggplot2)

### XY graph ---------------------------------------------------------------------------------------------------
# 전체적인 내용 파악 가능
# but, 수많은 데이터가 있을 때는 의미 파악 어려움

# ChickWeight dataset
data(ChickWeight)
head(ChickWeight)
#     weight Time Chick Diet
# 1     42    0     1    1
# 2     51    2     1    1
# 3     59    4     1    1
# 4     64    6     1    1
# 5     76    8     1    1
# 6     93   10     1    1

# 1) 기본 XY graph
# X : time / y : weight
# line color : Diet
# aes()
# - x축, y축 지정
# - colour 색 지정
# - group 각 행마다 그래프 그리기
# geom_line() : ggplot에서 선 그래프 그리기
ggplot(ChickWeight, aes(x=Time, y=weight, colour=Diet, group=Chick))+
  geom_line()

# 결과
# 먹이(diet)별 체중 변화가 있지만, 어떤 먹이가 효율적인지 알기는 어려움

# 2) 그래프 개선하기
# Point graph : geom_point()
#   - alpha : 점의 투명도
#   - size : 점의 크기
# Smooth graph : geom_smooth()
#   - alpha : 배경 색상의 투명도
#   - size : 평균 값 선의 굵기
ggplot(ChickWeight, aes(x=Time, y=weight, colour=Diet))+
  geom_point(alpha=.3)+
  geom_smooth(alpha=.2, size=1)


### Histogram --------------------------------------------------------------------------------------------------
# 도수분포표를 그래프로 나타낸 것
# 분포가 연속적인 값이고 선으로 되어 있어서 내용 파악이 어려운 경우
# 분류유형이 많은 경우

# subset(ChickWeight, Time=21) : ChickWeight data의 Time 변수가 21인 행만 선택
# geom_histogram(colour="black", binwidht=50) : 막대 테두리 색상, 막대 너비
# facet_grid(Diet~.) : 위에서 아래로 Diet 유형에 따라 분리
# facet_grid(.~Diet) : 좌에서 우로 Diet 유형에 따라 분리
ggplot(subset(ChickWeight, Time=21), aes(x=weight, fill=Diet))+
  geom_histogram(colour="black", binwidth=50)+
  facet_grid(Diet~.)

# Histogram 형식에 색상 적용하기
# movies dataset
install.packages("ggplot2movies")
library(ggplot2movies)
ggplot(movies, aes(x=rating))+
  geom_histogram()+
  geom_histogram(aes(fill=..count..)) # count를 색상으로 표시

# diamonds dataset
data(diamonds)
head(diamonds)
# # A tibble: 6 x 10
#   carat cut       color clarity depth table price     x     y     z
#   <dbl> <ord>     <ord> <ord>   <dbl> <dbl> <int> <dbl> <dbl> <dbl>
# 1 0.23  Ideal     E     SI2      61.5    55   326  3.95  3.98  2.43
# 2 0.21  Premium   E     SI1      59.8    61   326  3.89  3.84  2.31
# 3 0.23  Good      E     VS1      56.9    65   327  4.05  4.07  2.31
# 4 0.290 Premium   I     VS2      62.4    58   334  4.2   4.23  2.63
# 5 0.31  Good      J     SI2      63.3    58   335  4.34  4.35  2.75
# 6 0.24  Very Good J     VVS2     62.8    57   336  3.94  3.96  2.48

# cut 등급별 carat
ggplot(diamonds, aes(carat, ..density..))+
  geom_histogram(bandwidth=0.2)+
  facet_grid(.~cut)


### Point graph ------------------------------------------------------------------------------------------------
# 가장 간단하게 데이터를 정적으로 보여줌
# 유형별로 색상을 다르게 하여 특성 파악 가능

# rnorm : 정규분포 난수 생성 (평균 0, 표준편차 1)
# alpha : 투명도
df <- data.frame(x=rnorm(5000), y=rnorm(5000))
ggplot(df, aes(x,y))+
  geom_point(alpha=1/10)

# mtcars dataset
data(mtcars)
head(mtcars)
#                    mpg cyl disp  hp drat    wt  qsec vs am gear carb
# Mazda RX4         21.0   6  160 110 3.90 2.620 16.46  0  1    4    4
# Mazda RX4 Wag     21.0   6  160 110 3.90 2.875 17.02  0  1    4    4
# Datsun 710        22.8   4  108  93 3.85 2.320 18.61  1  1    4    1
# Hornet 4 Drive    21.4   6  258 110 3.08 3.215 19.44  1  0    3    1
# Hornet Sportabout 18.7   8  360 175 3.15 3.440 17.02  0  0    3    2
# Valiant           18.1   6  225 105 2.76 3.460 20.22  1  0    3    1

p = qplot(wt, mpg, colour=hp, data=mtcars)
p+coord_cartesian(ylim=c(0,40))              # y축 0~40 범위지정
p+scale_colour_continuous(breaks=c(100,300)) # hp 범위 100~300 지정
p+guides(colour="colourbar")                 # hp 수치에 따른 색 범위 표시

# 치환 데이터를 이용한 포인트 그래프
# 특정 데이터만 이용해 그리기 (데이터가 너무 많은 경우)
m <- mtcars[1:10,]   # 10개만 추출
p%+%m                # 앞서 그린 전체 그래프에서 m만 남기기 (filter)

# 깔끔 기본
p <- ggplot(mtcars, aes(wt, mpg))
p+geom_point()

# 모양 할당
p+geom_point(shape=5)

# 모양 할당 - 문자
# 문자 : 문자, 숫자, 특수문자(.) 모두 가능
# 포인트를 사이즈가 3인 "k" 문자 모양으로 바꾸기
p+geom_point(shape="k", size=3)

# 25가지 모양 할당
df <- data.frame(x=1:5, y=1:25, z=1:25)
ggplot(df,aes(x=x,y=y))+
  geom_point(aes(shape=z),size=4)+
  scale_shape_identity()

# 모양 없애기
# -> 빈그래프 출력됨
p+geom_point(shape=NA)

# 요인 별 색 할당
# colour : cyl 변수에 따른 색
p+geom_point(aes(colour=factor(cyl)),size=4)

# 요인 별 모양 할당
# shape : cyl 변수에 따른 모양
p+geom_point(aes(shape=factor(cyl)),size=4)

# 요인 별 크기 할당
# size : qsec 변수에 따른 모양
p+geom_point(aes(size=qsec))

# 임의의 선 삽입
# y=25
# size : 선 굵기
p+geom_point(size=2.5)+
  geom_hline(yintercept=25, size=3.5)

# Point graph with Box
# 특정 부분 강조
# rect : rectangle
ggplot(mtcars, aes(wt, mpg))+
  geom_point()+
  annotate("rect", xmin=2, xmax=3.5, ymin=2, ymax=25, fill="dark grey", alpha=.5)


### Bar graph --------------------------------------------------------------------------------------------------
# ★ factor(변수) : 범주형 변수는 factor로 전환해야 함

# factor(cyl) : 범주형 변수(cyl) factor로 전환
# fill : 막대 내부 색상
# colour : 막대 테두리 색상
ggplot(mtcars, aes(factor(cyl)))+
  geom_bar(fill="white", colour="red")

# diamonds dataset
data(diamonds)
head(diamonds)
# # A tibble: 6 x 10
#   carat cut       color clarity depth table price     x     y     z
#   <dbl> <ord>     <ord> <ord>   <dbl> <dbl> <int> <dbl> <dbl> <dbl>
# 1 0.23  Ideal     E     SI2      61.5    55   326  3.95  3.98  2.43
# 2 0.21  Premium   E     SI1      59.8    61   326  3.89  3.84  2.31
# 3 0.23  Good      E     VS1      56.9    65   327  4.05  4.07  2.31
# 4 0.290 Premium   I     VS2      62.4    58   334  4.2   4.23  2.63
# 5 0.31  Good      J     SI2      63.3    58   335  4.34  4.35  2.75
# 6 0.24  Very Good J     VVS2     62.8    57   336  3.94  3.96  2.48

ggplot(diamonds, aes(clarity, fill=cut))+
  geom_bar()
#geom_bar(aes(order=desc(cut)))


### Line graph -------------------------------------------------------------------------------------------------
# 주로 시계열 지표에서 사용 (시간의 흐름에 따른 등락)

data(economics) # R 내장 시계열데이터
head(economics)
# A tibble: 6 x 6
#   date         pce    pop psavert uempmed unemploy
#   <date>     <dbl>  <dbl>   <dbl>   <dbl>    <dbl>
#   1 1967-07-01  507. 198712    12.6     4.5     2944
# 2 1967-08-01  510. 198911    12.6     4.7     2945
# 3 1967-09-01  516. 199113    11.9     4.6     2958
# 4 1967-10-01  512. 199311    12.9     4.9     3143
# 5 1967-11-01  517. 199498    12.8     4.7     3066
# 6 1967-12-01  525. 199657    11.8     4.8     3018

# geom_line
# - colour : 선의 색상
# - size : 선의 굵기
# - linetype : 선의 종류
#   (1 : 실선, 2 : 선이 긴 점선, 3 : 선이 짧은 점선, 4 : 선이 긹고 짧음이 반복되는 점선)
ggplot(economics, aes(x=date, y=unemploy))+
  geom_line(colour="blue", size=0.3, linetype=3)

df <- data.frame(x=1:10,y=1:10)
ggplot(df, aes(x=x,y=y))+
  geom_line(linetype=2)+
  geom_line(linetype="dotdash")


### Boxplot ----------------------------------------------------------------------------------------------------
# 가운데 : 중앙값
# 아래선 : 1사분위수, 상단선 : 4사분위수
# 점 : 이상값
qplot(cut, price, data=diamonds, geom="boxplot")

# 가로로 그리기
last_plot()+
  coord_flip()


### Others -----------------------------------------------------------------------------------------------------

# 1. 선형모델링
# cut 정보를 이용해 다이아몬드 가격 예측
# 값의 범위 : point range 그래프로 표현
# se : standard error
dmod <- lm(price~cut, data=diamonds)
cuts <- data.frame(cut=unique(diamonds$cut),
                   predict(dmod, data.frame(cut=unique(diamonds$cut)), se=TRUE)[c("fit","se.fit")])

ggplot(cuts, aes(x=cut, y=fit, ymin=fit-se.fit, ymax=fit+se.fit, colour=cut))+
  geom_pointrange()

# 2. 축의 범위 지정
# 원하는 범위에서만 그래프 그리기
qplot(disp, wt, data=mtcars)+
  geom_smooth()+
  scale_x_continuous(limits=c(325,500))

# 3. qplot
qplot(cut, data=diamonds, geom="bar")


### aplpack ----------------------------------------------------------------------------------------------------
# 줄기-잎 그림, 체르노프 페이스, 스타차트 등 제공

install.packages("aplpack")
library(aplpack)

# 1. 줄기-잎 그림
# 22명의 1학기 중간고사 수학 성적 데이터
score <- c(1,2,3,4,10,2,30,42,31,50,80,76,90,87,21,43,65,76,32,12,34,54)
stem.leaf(score)
# 1 | 2: represents 12
# leaf unit: 1
#          n: 22
# 5    0 | 12234
# 7    1 | 02
# 8    2 | 1
# (4)  3 | 0124
# 10   4 | 23
# 8    5 | 04
# 6    6 | 5
# 5    7 | 66
# 3    8 | 07
# 1    9 | 0

# 2. 얼굴 그림
faces(WorldPhones)

# 3. 별 그림
stars(WorldPhones)


### Multiple Axis ----------------------------------------------------------------------------------------------
# 2중축, 3중축 그리기
# 모든 과정은 그래프 창을 열어놓은 상태에서 진행되어야 함

# 1. 사용 data 입력
time <- seq(7000,3400,-200)
pop <- c(200,400,450,500,300,100,400,700,830,1200,400,350,200,700,370,800,200,100,120)
grp <- c(2,5,8,3,2,2,4,7,9,4,4,2,2,7,5,12,5,4,4)
med <- c(1.2,1.3,1.2,0.9,2.1,1.4,2.9,3.4,2.1,1.1,1.2,1.5,1.2,0.9,0.5,3.3,2.2,1.1,1.2)

# par() : 그래픽 인수를 설정하고 조회하는 함수
# 그래프 모양을 다양하게 조절
# mar : margin 여백 지정 (default=c(5,4,4,2)+0.1)
# new : 새로운 그래픽 함수 호출
# - FALSE (default) : 현재 figure region 초기화 후 새 그래프 그림
# - TRUE : 화면 분할 없이 현 그래프에 그림 추가
# 한 화면에서 여러 그래프 비교 시
# - mfrow : 행 우선 배치 / mfcol : 열 우선 배치
par(mar=c(5,12,4,4)+0.1)

# 2. 다축 생성 절차

# axes=F (축지정X)
plot(time, pop, axes=F, xlim=c(7000,3400), ylim=c(0,max(pop)),
     xlab="", ylab="", type="l", col="black", main="",)      # 1) 첫 번째 그래프 생성
points(time, pop, pch=20, col="black")                       # 2) 점 추가
axis(2, ylim=c(0,max(pop)), col="black", lwd=2)              # 3) y축 생성
mtext(2, text="Population", line=2)                          # 4) y축 이름 지정

par(new=T)                                                   # 5) 두 번째 그래프 추가
plot(time, med, axes=F, xlim=c(7000,3400), ylim=c(0,max(med)),
     xlab="", ylab="", type="l", col="black", lty=2, lwd=2, main="",)
points(time, med, pch=20, col="black")                       # 6) 점 추가
axis(2, ylim=c(0,max(med)), col="black", lwd=2, line=3.5)    # 7) y축 생성
mtext(2, text="Median Group size", line=5.5)                 # 8) y축 이름 지정

par(new=T)                                                   # 9) 세 번째 그래프 추가
plot(time, grp, axes=F, xlim=c(7000,3400), ylim=c(0,max(grp)),
     xlab="", ylab="", type="l", col="black", lty=3, lwd=2, main="",)
points(time, grp, pch=20, col="black")                       # 10) 점 추가
axis(2, ylim=c(0,max(grp)), col="black", lwd=2, line=7)      # 11) y축 생성
mtext(2, text="Number of Groups", line=9)                    # 12) y축 이름 지정

# pretty : computes a sequence of equally spaced round values.
axis(1, pretty(range(time), 10))                             # 13) x축 추가
mtext(1, text="cal BP", col="black", line=2)                 # 14) x축 이름 지정
legend(x=7000, y=12,
       legend=c("Population","Median Group size", "Number of Groups"),
       lty=c(1,2,3))                                         # 15) 범례 추가
