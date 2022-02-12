
#===============================================================================================================
# 다차원척도법
# 객체 간 유사성/비유사성 측정하여 2차원 or 3차원 공간상에 점으로 표현
#
# 목적
# 1) 패턴발굴
# 2) 구조를 공간에 기하학적으로 표현
# 3) 데이터 축소
#===============================================================================================================

# 1) 계량적 MDS (Metric MDS)
# 구간척도, 비율척도인 경우 활용

# cmdscale 사례
# MASS package - eurodist dataset
# 유럽의 21개 도시 간 거리 측정
# cmdscale 이용해서 2차원으로 21개 도시 mapping
# 종축은 북쪽 도시를 상단에 표시하기 위해 부호 바꾸기

library(MASS)
loc <- cmdscale(eurodist)
x <- loc[,1]
y <- -loc[,2]

# ★ type="n" : create the graph without plotting the points
# asp : 종횡의 비율 (aspect ratio)
# abline : 점선
# - a/b/h/v : 절편, 기울기, 수평선일 때 y값, 수직선일 때 x값
# - lty : line type 선모양
# - lwd : line width 선굵기
plot(x,y, type="n", asp=1, main="Metric MDS")
text(x,y, rownames(loc), cex=0.7)
abline(v=0, h=0, lty=2, lwd=0.5)

# 2) 비계량적 MDS (nonmetric MDS)
# 순서척도인 경우 활용

# isoMDS 사례
# MASS package - Swiss dataset
# 1888년경 스위스 연방 47개 불어권 주의 토양 비옥도 지수와 여러 사회경제적 지표를 측정한 자료
# 2차원으로 도시 mapping

library(MASS)
data(swiss)
swiss.x <- as.matrix(swiss[,-1])
swiss.dist <- dist(swiss.x)
swiss.mds <- isoMDS(swiss.dist)
plot(swiss.mds$points, type="n")
text(swiss.mds$points, labels=as.character(1:nrow(swiss.x)))
abline(v=0, h=0, lty=2, lwd=0.5)

# sammon : One form of non-metric multidimensional scaling
swiss.x <- as.matrix(swiss[,-1])
swiss.sammon <- sammon(dist(swiss.x))
plot(swiss.sammon$points, type="n")
text(swiss.sammon$points, labels=as.character(1:nrow(swiss.x)))
abline(v=0, h=0, lty=2, lwd=0.5)
