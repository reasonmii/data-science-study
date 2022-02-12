
#===============================================================================================================
# 주성분분석 Principal Component Analysis
# 서로 상관성이 높은 변수들의 선형 결합 -> 데이터 요약, 축소
# 
# ※ 요인분석 Factor Analysis
# 두 개 이상 변수에 잠재된 공통인자를 찾아내는 기법
#
# 주성분분석 vs 요인분석
# 공통점 : 데이터 축소에 활용
# 차이점
# 1) 생성된 변수 수
#    요인 : 변수 무한 생성 가능
#    주성분 : 보통 1~3개 생성
# 2) 생성된 변수 이름
#    요인 : 요인의 이름 명명
#    주성분 : 제1주성분, 제2주성분 등
# 3) 생성된 변수들 간의 관계
#    요인 : 변수 간 대등한 관계 (분류/예측에 다음 단계에 사용되면 중요성 의미 부여)
#    주성분 : 제1주성분이 가장 중요
# 4) 분석 방법 의미
#    요인 : 변수들을 비슷한 성격들로 묶어 새로운 [잠재]변수 생성
#    주성분 : 목표 변수를 잘 예측/분류하기 위해 선형 결합으로 이루어진 주성분 발굴
#
# scree plot
# 보통 누적기여율 cumulative proportion 85% 이상이면 주성분의 수로 결정
#===============================================================================================================

# USArrests dataset
# 1973년 미국 50개 주 10만명 당 체포된 세 가지 강력범죄수 (assault, murder, rape)
# & 각 주마다 도시에 거주하는 인구 비율

# 변수 간 척도 차이가 크니 상관행렬 사용해 분석
# 특이치 분해를 사용하는 경우 자료 행렬의 각 변수의 평균과 제곱의 합이 1로 표준화되었다고 가정

library(datasets)
data(USArrests)

# panel.smooth : 각 그래프 안에 빨간 추세선 추가
pairs(USArrests, panel=panel.smooth, main="USArrests data")

# 결과
# Murder와 UrbanPop 비율 간 관련성이 작아 보임

US.prin <- princomp(USArrests, cor=TRUE)
summary(US.prin)
# Importance of components:
#   Comp.1    Comp.2    Comp.3     Comp.4
# Standard deviation     1.5748783 0.9948694 0.5971291 0.41644938
# Proportion of Variance 0.6200604 0.2474413 0.0891408 0.04335752
# Cumulative Proportion  0.6200604 0.8675017 0.9566425 1.00000000

# 결과
# 두 개의 주성분으로 86.8% 설명 가능

screeplot(US.prin, npcs=4, tpe="lines")

# 네 개의 변수가 각 주성분 Comp.1 ~ Comp.4에 기여하는 가중치 보기
loadings(US.prin)
# Loadings:
#           Comp.1 Comp.2 Comp.3 Comp.4
# Murder    0.536  0.418  0.341  0.649
# Assault   0.583  0.188  0.268 -0.743
# UrbanPop  0.278 -0.873  0.378  0.134
# Rape      0.543 -0.167 -0.818       
# 
#                 Comp.1 Comp.2 Comp.3 Comp.4
# SS loadings      1.00   1.00   1.00   1.00
# Proportion Var   0.25   0.25   0.25   0.25
# Cumulative Var   0.25   0.50   0.75   1.00

# 결과
# 제1주성분에는 네 개 변수가 평균적으로 기여
# 제2주성분에는 (Murder, Assault)와 (UrbanPop, Rape) 계수 부호가 다름

# 각 주성분Comp.1 ~ Comp.4의 선형식을 통해 각 지역(record) 별 결과 계산
US.prin$scores

# 제 1-2주성분에 의한 행렬도
arrests.pca <- prcomp(USArrests, center=TRUE, scale=TRUE)
biplot(arrests.pca, scale=0)

# 결과
# 조지아, 메릴랜드, 뉴멕시코 : 폭행, 살인 비율이 상대적으로 높음
# 미시간, 텍사스 : 강간 비율 높음
# 콜로라도, 캘리포니아, 뉴저지 : 도시에 거주하는 인구 비율이 높음
# 아이다호, 뉴햄프셔, 아이오와 : 도시에 거주하는 인구 비열 낮음 & 3대 강력범죄도 낮음
