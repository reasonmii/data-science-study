
#=====================================
# 결측값 처리
#
# 결측값 있으면 FALSE, 없으면 TRUE 반환
# complete.cases()
#
# 결측값을 NA로 인식
# 결측값 있으면 TRUE, 없으면 FALSE 반환
# is.na()
#
# NA값을 가운데 값(central value)으로 대치
# 숫자는 중위수, 요인(factor)은 최빈값
# DMwR 패키지
# centralImputation()
#
# NA값을 k최근이웃분류 알고리즘으로 대치
# k개 주변 이웃까지의 거리 고려하여 가중 평균 사용
# DMwR 패키지
# knnImputation
#
# time-series-cross-sectional data set(여러 국가에서 매년 측정된 자료)에서 활용
# randomForest 모델은 결측값 존재 시 에러 발생
# randomForest의 rfImpute() 사용하여 NA값 대치 후 알고리즘 사용
# Amelia 패키지
# amelia()
#=====================================
