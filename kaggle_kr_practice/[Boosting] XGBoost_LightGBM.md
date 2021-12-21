LightGBM, XGBoost 모두 부스팅 계열 알고리즘

XGBoost
- GBM 보다 학습속도가 빠르지만, GridSearch로 hyper parameter를 튜닝하기에는 매우 오랜 시간이 걸림

LightGBM
- XGBoost와 큰 예측 성능 차이를 보이지 않으면서, 학습 시간을 상당히 단축시킨 모델
- 단점 : 공식 문서에 따르면 일반적으로 10,000 건 이하의 데이터 세트를 다루는 경우 과적합 문제가 발생하기 쉽다고 함

