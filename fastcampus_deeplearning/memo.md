
## Deep Learning이 201X년에야 뜬 이유
- Gradient vanishing 문제 (DNN 발전 느림) -> non-saturating activation function으로 해결된 것이 상대적으로 최근
- 학습이 어려움 : Non-convex loss(비볼록 손실함수)를 포함하고 있어 Optimal Solution(최적해)를 찾는 것이 보장되지 않음
  - 새로 도입된 optimization 방법이 local minima에 빠지는 것 방지
  - 그러나 많은 critical points(임계점)이 local minima인 줄 알고 있던 많은 문제가 사실 saddle point이기도 했음
  - 기존 접근방식인 convex learning은 정답과 어느정도 거리가 있고 사람의 학습과정 역시 convex 방법으로 설명이 불가

## MLOps
- machine learning, deep learning의 상업적 성공
- 첫 운영 레벨의 정의
- machine learning model을 production으로 전환하는 process 간소화, 유지/관리/모니터링에 focus
- 협업 tool for data scientist, DevOps engineer, IT
- 필요한 이유
  - 머신 러닝 수명 주기는 데이터 수집, 데이터 준비, 모델 훈련, 모델 조정, 모델 배포, 모델 모니터링, 설명 가능성과 같은 복잡한 구성 요소가 많이 모인 형태로 구성
  - 이 모든 프로세스를 동기화하고 협력이 이루어지는 상태를 유지하려면 극히 엄격한 운영 원칙을 적용해야 함
  - MLOps는 머신 러닝 수명 주기의 실험, 반복과 지속적 개선을 총망라
- 구성
  - EDA (Exploratory Data Analysis)
  - Data prep & Feature engineering
  - Model training & tuning
  - Model review & governance
  - Model inference & serving
  - Model deployment & monitoring
  - Automated model retraining
- 장점
  - 효율성: MLOps를 사용하면 데이터 팀이 모델을 더욱 빨리 배포하고 양질의 ML 모델을 제공하며 배포와 프로덕션 속도 up
  - 확장성: MLOps는 엄청난 확장성과 관리를 지원하므로 수천 개의 모델을 감독, 제어, 관리, 모니터링하여 지속해서 통합, 제공하고 지속해서 배포 가능
    - MLOps는 ML 파이프라인 재현성을 제공하므로 여러 데이터 팀에서 좀 더 긴밀하게 결합된 협업 추진 가능
    - DevOps 팀과 IT 팀의 갈등이 줄어들며 release 속도 up
  - 리스크 완화: 머신 러닝 모델에는 철저한 규제 검토와 드리프트 검사가 필요할 때가 많은데
    - 투명성 강화, 요청에 빠른 대응, 주어진 기업이나 업계의 규정을 더욱 엄격히 준수하는 데 도움

MLOps vs DevOps
- machine learning projet에 국한된 엔지니어링 실무
- 소프트웨어 엔지니어링 분야에서 광범위하게 도입된 DevOps 원칙을 빌려온 것
  - DevOps는 애플리케이션 전달에 지속해서 반복적이면서 속도도 빠른 접근 방식을 도입
  - MLOps의 경우 머신 러닝 모델의 프로덕션 돌입까지의 과정에 같은 원칙을 적용
 - 두 경우 모두 소프트웨어 품질 개선, 패치 적용과 릴리스 속도 가속, 높은 고객 만족도 달성과 같은 결과를 낸다는 점은 같습니다.

## Pytorch vs TensorFlow vs Jax
- 연구에서는 2020년 이후 pytorch가 tensorflow 역전
- tensorflow가 편해졌지만 v1과 혼재되어 잘못 쓰기 쉽고 인터넷에 v1, 2 자료가 혼재
- pytorch가 research engineering에서는 부족하지만 많이 따라옴
- 최근 MLOps와 함께 model exchange 및 model serving방법이 많이 연구됨에 따라 framework-agnostic AI model serving이 가능해지고 어떤 frame work를 쓰든 상관없는 분위기
- 결론 : tensorflow v2와 pytorch 중 편한 것 쓰기

||Pytorch|TensorFlow|Jax|
|---|---|---|---|
|개발, 운영|Facebook|Google|Google|
|유연성|O|부분적|O|
|그래프 생성|Dynamic|Static/Dynamic|Static|
|target|researcher, developer|researcher, developer|researcher|
|Low/High-level API|Both|Both(high-level에 특화)|Both(low-level에 특화)|
|Learning curve|적당|v1:높음, v2:적당|약간 있음|
|제품화를 위한 engineering|좋아짐 (불가해도 추후 TF 등 변경 가능)|좋았고 더 좋아질 예정|X|
|Multi-GPU Training|편하지만 잘 쓰려면 까다로움 (Pytorch-lightning)|v2로 넘어오면서 편해짐|지원|
|TPU|지원|지원|지원|
|Single Thread 속도|빠름|빠름|빠름|



