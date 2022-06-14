
## Deep Learning이 201X년이 되어서야 뜬 이유
- Gradient vanishing 문제가 DNN 발전을 느리게 했는데 non-saturating activation function으로 해결된 것이 상대적으로 최근
- 학습이 어려움 : Non-convex loss(비볼록 손실함수)를 포함하고 있어 Optimal Solution(최적해)를 찾는 것이 보장되지 않음
  - 새로 도입된 optimization 방법이 local minima에 빠지는 것 방지
  - 그러나 많은 critical points(임계점)이 local minima인 줄 알고 있던 많은 문제가 사실 saddle point이기도 했음
  - 기존 접근방식인 convex learning은 정답과 어느정도 거리가 있고 사람의 학습과정 역시 convex 방법으로 설명이 불가

## MLOps
- machine learning, deep learning의 상업적 성공
- 첫 운영 레벨의 정의

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
|Learning curve|적당|v:높음, v2:적당|약간 있음|
|제품화를 위한 engineering|좋아짐, 불가해도 추후 TF 등 변경 가능|좋았고 더 좋아질 예정|X|
|Multi-GPU Training|편하지만, 잘 쓰려하는 순간 까다로움(Pytorch-lightning)|v2로 넘어오면서 편해짐|지원|
|TPU|지원|지원|지원|
|Single Thread 속도|빠름|빠름|빠름|



