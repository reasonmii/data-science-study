### Pandas에서 제공하는 기술통계량 주요 함수

사용법) 데이터셋.함수명()
<br/>

count : NA 값을 제외한 값의 수 반환
describe : 시리즈 혹은 데이터프레임의 각 열에 대한 기술 통계
<br/><br/>

min, max : 최소, 최대값
argmin, argmax : 최소, 최대값을 가지고 있는 색인 위치 반환
idxmin, idxmax : 최소, 최대값을 가지고 있는 색인의 값 반환
<br/><br/>

quantile : 0~1 분위수 계산
<br/><br/>

sum : 합
mean : 평균
median : 중위값
mad : 평균값에서 절대 평균편차
<br/><br/>

var : 표본 분산
std : 표본 정규분산
skew : 표본 비대칭도
kurt : 표본 첨도
<br/><br/>

cumsum : 누적 합
cummin, cummax : 누적 최소값, 누적 최대값
cumprod : 누적 곱
<br/><br/>

diff : 1차 산술차(시계열 데이터 사용 시 유용)
<br/><br/>

pct_change : 퍼센트 변화율 계산
<br/><br/>

corr : 데이터프레임의 모든 변수 간 상관관계 계산하여 반환
cov : 데이터프레임의 모든 변수 간 공분산을 계산하여 반환

