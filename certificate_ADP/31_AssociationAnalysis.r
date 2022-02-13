
#===============================================================================================================
# 연관규칙분석 Association Analysis
# 1) 장바구니분석 : 장바구니에 무엇이 같이 들었을까
# 2) 서열분석 : A를 산 다음에 B를 산다
#
# 지지도 support : A와 B가 동시에 포함된 거래 수 / 전체 거래 수
# 신뢰도 confidence : A와 B가 동시에 포함된 거래 수 / A를 포함하는 거래 수
#
# 향상도 Lift
# A가 구매되지 않았을 때 품목 B의 구매확률에 비해 A가 구매됐을 때 품목 B의 구매확률의 증가 비
# A와 B가 동시에 포함된 거래 수 / (A를 포함하는 거래수 * B를 포함하는 거래 수)
#===============================================================================================================

# arules package - Groceries dataset
# 식료품 판매점의 1달 동안 POS 데이터
# 총 169개 제품과 9835건의 거래건수 포함

install.packages("arules")
library(arules)

data(Groceries)

# inspect() : 거래내역 확인 가능
inspect(Groceries[1:3])

rules <- apriori(Groceries, parameter=list(support=0.01, confidence=0.3))
# Apriori
# 
# Parameter specification:
#   confidence minval smax arem  aval originalSupport maxtime support minlen maxlen target  ext
# 0.3    0.1    1 none FALSE            TRUE       5    0.01      1     10  rules TRUE
# 
# Algorithmic control:
#   filter tree heap memopt load sort verbose
# 0.1 TRUE TRUE  FALSE TRUE    2    TRUE
# 
# Absolute minimum support count: 98 
# 
# set item appearances ...[0 item(s)] done [0.00s].
# set transactions ...[169 item(s), 9835 transaction(s)] done [0.00s].
# sorting and recoding items ... [88 item(s)] done [0.00s].
# creating transaction tree ... done [0.00s].
# checking subsets of size 1 2 3 4 done [0.00s].
# writing ... [125 rule(s)] done [0.00s].
# creating S4 object  ... done [0.00s].

# 해석
# 총 88개의 아이템으로 연관규칙 생성
# 125개의 rule 발견
# - 규칙 수가 너무 적으면 지지도와 신뢰도 낮추기
# - 규칙 수가 너무 많으면 지지도와 신뢰도 높이기

inspect(sort(rules,by=c("lift"), decresing=TRUE)[1:20])
#      lhs                                       rhs                support    confidence coverage   lift     count
# [1]  {citrus fruit, other vegetables}       => {root vegetables}  0.01037112 0.3591549  0.02887646 3.295045 102  
# [2]  {tropical fruit, other vegetables}     => {root vegetables}  0.01230300 0.3427762  0.03589222 3.144780 121  
# [3]  {beef}                                 => {root vegetables}  0.01738688 0.3313953  0.05246568 3.040367 171  
# [4]  {citrus fruit, root vegetables}        => {other vegetables} 0.01037112 0.5862069  0.01769192 3.029608 102  
# [5]  {tropical fruit, root vegetables}      => {other vegetables} 0.01230300 0.5845411  0.02104728 3.020999 121  
# [6]  {other vegetables, whole milk}         => {root vegetables}  0.02318251 0.3097826  0.07483477 2.842082 228  
# [7]  {whole milk, curd}                     => {yogurt}           0.01006609 0.3852140  0.02613116 2.761356  99  
# [8]  {root vegetables, rolls/buns}          => {other vegetables} 0.01220132 0.5020921  0.02430097 2.594890 120  
# [9]  {root vegetables, yogurt}              => {other vegetables} 0.01291307 0.5000000  0.02582613 2.584078 127  
# [10] {tropical fruit, whole milk}           => {yogurt}           0.01514997 0.3581731  0.04229792 2.567516 149  
# [11] {yogurt, whipped/sour cream}           => {other vegetables} 0.01016777 0.4901961  0.02074225 2.533410 100  
# [12] {other vegetables, whipped/sour cream} => {yogurt}           0.01016777 0.3521127  0.02887646 2.524073 100  
# [13] {tropical fruit, other vegetables}     => {yogurt}           0.01230300 0.3427762  0.03589222 2.457146 121  
# [14] {root vegetables, whole milk}          => {other vegetables} 0.02318251 0.4740125  0.04890696 2.449770 228  
# [15] {whole milk, whipped/sour cream}       => {yogurt}           0.01087951 0.3375394  0.03223183 2.419607 107  
# [16] {citrus fruit, whole milk}             => {yogurt}           0.01026945 0.3366667  0.03050330 2.413350 101  
# [17] {onions}                               => {other vegetables} 0.01423488 0.4590164  0.03101169 2.372268 140  
# [18] {pork, whole milk}                     => {other vegetables} 0.01016777 0.4587156  0.02216573 2.370714 100  
# [19] {whole milk, whipped/sour cream}       => {other vegetables} 0.01464159 0.4542587  0.03223183 2.347679 144  
# [20] {curd}                                 => {yogurt}           0.01728521 0.3244275  0.05327911 2.325615 170  

# 해석
# 향상도 기준 내림차순 정렬, 상위 5개 규칙 확인
# rhs 제품만 구매할 확률에 비해 lhs 제품을 샀을 때 rhs 제품도 구매할 확률이 3배 가량 높음 (Lift > 3이기 때문)
# 따라서 rhs와 lhs 제품 간 결합상품 할인쿠폰 or 품목배치 변경 제안 가능



