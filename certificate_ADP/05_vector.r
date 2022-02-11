
#=====================================
# 벡터
#
# 기본연산 : + - * /
#
# 벡터 요소 추출 $ : a$coef
#
# 함수적용
# sapply(변수, 연산함수) : sapply(a,log)
#=====================================

# 벡터생성
# 원소 중 하나라도 문자면 전체 문자형으로 인식
c("a", 1, "bc")
v <- c(1,2,3,4)

# data 추가
newItems <- 5
v <- c(v, newItems)         # 1 2 3 4 5
v[length(v)+1] <- newItems  # 1 2 3 4 5

# 요인 생성
f <- factor(v)

# 합치기
v1 <- c("a","b")
v2 <- c("apple", "banana")
v3 <- c("carrot")
comb <- stack(list(v1=v1, v2=v2, v3=v3))
# values ind
# 1      a  v1
# 2      b  v1
# 3  apple  v2
# 4 banana  v2
# 5 carrot  v3

# 값 조회
v <- c(10,20,30,40,50)
v[c(1,3,5)]   # 10 30 50
v[-c(2,4)]    # 10 30 50


