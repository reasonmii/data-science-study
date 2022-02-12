

#=====================================
# 행렬
# 행렬의 연산 : + -
#=====================================

# matrix(data,행수,열수)
# a <- matrix(data,2,3)

# data 대신 숫자 입력 시 행렬의 값이 동일한 수치값 부여
a <- matrix(0,4,5)
# [,1] [,2] [,3] [,4] [,5]
# [1,]    0    0    0    0    0
# [2,]    0    0    0    0    0
# [3,]    0    0    0    0    0
# [4,]    0    0    0    0    0

a <- matrix(1:20,4,5)
# [,1] [,2] [,3] [,4] [,5]
# [1,]    1    5    9   13   17
# [2,]    2    6   10   14   18
# [3,]    3    7   11   15   19
# [4,]    4    8   12   16   20

# 차원
dim(a)   # 4 5

# 대각
diag(a)  # 1 6 11 16

# 전치(transpose)
t(a)
# [,1] [,2] [,3] [,4]
# [1,]    1    2    3    4
# [2,]    5    6    7    8
# [3,]    9   10   11   12
# [4,]   13   14   15   16
# [5,]   17   18   19   20

# 역 (정방행렬만 가능)
solve(a)

# 행렬의 곱
a * 3
a %*% t(a)

# 행, 열 이름
rownames(a) <- c("1번","2번","3번","4번")
colnames(a) <- c("가","나","다","라","마")
a
# 가 나 다 라 마
# 1번  1  5  9 13 17
# 2번  2  6 10 14 18
# 3번  3  7 11 15 19
# 4번  4  8 12 16 20

# 행 선택
a[1,]
# 가 나 다 라 마 
# 1  5  9 13 17 

# 열 선택
a[,3]
# 1번 2번 3번 4번 
# 9  10  11  12 

