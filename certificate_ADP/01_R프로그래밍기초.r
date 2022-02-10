
#=====================================
# working directory 지정 : setwd("~\")
# package 설치 : install.packages("패키지명")
# package 불러오기 : library("패키지명")
# data 불러오기 : data(dataset)
# data 요약 : summary(dataset)
# data 조회 : head(dataset)
# 작업종료 : q()
#=====================================


#=====================================
# R 기초
#
# comment : ctrl + shift + C
#
# 출력 (형식지정X) : print(값)
#
# 변수목록 : ls(), ls.str()
# 변수삭제 : rm(변수)
# 전체 변수삭제 : rm(list=ls())
#
# 도움말 : ? ex) ?lm
#
# 값 할당 : <-, =, ->
#
# 항목 연결 : cat(값1, 값2, 값3...)
#
# 사칙연산 : + - * / 
#
# 특수연산
# %/% : 나눗셈 몫
# %% : 나눗셈 나머지
# %*% : 행렬의 곱
#
# 비교 : ==, != <, <=, >, >=
# 논리부정 : ! ex) !(3==5)
# 논리 and, or : & |
# 
# 기초통계
# 평균 mean(변수)
# 합계 sum(변수)
# 중앙값 median(변수)
# 로그 log(변수)
# 표준편차 sd(변수)
# 분산 var(변수)
# 공분산 cov(변수1, 변수2)
# 상관계수 cor(변수1, 변수2)
# 변수길이 length(변수)
#
# 식 formula : ~ ex) lm(log(brain)~log(body),data=Animals)
#=====================================

# 수열
1:5                               # 1 2 3 4 5
9:-2                              # 9  8  7  6  5  4  3  2  1  0 -1 -2
seq(from=0, to=20, by=2)          # 0  2  4  6  8 10 12 14 16 18 20
seq(from=0, to=20, length.out=5)  # 0  5 10 15 20

# 지수
5^2     # 25

# 반복
rep(1,time=5)    # 1 1 1 1 1
rep(1:4, each=2) # 1 1 2 2 3 3 4 4
rep(x, each=2)   # 1 1 2 2 3 3 4 4

# 문자 붙이기
A <- paste("a","b","c",sep="-")  # "a-b-c"
paste(A,c("e","f"))              # "a-b-c e" "a-b-c f"
paste(A,10,sep="")               # "a-b-c10"

# 문자추출
substr("Bigdataanalysis",1,4)  # Bigd

# 논리값
a <- True
a <- T
b <- False
b <- F

# 인덱스
a <- c(1, 2, 3, 4, 5)
a[1]    # 1
a[-1]   # 2 3 4 5

# R 함수정의
# function(매개변수1, 매개변수2, ..)
# 조건문 if
# 반복문 for, while, repeat
# 함수 내에 값 할당했지만, 지역변수가 아닌 전역변수로 하고 싶을 때 : <<- 사용
b <<- 45

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



#=====================================
# 리스트
# list(숫자, 문자, 함수)
#
# 이름으로 원소선택
# a[["name"]]
# a$name
#
# 리스트에서 원소 제거
# a[["name"]] <- NULL
#
# NULL 원소를 리스트에서 제거
# a[sapply(a, is.null)] <- NULL
# a[a==o] <- NULL
# a[is.na(a)] <- NULL
#=====================================

a <- list(v1,v2,v3)
# [[1]]
# [1] "a" "b"
# [[2]]
# [1] "apple"  "banana"
# [[3]]
# [1] "carrot"

a[[1]]   # "a" "b"

a[c(1,3)]
# [[1]]
# [1] "a" "b"
# [[2]]
# [1] "carrot"



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



#=====================================
# dataframe
#
# 벡터들로 dataset 생성
# data.frame(벡터,벡터,벡터)
#
# 열 data로 data 프레임 생성
# df <- data.frame(v1,v2,v3,f1,f2)
# df <- as.data.frame(list.of.vectors)
#
# dataframe 할당
# N = 1,000,000
# dt <- data.frame(dosage=numeric(N),
#    lab=character(N),
#    respones=numeric(N))
#
# dataframe 조회
# df[df$gender="m"]
# df[df$변수1 > 4 & df$변수2 > 5, c(변수3, 변수4)]
# df[grep("문자", df$변수1, ignore.case=T), c("변수2","변수3")]
#   -> dataset 변수1 내 "문자"가 들어간 케이스들의 변수2, 변수3 값 조회
#
# dataset 조회
# subset(df, select=변수, subset=변수>조건)
#
# data 선택
# lst[[2]], lst[2], lst[2,], lst[,2]
# lst[["name"]], lst$name
# lst[c("name1","name2","name3")]
#
# data 병합
# merge(df1, df2, by="공통 열이름")
# merge(df1, df2, by="열이름", all=T)
#
# 열이름 조회
# colnames(변수)
#
# 행/열 선택
# subset(df, select=열이름)
# subset(df, select=c(열1,열2,열3))
# subset(df, select=열이름, subset=(조건))
#
# 이름으로 열 제거
# subset(df, select=-"열이름")
#
# 열 이름 변경
# colnames(df) <- newnames
#
# NA 포함 행 삭제
# a <- na.omit(df)
#=====================================

a <- data.frame(a=1,b=2,c=3,d='a')
#   a b c d
# 1 1 2 3 a

b <- data.frame(a=4,b=5,c=6,d='b')

# dataset 행결합
rbind(a,b)
#   a b c d
# 1 1 2 3 a
# 2 4 5 6 b

# dataset 열결합
cbind(a,b)
#   a b c d a b c d
# 1 1 2 3 a 4 5 6 b



#=====================================
# 자료형 data 구조 변환
#
# data 프레임 내용 접근
# with(df, expr)
# attach(df)
# detach(df)
#
# 자료형 변환
# as.character()
# as.complex()
# as.numeric()
# as.double()
# as.integer()
# as.logical()
#
# data 구조 변환
# 벡터 -> 리스트 : as.list(vec)
# 벡터 -> 행렬
#   1열 행렬 : cbind(vec), as.matrix(vec)
#   1행 행렬 : rbind(vec)
#   n*m 행렬 : matrix(vec,n,m)
# 벡터 -> data프레임
#   1열 df : as.data.frame(vec)
#   1행 df : as.data.frame(rbind(vec))
# 리스트 -> 벡터 : unlist(list)
# 리스트 -> 행렬
#   1열 행렬 : as.matrix(lst)
#   1행 행렬 : as.matrix(rbind(lst))
#   n*m 행렬 : matrix(lst,n,m)
# 리스트 -> data프레임
#   목록 원소들이 data의 열인 경우 : as.data.frame(lst)
#   리스트 원소들이 data 행인 경우 : rbind(obs[[1]],obs[[2]])
# 행렬 -> 벡터 : as.vector(mat)
# 행렬 -> 리스트 : as.list(mat)
# 행렬 -> data프레임 : as.dataframe(mat)
# data프레임 -> 벡터
#   1열 df : df[[1]], df[,1]
#   1행 df : df[1,]
# data프레임 -> 리스트 : as.list(df)
# data프레임 -> 행렬 : as.matrix(df)
#=====================================



#=====================================
# 파일저장
# write.csv(변수이름, "파일이름.csv")
# save(변수이름, file="파일이름.Rdata")
#
# 파일읽기
# read.csv("파일이름.csv")
#
# 파일 불러오기
# load("파일.R")
# source("파일.R")
#=====================================



#=====================================
# 주요코드
#=====================================

# 요인으로 집단 정의
v1 <- c(1,2,3,4)
v2 <- c(56,80,41,67)
f <- factor(c("A","B","C","D"))

# 벡터를 여러 집단으로 분할
groups <- split(v1, f)
groups <- split(v2, f)

groups <- unstack(data.frame(v1,f))
# res
# A   1
# B   2
# C   3
# D   4

# dataframe을 여러 집단으로 분할
library(MASS)
sp <- split(Cars93$MPG.city, Cars93$Origin)
median(sp[[1]])


#=====================================
# 함수 적용
#
# 리스트의 각 원소에 함수 적용
# lapply(결과를 리스트 형태로 반환) : list <- lapply(l, func)
# sapply(결과를 벡터 또는 행렬로 반환) : vec <- sapply(l, func)
#
# 행렬에 함수 적용
# m <- apply(m1, 1, func)
#
# dataframe에 함수 적용
# df <- lapply(df, func)
# df <- sapply(df, func)
# df <- apply(df, func)
# ※ appply : dataframe이 동질적인 경우만(모두 숫자, 문자) 가능
# dataframe을 행렬로 변환 후 함수 적용
#
# 대용량 data의 함수 적용 : 다중회귀분석
# sapply를 통한 간단한 R coding
# 1) 타겟변수와 상관계수 구하기
#    cors <- sapply(df, cor, y=target)
# 2) 상관계수가 높은 상위 10개의 변수를 입력변수로 선정
#    mask <- (rank(-abs(cors))<=10)
#    best.pred <- df[,mask]
# 3) 타겟변수와 입력변수로 다중회귀분석 실시
#    lm(target~bes.pred)
#
# 집단별 함수 적용
# tapply(vec, factor, func)
#
# 행집단 함수 적용
# by(df, factor, func)
#
# 병렬 벡터, 리스트 함수 적용
# mapply(factor, vec1, vec2, ...)
# mapply(vector, list1, list2, ...)
#=====================================



