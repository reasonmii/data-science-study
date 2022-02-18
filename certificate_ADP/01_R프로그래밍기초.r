
#=====================================
# R 기초
# comment : ctrl + shift + C
#
# 배치모드
# 1) 사용자와 인터렉션이 필요하지 않는 방식으로 매일 돌아가야 하는 시스템에서 프로세스를 자동화 할 때 유용
# 2) Path 지정
#    '내컴퓨터' 우클릭 -> 속성 -> 고급시스템 설정 -> 환경변수
#    -> 변수명 path -> R프로그램의 실행파일 위치 찾아서 추가 -> 저장
# 3) 배치파일 실행
#    윈도우 창 batch.R 실행파일이 있는 위치에서 "R CMD BATCH batch.R" 실행
#=====================================


#=====================================
# working directory 지정 : setwd("~\")
# package 설치 : install.packages("패키지명")
# package 불러오기 : library("패키지명")
# data 불러오기 : data(dataset), data(dataset, package="")
# data 요약 : summary(dataset)
# data 조회 : head(dataset)
# 작업종료 : q()
#=====================================


#=====================================
# 출력 (형식지정X)
# 하나 객체만 : print(값)
# 항목 연결 : cat(값1, 값2, 값3...)
#
# 변수목록 : ls(), ls.str()
# 변수삭제 : rm(변수)
# 전체 변수삭제 : rm(list=ls())
#
# 도움말 : ? ex) ?lm
#
# 값 할당 : <-, =, ->
#=====================================

# 반복
rep(1,time=5)    # 1 1 1 1 1
rep(1:4, each=2) # 1 1 2 2 3 3 4 4
rep(x, each=2)   # 1 1 2 2 3 3 4 4

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
# 주요코드
#=====================================

# 요인으로 집단 정의
# c() : 벡터생성
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


