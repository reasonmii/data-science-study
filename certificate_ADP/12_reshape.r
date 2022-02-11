#=====================================
# reshape 핵심함수 2개
# 1) melt() : 원데이터 형태 생성
#    쉬운 casting을 위해 적당한 형태로 만들어주는 함수
# 2) cast() : 요약 형태 생성
#    데이터를 원하는 형태로 계산 또는 변형 시켜주는 함수
#=====================================

install.packages("reshape")
library(reshape)

# 변수 : Ozone, Solar.R, Wind, Temp, Month, Day
head(airquality)

# 변수 : Month, Day, variable(Ozone, Solar.R, Wind, Temp), value
aqm <- melt(airquality, id=c("Month","Day"), na.rm=T)

# 4개의 테이블 생성 : Ozone, Solar.R, Wind, Temp
# 행 : Day / 열 : Month / 값 : variable
cast(aqm, Day ~ Month ~ variable)

# id, variable에 대해 Time의 value를 확인할 때
# 열1 : id / 열2 : variable / 값 : time
cast(md, id + variable ~ time)
