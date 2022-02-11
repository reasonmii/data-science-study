
#=====================================
# data.table
# 가장 많이 사용하는 data handling package
# 큰 data를 탐색, 연산, 병합할 때 유용
# data.frame보다 월등히 빠른 속도 (빠른 grouping, 짧은 문장 지원)
#=====================================

install.packages("data.table")
library(data.table)

df1 <- data.frame(x=runif(2.6e+07), y=rep(LETTERS, each=10000))
df2 <- data.frame(x=runif(2.6e+07), y=rep(letters, each=10000))

system.time(x <- df1[df1$y == "C",])
# 사용자  시스템 elapsed 
# 0.19    0.01    0.22

dt <- as.data.table(df1)
setkey(dt, y)
system.time(x <- dt[J("C"),])
# 사용자  시스템 elapsed 
# 0.03    0.00    0.03 
