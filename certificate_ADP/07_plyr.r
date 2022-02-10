
#=====================================
# plyr
# apply 함수 기반
# 데이터와 출력변수를 동시에 배열로 치환하여 처리
# split-apply-combine
#=====================================

install.packages("plyr")
library(plyr)

head(iris)

test <- ddply(iris, "Species", function(x) {
  m.value <- mean(x$Sepal.Length)
  sd.value <- sd(x$Sepal.Length)
  cv <- round(sd.value/m.value,4)
  data.frame(cv.value=cv)
})

test
#      Species cv.value
# 1     setosa   0.0704
# 2 versicolor   0.0870
# 3  virginica   0.0965
