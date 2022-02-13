
#===============================================================================================================
# 의사결정나무
#===============================================================================================================

# party package
# 의사결정나무를 사용하기 편한 다양한 분류 패키지 중 하나
# 단점 : 분실값을 잘 처리하지 못함
#        tree에 투입된 데이터가 표시되지 않거나 predict가 실패하는 경우 문제 발생

install.packages("party")
library(party)

# train : 70%, test : 30%
idx <- sample(2, nrow(iris), replace=TRUE, prob=c(0.7,0.3))  # 1, 2 대입
train.data <- iris[idx==2,]
test.data <- iris[idx==1,]

# modeling
iris.tree <- ctree(Species~., data=train.data)

# Plot
plot(iris.tree)
plot(iris.tree, type="simple")

# Evaluate : Train
table(predict(iris.tree), train.data$Species)
#             setosa versicolor virginica
# setosa         20          0         0
# versicolor      0         14         1
# virginica       0          0        13

# Evaluate : Test
test.pre <- predict(iris.tree, newdata=test.data)
table(test.pre, test.data$Species)
# test.pre     setosa versicolor virginica
# setosa         24          0         0
# versicolor      6         34         3
# virginica       0          2        33

