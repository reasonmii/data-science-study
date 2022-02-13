
#===============================================================================================================
# Random Forest
#===============================================================================================================

install.packages("randomForest")
library(randomForest)

idx <- sample(2, nrow(iris), replace=TRUE, prob=c(0.7,0.3))
train.data <- iris[idx==2,]
test.data <- iris[idx==1,]
rf <- randomForest(Species~., data=train.data, ntree=100, proximity=TRUE)

# Evaluate : Train
table(predict(rf), train.data$Species)
#             setosa versicolor virginica
# setosa         14          0         0
# versicolor      0         12         1
# virginica       0          1        16

# Plot
plot(rf)

# Plot : variance importance
varImpPlot(rf)

# Evaluate : Test
pre.rf <- predict(rf, newdata=test.data)
table(pre.rf, test.data$Species)
# pre.rf       setosa versicolor virginica
# setosa         36          0         0
# versicolor      0         33         3
# virginica       0          4        30

# Plot
plot(margin(rf, test.data$Species))

