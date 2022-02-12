
#===============================================================================================================
# ROCR : Receiver Operating Characteristic Curve
# 가로축 : FPR (False Positive Rate, 1-특이도)
# 세로축 : TPR (True Positive Rate, 민감도)
#===============================================================================================================

install.packages("party")
install.packages("ROCR")

library(rpart)
library(party)
library(ROCR)

# kyphosis dataset
x <- kyphosis[sample(1:nrow(kyphosis), nrow(kyphosis), replace=F),]

# train vs test
x.train <- kyphosis[1:floor(nrow(x)*0.75),]          # 전체의 75%
x.evaluate <- kyphosis[floor(nrow(x)*0.75):nrow(x),] # 남은 25%

x.model <- cforest(Kyphosis~Age+Number+Start, data=x.train)

x.evaluate$prediction <- predict(x.model, newdata=x.evaluate)
x.evaluate$correct <- x.evaluate$prediction == x.evaluate$Kyphosis

print(paste("% of predicted classification correct", mean(x.evaluate$correct)))
# "% of predicted classification correct 0.818181818181818"

x.evaluate$probabilities <- 1 - unlist(treeresponse(x.model, newdata=x.evaluate), use.names=F)[seq(1,nrow(x.evaluate)*2,2)]

pred <- prediction(x.evaluate$probabilities, x.evaluate$Kyphosis)
perf <- performance(pred, "tpr", "fpr")
plot(pref, main="ROC curve", colorize=T)

pref <- performance(pred, "lift", "rpp")
plot(pref, main="LIFT curve", colorize=T)
