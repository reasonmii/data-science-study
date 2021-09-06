
# the library to draw ROC graphs
install.packages("pROC")
library(pROC)

# To classify samples
library(randomForest)

set.seed(420)

num.samples <- 100

# rnorm() : generate 100 random values from a normal distribution
#           with the mean set to 172 and the standard deviation set to 29
weight <- sort(rnorm(n=num.samples, mean=172, sd=29))

# classify an individual as obese and not obese
# rank() : rank the weights from lightest (1) to heaviest (100)
# /100   : scale the ranks by 100 (1 ~ 100 -> 0.01 ~ 1)
#
# runif(n=num.samples) < (rank(weight)/100)
# Compare the scaled ranks to random numbers between 0 and 1
#
# If the random number is smaller than the scaled rank, the individual is classified as obese
# Otherwise it is classified as not obese
obese <- ifelse(test=(runif(n=num.samples) < (rank(weight)/100)), yes=1, no=0)
obese

plot(x=weight, y=obese)

# Fit a logistic regression curve to the data
glm.fit = glm(obese ~ weight, family="binomial")

# Draw a curve that tells us the predicted probability
#
# glm.fit$fitted.values
# It contains the y-axis coordinates along the curve for each sample
# It contains estimated probabilities that each sample is obese
# We will use the known classifications and the estimated probabilities to draw an ROC curve
lines(weight, glm.fit$fitted.values)

# roc() from the pROCC library to draw the ROC graph
# obese : Pass in the known classifications, obese or not obese, for each sample
# glm.fit$fitted.values : The estimated probabilities that each sample is obese
# plot=TRUE : Draw the graph, not just calculate all of the numbers used to draw the graph 
roc(obese, glm.fit$fitted.values, plot=TRUE)

# Result
# The diagonal line shows where the True Positive Rate is the same as the False Positive Rate
#
# When you use the roc() function, it prints out a bunch of stuff
# 
# Data: glm.fit$fitted.values in 45 controls (obese 0) < 55 cases (obese 1).
# It tells us how many samples were not obese (=0) and obese (=1)
#
# Area under the curve: 0.8291

# When you use RStudio, you'll also see the padding on each side on the graph
# To get rid of the ugly padding, we have to use the par() function
# and much around with the graphics parameters
# pty :the plot type
# s : square
par(pty="s")

# Plot the graph again
# We get a much nicer ROC graph
roc(obese, glm.fit$fitted.values, plot=TRUE)

# By default, the roc() function plots Specificity on the x-axis instead of 1 - Specificity
# As a result, the x-axis goes from 1, on the left side, to 0 on the right side
# set legacy.axes to TRUE to change this
roc(obese, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

# Set percent to TRUE
# so that the axes are in percentages, rather than values between 0 and 1
#
# col="$377eb8", lwd=4
# Change the color of the ROC curve and make it thicker
roc(obese, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE,
    xlab="False Positive Percentage",
    ylab="True Positive Percentage",
    col="#377eb8", lwd=4)

# Now, imagine we're interested in the range of thresholds that resulted in the part of the ROC curve
# We can access those thresholds by saving the calculations that the roc() function does in a variable
roc.info <- roc(obese, glm.fit$fitted.values, legacy.axes=TRUE)

# Make a data.frame that contains all of the True Positive Percentages,
# by multiplying the Sensitivities by 100
# and the False Positive Percentages, by multiplying 1 - Specificities by 100
roc.df <- data.frame(
  tpp=roc.info$sensitivities*100,
  fpp=(1-roc.info$specificities)*100,
  thresholds=roc.info$thresholds
)

# See that when the threshold is set to negative infinity,
# so that every single sample is called obese
# then the TPP, the True Positive Percentage, is 100
# because all of these obese samples were correctly classified
# and the FPP, the False Positive Percentage, is also 100
# because all of the samples taht were not obese were incorrectly classified
# So the first row in roc.df corresponds to the upper right-hand corner of the ROC graph
head(roc.df)

# We see when the threshold is set to positivie infinity,
# so that every single sample is classified not obese
# then the TPP and FPP are both 0
# because none of samples were classified, either correctly or incorrectly, obese
# So the last row in roc.df corresponds to the bottom left-hand corner of the ROC curve
tail(roc.df)

# Now we can isolate the TPP, the FPP and the thresholds
# used when the True Positive Rate is between 60 and 80
# If we were interested in choosing a threshold in this range,
# we would pick the one that had the optmial balance of True Positives and False Positives
roc.df[roc.df$tpp > 60 & roc.df$tpp < 80, ]

# Now let's go back to talking about customizing what the roc() function draws
# If we want to print the AUC directly on the graph
# Then set the print.auc parameter to TRUE
roc(obese, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE,
    xlab="False Positive Percentage",
    ylab="True Positive Percentage",
    col="#377eb8", lwd=4,
    print.auc=TRUE)


# ========================================
# Draw and calculate
# a partial Area Under the Curve
# ========================================

# These are useful when you want to focus on the part of the ROC curve
# that only allows for a small number of False Positives
#
# print.auc.x=45
# Specify where along the x-axis you want the AUC to be printed,
# otherwise the text might overlap something important
# ※ Try a bunch of different locations and find the best one
# 
# partial.auc
# The range of Specificity values taht we want to focus on (x-axis)
# NOTE : The range of value is in terms of Specificity, not 1-Specificity
# So 100% Specificity corresponds to 0% on our 1-Specificity axis
#
# auc.polygon=TRUE
# Draw the partial Area Under the Curve
#
# #377eb822
# Added 22 to the end to make the color semi-parent
roc(obese, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE,
    xlab="False Positive Percentage",
    ylab="True Positive Percentage",
    col="#377eb8", lwd=4,
    print.auc=TRUE,
    print.auc.x=45, partial.auc=c(100, 90), auc.polygon=TRUE, auc.polygon.col="#377eb822")


# ========================================
# How to overlap two ROC curves
# so that they are easy to compare
# ========================================

# Make a Random Forest classifier with the same dataset
rf.model <- randomForest(factor(obese)~weight)

# Draw the original ROC curve for the Logistic Regression
roc(obese, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE,
    xlab="False Positive Percentage",
    ylab="True Positive Percentage",
    col="#377eb8", lwd=4,
    print.auc=TRUE)

# Add the ROC curve for the Random Forest
#
# rf.model$votes[,1]
# Since we are using a Random Forest for the second ROC,
# pass in the number of trees in the forest that voted corrctly
plot.roc(obese, rf.model$votes[,1], percent=TRUE, col="#4daf4a", lwd=4,
         print.auc=TRUE, add=TRUE, print.auc.y=40)

# Draw a legend in the bottom right-hand corner
legend("bottomright", legend=c("Logistic Regression", "Random Forest"),
       col=c("#377eb8", "#4daf4a"), lwd=4)


# ========================================
# ★ Once we're all done drawing ROC graphs,
# we ned to reset the pty graphcial parameter back to its default value, m
# which is short of Maximum
# As in, "use the maximum amount of space provided to draw graphs"
# ========================================

par(pty="m")

