url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data <- read.csv(url, header=FALSE)
head(data)

colnames(data) <- c(
  "age",
  "sex",
  "cp",
  "trestbps",
  "chol",
  "fbs",
  "restecg",
  "thalach",
  "exang",
  "oldpeak",
  "slope",
  "ca",
  "thal",
  "hd"
)

head(data)
str(data)

data[data == "?"] <- NA
data[data$sex == 0,]$sex <- "F"
data[data$sex == 1,]$sex <- "M"
data$sex <- as.factor(data$sex)

data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)

data$ca <- as.integer(data$ca)
data$ca <- as.factor(data$ca)

data$thal <- as.integer(data$thal)
data$thal <- as.factor(data$thal)

data$hd <- ifelse(data$hd==0, "Healthy", "Unhealthy")
data$hd <- as.factor(data$hd)

str(data)

# See how many samples (rows of data) have NA values
# Later we will decide if we can just toss these samples out
# or if we should impute values for the NAs
nrow(data[is.na(data$ca) | is.na(data$thal),])

# View the samples with NAs by selecting those rows from the data frame
data[is.na(data$ca) | is.na(data$thal), ]

# If we wanted to, we can impute values for the NAs using a Random Forest or some other method
# However, for this example, we'll just remove these samples

# Including the 6 samples (rows) with NAs, there are 303 samples (rows)
nrow(data)

# Remove the 6 samples (rows) that have NAs
data <- data[!(is.na(data$ca) | is.na(data$thal)), ]

# After removing those samples (rows), there are 297 samples remaining
nrow(data)


#===============================================
# Create a table to check
# whether each column's level has
# appropriate size to predict heart disease
#===============================================

# Make sure that healthy and diseased samples come from each gender (female and male)
# ★ xtabs() function : use "model syntax" to select the columns in the data we want to build a table from
# In this case, we want a table with heart disease and sex
xtabs(~hd + sex, data=data)

# Verify that all 4 levels of Chest Pain (cp for short) were reported by a bunch of patients
xtabs(~hd + cp, data=data)

# Do the same thing for all of the boolean and categorical variables that we are using to predict heart disease
xtabs(~hd + fbs, data=data)

# ★ Here's something that can cause trouble for the "Resting Electrocardiographic Results" (restecg)
# Only 4 patients represent level 1
# This could, potentially, get in the way of finding the best fitting line
# However, for now we'll just leave it and see what happens
xtabs(~hd + restecg, data=data)

# We just keep looking at the remaining variables to make sure that they are all represented by a number of patients
xtabs(~hd + exang, data=data)
xtabs(~hd + slope, data=data)
xtabs(~hd + ca, data=data)
xtabs(~hd + thal, data=data)


#===============================================
# Do Logistic Regression
#===============================================

# Super Simple Model
# We'll try to predict heart disease (hd) using only the gender (sex) of each patient
# ★ family="binomial"
# - Specify that we want the binomial family of generalized linear models
# - This makes the glm() function do Logistic Regression,
#   as apposed to some other type of generalized linear model
logistic <- glm(hd~sex, data=data, family="binomial")

# Get details about the logistic regression
summary(logistic)

#===============================================
# Result Interpretation

# Call:
# The first line has the original call to the glm() function

# Deviance Residuals:
# It gives you a summary of the deviance residuals
# They look good since they are close to being centered on 0 and are roughly symmetrical

# Coefficients:
# They correspond to the following model:
# heart disease = -1.0438 + 1.2737 * the patient is male
# - The variable, the patient is male, is equal to 0 when the patient is female
#   and 1 when the patient is male
# - Thus, if we are predicting heart disease for a female patient, we get the following equation:
#   heart disease = -1.0438 + 1.2737 * 0
#   → heart disease = -1.0438
#   Thus, the log(odds) that a female has heart disease = -1.0438
# - If we are predicting heart disease for a male patient, we get the following equation:
#   heart disease = -1.0438 + 1.2737 * 1
#   → heart disease = -1.0438 + 1.2737
#   Since the first term (-1.0438) is the log(odds) of a female having heart disease,
#   the second term (1.2737) indicates the increase in the log(odds) that a male has of having heart disease
#   In other words, the second term is the log(odds ratio) of the odds that a male will have heart disease
#   over the odds that a female will have heart disease

# Std.Error, z value
# This part of the logistic regression output shows how the Wald's test was computed for both coefficients

# Pr(>|z|)
# Here are the p-values
# Both p-values are well below 0.05, and thus, the log(odds) and the log(odds ratios)
# are both statistically significant
# But remember, a small p-value alone isn't interesting
# We also want large effect sizes, and that's what the log(odds) and log(odds ratio) tells us

# (Dispersion parameter for binomial family taken to be 1)
# We see the default dispersion parameter used for this logistic regression
# When we do "normal" linear regression, we estimate both the mean and the variance from the data
# In contrast, with logistic regression, we estimate the mean of the data,
# and the variance is derived from the mean
# Since we are not estimating the variance from the data
# (and, instead, just deriving it from the mean),
# it is possible that the variance is underestimated
# If so, you can adjust the dispersion parameter in the summary() command

# Null deviance:, Residual deviance:
# These can be used to compare models, compute R-squared and an overall p-value

# AIC: 
# The Akaike Information Criterion, in this context, is just the Residual Deviance
# adjusted for the number of parameters in the model
# The AIC can be used to compare one model to another

# Number of Fisher Scoring iterations: 4
# It tells us how quickly the glm() function converged on the maximum likelihood estimates for the coefficients
#===============================================


# Create a fancy model that uses all of the variables to predict heart disease
logistic <- glm(hd ~., data=data, family="binomial")

summary(logistic)

#===============================================
# Result Interpretation

# We see that 'age' isn't useful predictor
# because it has a large p-value
# However, the median age in our dataset was 56,
# so most of the folks were pretty old and that explains why it wasn't very useful

# Gender is still a good predictor, though

# We see that the Residual Deviance and the AIC are both much smaller for this fancy model
# ★ than they were for the simple model,
# when we only used gender to predict heart disease
#===============================================


# If we want to calculate McFadden's Pseudo R2,
# we can pull the log-likelihood of the null model out of the logistic variable
# by getting the value for the null deviance and dividing by -2
ll.null <- logistic$null.deviance/-2

# We can pull the log-likelihood for the fancy model out of the logistic variable
# by getting the value for the residual deviance and dividing by -2
ll.proposed <- logistic$deviance/-2

# Then we just do the math and we end up with a Pseudo R2 = 0.55
# This can be interpreted as the overall effect size
(ll.null - ll.proposed) / ll.null

# We can use those same log-likelihoods to calculate a p-value
# for that R2 using a Chi-square distribution
# In this case, the p-value is tiny, so the R2 value isn't due to dumb luck
1 - pchisq(2*(ll.proposed - ll.null), df=(length(logistic$coefficients)-1))


#===============================================
# Draw a graph
# that shows the predicted probabilities
# that each patient has heart disease
# along with their actual heart disease status
#===============================================

# To draw the graph, we start by creating a new data.frame
# that contains the probabilities of having heart disease along with the actual heart disease status
predicted.data <- data.frame(
  probability.of.hd=logistic$fitted.values,
  hd=data$hd
)

# Then sort the data.frame from low probabilities to high probabilities
predicted.data <- predicted.data[order(predicted.data$probability.of.hd, decreasing=FALSE),]

# Add a new column to the data.frame that has the rank of each sample,
# from low probability to high probability
predicted.data$rank <- 1:nrow(predicted.data)

library(ggplot2)
library(cowplot) # ggplot can have nice looking defaults

ggplot(data=predicted.data, aes(x=rank, y=probability.of.hd))+
  geom_point(aes(color=hd), alpha=1, shape=4, stroke=2)+
  xlab("Index")+
  ylab("Predicted probability of getting heart disease")

# Graph Interpretation
# Most of the patients with heart disease (the ones in turquoise)
# are predicted to have a high probability of having heart disease
# And most of the patients without heart disease (the ones in salmon)
# are predicted to have a low probability of having heart disease
# Thus, our logistic regression has done a pretty good job
# However we could use cross-validation to get a better idea of how well it might perform with new data,
# but, we'll save that for another day

# ★ Save the graph as a PDF file
ggsave("heart_disease_probabilities.pdf")

