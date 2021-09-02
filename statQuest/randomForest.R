
library(ggplot2)

# Improve some of ggplot2's default settings
install.packages("cowplot")
library(cowplot)

library(randomForest)

# We're going to get a real dataset from the UCI machine learning repository
# we'll use 'Heart disease' dataset
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

data <- read.csv(url, header=FALSE)

# Unfortunately, none of the columns are labeled
head(data)

# Name the columns after the names that were listed on the UCI website
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

# Some of the columns are messed up
# ex) 'sex' is supposed to be a factor, where 0 represents "female" and 1 represents "male"
#     'cp' (chest pain) is also supposed to be a factor,
#          where levels 1~3 represent different types of pain and 4 represents no chest pain
str(data)


# ==================================================
# Cleaning up the data
# ==================================================

# Change the "?"s to NAs
data[data == "?"] <- NA

# Convert the 0s in sex to F, for female and 1s to M, for male
data[data$sex == 0,]$sex <- "F"
data[data$sex == 1,]$sex <- "M"

# Convert the column into a factor
data$sex <- as.factor(data$sex)
data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)

# Since the 'ca' column originally had ? in it, rather than NA,
# R rhinks it's a column of strings
# ★ We correct that assumption by telling R that it's a column of integers
# then convert it to a factor
data$ca <- as.integer(data$ca)
data$ca <- as.factor(data$ca)

data$thal <- as.integer(data$thal)
data$thal <- as.factor(data$thal)

# 'hd' (heart disease)
# ifelse() to convert the 0s to "Healthy" and the others to "Unhealthy"
# ★ We could have done a similar trick for sex
data$hd <- ifelse(test=data$hd == 0, yes="Healthy", no="Unhealthy")
data$hd <- as.factor(data$hd)

# Check the data we have made the appropriate changes
str(data)


# ==================================================
# Impute the missing values
# ==================================================

# Set the seed for the random number generator so that we can reproduce our results
set.seed(42)

# Impute values for the NAs in the dataset with rfImpute()
# This means we want the hd column to be predicted by the data in all of the other columns
# ★ iter : How many random forests rfImpute() should build to estimate the missing values
#           In theory, 4 to 6 iterations is enough
#           Just for fun, I set iter=20, but it didn't improve the estimates
data.imputed <- rfImpute(hd~., data=data, iter=6)

# After each iteration, rfImpute() prints out the Out-Of-Bag (OOB) error rate
# ★ This should get smaller if the estimates are improving
# Since it doesn't, we can conclude our estimates are as good as they are going to get with this method


# ==================================================
# Random Forest
# ==================================================

# Predict hd using all of the other columns in the dataset
# proximity=TRUE : We want randomForest() to return the proximity matrix
#                  ★ We'll use it to cluster the samples
model <- randomForest(hd~., data=data.imputed, proximity=TRUE)

# Get a summary of the random forest and how well it performed
model

# ==================================================
# Result

# Type of random forest: classification
# We can see that the random forest was bult to classify samples
# If we had used the random forest to predict weight or height, it would say regression
# And if we had omitted the thing the random forest was supposed to predict entirely, it would say ★ "unsupervised"

# Number of trees: 500
# Then it tells us how many trees are in the random forest
# The default value is 500
# Later we will check to see if 500 trees is enough for optimal classification

# No. of variables tried at each split: 3
# Then it tells us how many variables (or columns of data) were considered at each internal node
# ★ Classification trees have a default setting of the square root of the number of variables
# ★ Regression trees have a default setting of the number of variables divided by 3
# Since we don't now if 3 is the best value, we'll fiddle with this parameter later on

# OOB estimate of  error rate: 16.83%
# Here's the Out-Of-Bag (OOB) error estimate
# This means that 83.17% of the OOB samples were correctly classified by the random forest

# Lastly, we have a confusion matrix
# There were 142 healthy patients that were correctly labeled "Healthy"
# There were 29 unhealthy patients that were incorrectly classified "Healthy"
# There were 22 healthy patients that were incorrectly classified "Unhealthy"
# There were 110 unhealthy patients that were correctly classified "Unhealthy"
# ==================================================


# This is what the err.rate matrix looks like
# There is one column for the OOB error rate
# One column for the Healthy error rate (how frequently healthy patients are misclassified)
# One column for the unhealthy error rate (how frequently unhealthy patients are misclassified)
# Each row reflects the error rates at different stages of creating the random forest
# The first row contains the error rate after making the first tree
# The second row contains the error rate after making the first two trees
# The last row contains the error rate after making the all 500 trees
head(model$err.rate, 10)

# To see if 500 trees is enough for optimal classification, we can plot the error rates
oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "Healthy", "Unhealthy"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"],
          model$err.rate[,"Healthy"],
          model$err.rate[,"Unhealthy"])
)

# There's one column for the number of trees
# One column for the type of error
# One column for the actual error value
head(oob.error.data,10)

# Plot the error rates
# The blue line shows the error rate when classifying unhealthy patients
# The green line shows the overall OOB error rate
# The red line shows the error rate when classifying healthy patients
# ★ In general, we see the error rates decrease when our random forest has more trees
ggplot(data=oob.error.data, aes(x=Trees, y=Error))+
  geom_line(aes(color=Type))


# ==================================================
# Add more trees
# ==================================================

# If we added more trees, would the error rate go down further?
model <- randomForest(hd~., data=data.imputed, ntree=1000, proximity=TRUE)

# The OOB error rate is the same as before
# And the confusion matrix shows that we didn't do a better job classifying patients
model

# Plot the error rates
oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "Healthy", "Unhealthy"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"],
          model$err.rate[,"Healthy"],
          model$err.rate[,"Unhealthy"])
)

# We see that the error rates stabilize right after 500 trees
# So adding more trees didn't help
ggplot(data=oob.error.data, aes(x=Trees, y=Error))+
  geom_line(aes(color=Type))


# ==================================================
# Now we need to make sure we are considering
# the optimal number of variables
# at each internal node in the tree
# ==================================================

# ★ Make an empty vector that can hold 10 values
oob.values <- vector(length=10)

# Create a loop that tests "different numbers of variables" at each step
# Each time we go through the loop, "i" increases by 1
# It starts at 1 and ends after 10
for(i in 1:10) {
  temp.model <- randomForest(hd~., data=data.imputed, mtry=i, ntree=1000)
  
  # Store the OOB error rate after we build each random forest
  # that uses a different value for mtry
  # ★ nrow(temp.model$err.rate), 1
  # → Access the value in the last row and in the first column
  #   i.e. the OOB error rate when all 1000 trees have been made
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate), 1]
}

# Print out the OOB error rate for different values for mtry
# ★ The 3rd value, corresponding to mtry=3, which is the default in this case,
# has the lowest OOB error rate
# So the default value was optimal
oob.values


# ==================================================
# Use the Random Forest to draw an MDS plot with samples
# This will show us how they are related to each other
# ==================================================

# Make a distance matrix from 1 - proximity matrix
distance.matrix <- dist(1-model$proximity)

# Classical multi-dimensional scale
mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)

# Calculate the percentage of variation in the distance matrix what the X and Y axes account for
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

# Format the data for ggplot
mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2],
                       Status=data.imputed$hd)

# Unhealthy samples are on the left side
# Healthy samples are on the right side
# The x-axis accounts for 47% of the variation in the distance matrix
# The y-axis accounts for only 13% of the variation in the distance matrix
# That means that the big differences are along the X-axis
ggplot(data=mds.data, aes(x=X, y=Y, label=Sample))+
  geom_text(aes(color=Status))+
  theme_bw()+
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep=""))+
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep=""))+
  ggtitle("MDS plot using (1 - Random Forest Proximity)")

# If we got a new patient and didn't know if they had heart disease
# and they clustered down the left side,
# we'd be pretty confident that they had heart disease


