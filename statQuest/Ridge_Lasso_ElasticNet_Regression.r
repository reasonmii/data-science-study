install.packages("glmnet")
library(glmnet)

set.seed(42)

# ======================================================
# Make up a dataset
# ======================================================

# The made up dataset will have n = 1,000 samples
# and p = 5,000 parameters to estimate
# However, only 15 those parameters will help up predict the outcome
# The remaining 4,985 parameters will just the random noise
n <- 1000
p <- 5000
real_p <- 15

# Create a matrix called x that is full of randomly generated data
# nrow, ncol : The matrix has 1,000 rows (since n = 1,000) and 5,000 columns (since p = 5,000)
# rnorm : The values in the matrix come from a standard normal distribution (with mean = 0 and standard deviation = 1)
#         We'll need n * p (or 1,000 * 5,000 = 5,000,000) values, since our matrix has n rows and p columns
x <- matrix(rnorm(n*p), nrow=n, ncol=p)

# Create a vector of values, called y
# that we will try to predict with the data in x
#
# apply() : It will return a vector of 1,000 values that are the sums of the first 15 columns in x,
#           since x has 1,000 rows
#   - x[, 1:real_p] : This is what isolates columns 1 through 15 from x
#   - 1 : It specifies that we want to perform a function on each row of data that we've isolated from x
#   - sum : The function that we want to apply to each row
#   - To summarize, this call to apply() will return a vector of values that depend on the first 15 columns in x
#
# rnorm() : Once we have that vecotr of sums, we add a little bit of noise using the rnorm() function
#           which, in this case, returns 1,000 random values from a standard normal distribution
#
# So this whole thing creates a vector called y that is dependent on the frist 15 columns in x,
# plus a little noise to make things interesting
y <- apply(x[, 1:real_p], 1, sum) + rnorm(n)

# Thus, x is a matrix of data that we will use
# Ridge, Lasso and Elastic-Net Regression to predict the value in y


# ======================================================
# Divide the data into Training and Testing Sets
# ======================================================

# Make a vector of indexe, called train_rows,
# that contains the row numbers of the rows that will be in the Training set
# sample() : It randomly selects numbers between 1 and n, the number of rows in our dataset
# .66*n : It will select 0.66 * n row numbers
#         In other words, two-thirds of the data will be in the Training set
train_rows <- sample(1:n, .66*n)

# Now that we have the indexes for the rows in the training set in train_rows
# we can make a new matrix, x.train, that just contains the Training data
x.train <- x[train_rows, ]

# And we can make a Testing set, x.test that contains the remaining rows
# ★ This is done by putting a negative sign in front of train_rows when we select the rows from x
x.test <- x[-train_rows, ]

# Now select the training values in y and save them in y.train
y.train <- y[train_rows]

# Select the testing values in y and save them in y.test
y.test <- y[-train_rows]

# We'll apply Ridge, Lasso and Elastic-Net regression separately
# to these datasets so that we can see how it's done and see which method works best


# ======================================================
# Ridge Regression
# ======================================================

# Fit a model to the training data
# cv : It means we want to use "Cross Validation" to obtain the optimal values for λ
#      ★ By default, cv.glmnet() uses 10-Fold Cross Validation
#
# x.train, y.train : It specifies the Training sets
#                    Unlike the lm() or glm() functions, cv.glmnet() does not accept formula notation;
#                    x and y must be passed in separately
#
# type.measure="mse" : It is how the cross-validation will be evaluated
#                      MSE stands for "mean squared error", just the sum of the squared residuals divided by the sample size
#                      ★ NOTE : If we were applying Elastic-Net Regression to Logistic Regression, we would set this to "deviance"
#
# alpha=0 : Since we're starting with Ridge Regression, we set alph to 0
#
# family="gaussian" : This tells glmnet that we are doing Linear Regression
#                     ★ NOTE : If we were doing Logistic Regression, we would set this to "binomial"
#
# All together, this call to cv.glmnet() will fit a Linear Regression with a Ridge Regression penalty
# using 10-fold Cross Validation to find optimal values for λ
# and the fit model, along with optimal values for λ, is saved as alpha0.fit,
# which will help us remember that we set alpha to 0 for Ridge Regression
alpha0.fit <- cv.glmnet(x.train, y.train, type.measure="mse", alpha=0, family="gaussian")

# Now we will use the predict() function to apply alpha0.fit to the Testing data
#
# alpha0.fit : It is a fitted model
#
# s=alpha0.fit$lambda.1se
# s, which Josh thinks stands for "size", as in "the size of the penalty"
# is set to one of the optimal values for λ stored in alpha0.fit
# In this example, we are setting s to lambda.1se
# lambda.1se is the value for λ, stored in alpha0.fit,
# that resulted in the simplest model (i.e. the model with the fewest non-zero parameters)
# and was within 1 standard error of the λ that had the smallest sum
#
# NOTE : Alternatively, we can set s to lambda.min, which would be the λ that resulted in the smallest sum
# However, in this example, we will use lambda.1se
# because, in a statistical sense, it is indistinguishable from lambda.min,
# but it results in a model with fewer parameters
#
# Wait a minute! I thought only Lasso and Elastic-Net Regression could eliminate parameters
# What's going on?
# Since we will compare Ridge to Lasso and Elastic-Net Regression,
# we will use lambda.1se for all three cases to be consistent
alpha0.predicted <- predict(alpha0.fit, s=alpha0.fit$lambda.1se, newx=x.test)

# Calculate the mean squared error of the difference between the true values,
# stored in y.test, and the predicted values, stored in alpha0.predicted
mean((y.test - alpha0.predicted)^2)


# ======================================================
# Lasso Regression
# ======================================================

# Just like before, call cv.glmnet() to fit a Linear Regression using 10-Fold Cross Validation to determine optimal values for λ
# Only this time we set alpha to 1
alpha1.fit <- cv.glmnet(x.train, y.train, type.measure="mse", alpha=1, family="gaussian")

# Call the predict() function
alpha1.predicted <- predict(alpha1.fit, s=alpha1.fit$lambda.1se, newx=x.test)

# Calculate the mean squared error
mean((y.test - alpha1.predicted)^2)

# 1.18 is way smaller than 14.88
# So, Lasso Regression is much better than this data than Ridge Regression


# ======================================================
# Elastic-Net Regression
# ======================================================

# Call cv.glmnet() todetermine optimal values for λ
alpha0.5.fit <- cv.glmnet(x.train, y.train, type.measure="mse", alpha=0.5, family="gaussian")

# Call the predict() function
alpha0.5.predicted <- predict(alpha0.5.fit, s=alpha0.5.fit$lambda.1se, newx=x.test)

# Calculate the mean squared error
mean((y.test - alpha0.5.predicted)^2)

# 1.23 is slightly larger than the 1.18, we got with Lasso Regression
# So, so far, Lasso wins
# But to really know if Lasso wins, we need to try a lot of different values for alpha


# ======================================================
# Try a bunch of values for alpha
# ======================================================

# We'll start by making an empty list called list.of.fits
# that will store a bunch of Elastic-Net Regression fits
list.of.fits <- list()

# Then use a for loop to try different values for alpha
for (i in 0:10) {
  
  # First, paste together a name for the Elastic-Net fit that we are going to create
  # ex) when i=0, then fit.name will be alpha0 (0/10)
  # ex) when i=1, then fit.name will be alpha0.1 (1/10)
  fit.name <- paste0("alpha", i/10)
  
  # Create the Elastic-Net fit using the cv.glmnet() function
  # When i=0, then alpha will be 0 and result in Ridge Regression
  # When i=1, then alpha will be 0.1
  # When i=10, then alpha will be 1 and result in Lasso Regression
  # Each fit will be stored in list.of.fits under the name we stored in fit.name
  list.of.fits[[fit.name]] <-
    cv.glmnet(x.train, y.train, type.measure="mse", alpha=i/10,
              family="gaussian")
}

# Now we are ready to calculate the mean squared errors for each fit with the Testing dataset

# Start by creating an empty data.frame, called results,
# that will store the mean squared errors and a few other things
results <- data.frame()

# Use another for loop to predict the values using the Testing dataset
# and to calculate the mean squared errors
for (i in 0:10) {
  
  # Create a variable caalled fit.name that contanis the name of the Elastic-Net Regression fit
  fit.name <- paste0("alpha", i/10)
  
  # Use the list.of.fits list and fit.name to pass a specific fit to the predict() function
  predicted <-
    predict(list.of.fits[[fit.name]],
            s=list.of.fits[[fit.name]]$lambda.1se,
            newx=x.test)
  
  # Calculate the mean squared error
  mse <- mean((y.test - predicted)^2)
  
  # Store the value for alpha, the mean squared error
  # and the name of the fit in a temporary data.frame called temp
  temp <- data.frame(alpha=i/10, mse=mse, fit.name=fit.name)
  
  # Use the rbind() function to append temp to bottom row of the results data.frame
  results <- rbind(results, temp)
}

# Print out the result
#
# The first column has the values for alpha, ranging from 0 to 1
#
# The second column has the mean squared errors
# NOTE : These are slightly different from what we got before,
#        because the parameter values, prior to regularization and optimization are randomly initialized
#        ★ Thus, this is another good reason to use set.seed() to ensure consistent results
#
# In the last column, we have the name of the fit
#
# The fit where the alpha=1, is still the best,
# so Lasso Regression is the best method to use with this data
results
