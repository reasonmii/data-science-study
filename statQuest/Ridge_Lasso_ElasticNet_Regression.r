
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
# s=alpha0.fit$lambda.1se : s, which Josh thinks stands for "size", as in "the size of the penalty"
#                           is set to one of the optimal values for λ stored in alpha0.fit
#                           In this example, we are setting s to lambda.1se
#                           lambda.1se is the value for λ, stored in alpha0.fit,
#                           that resulted in the simplest model (i.e. the model with the fewest non-zero parameters)
#                           and was within 1 standard error of the λ that had the smallest sum
alpha0.predicted <- predict(alpha0.fit, s=alpha0.fit$lambda.1se, newx=x.test)















