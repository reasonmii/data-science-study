########################
# Simple Regression
########################

# The raw data for mouse size, weight and tail length
mouse.data <- data.frame(
  size = c(1.4, 2.6, 1.0, 3.7, 5.5, 3.2, 3.0, 4.9, 6.3),
  weight = c(0.9, 1.8, 2.4, 3.5, 3.9, 4.4, 5.1, 5.6, 6.3),
  tail = c(0.7, 1.3, 0.7, 2.0, 3.6, 3.0, 2.9, 3.9, 4.0)
)

mouse.data

# Plot your data
plot(mouse.data$weight, mouse.data$size)

# Use the lm() (linear model) function to fit a line to the data
# size = y-intercept + slope * weight
# By default, R adds the terms for the y-intercept and the slope
# R then uses least-squares to find the values for y-intercept and slope
# that minimize the squared residuals from the line
simple.regression <- lm(size ~ weight, data=mouse.data)

# The R2 (0.613) and the p-value (0.012) say that weight does a pretty good job predicting size
summary(simple.regression)

# Add the least-squares fit line to the graph
abline(simple.regression, col="red", lwd=2)


########################
# Multiple Regression
########################

# For multiple regression, we'll use weight and tail to predict size
mouse.data <- data.frame(
  size = c(1.4, 2.6, 1.0, 3.7, 5.5, 3.2, 3.0, 4.9, 6.3),
  weight = c(0.9, 1.8, 2.4, 3.5, 3.9, 4.4, 5.1, 5.6, 6.3),
  tail = c(0.7, 1.3, 0.7, 2.0, 3.6, 3.0, 2.9, 3.9, 4.0)
)

# Plot your data
# Since we didn't specify the x and y-axes,
# R plots all the data columns (size, weight and tail) against each other
# We can see that both weight and tail are correlated with size
# It means that both weight and tail are reasonable predictors for size
# We can also see that weight and tail are correlated
# - this meas that they provide similar information and that we might not need both in our model
# - we might only need weight or tail
plot(mouse.data)

# Use the lm() (linear model) function to fit a plane to the data
# size = y-intercept + slope1 * weight + slope2 * tail
multiple.regression <- lm(size ~ weight + tail, data=mouse.data)

# The R2, adjusted R2 and the p-value look good
# ★ Since we're doing multiple regression, we're more interested in adjusted R2
# ★ With multiple regression, "Coefficients" section is more interesting
# ★ 'Coefficients : weight' line compares the multiple regression to the simple regression
#    - size = y-intercept + slope1 * weight + slope2 * tail
#      vs. size = y-intercept + slope2 * tail
#    - 0.4345 is the p-value
#      ★ it means that using weight and tail isn't significantly better than using tail alone to predict size
# Now let's look at 'Coefficients : tail' line
#    - size = y-intercept + slope1 * weight + slope2 * tail
#      vs. size = y-intercept + slope1 * weight
#    - 0.0219 is the p-value
#      ★ it means that using weight and tail is significantly better than using weight alone to predict size
# In summary, using weight and tail to predict size is good (p-value : 0.003399)
# but if we wanted to save time, we could spare ourselves the agony of weighting mice and just use their tail lengths to predict size
summary(multiple.regression)
