```{r}
mouse.data <- data.frame (
  weight = c(0.9, 1.8, 2.4, 3.5, 3.9, 4.4, 5.1, 5.6, 6.3),
  size = c(1.4, 2.6, 1.0, 3.7, 5.5, 3.2, 3.0, 4.9, 6.3))

mouse.data

plot(mouse.data$weight, mouse.data$size)
mouse.regression <- lm(size ~ weight, data = mouse.data)
summary(mouse.regression)

abline(mouse.regression, col="blue")
```

## Summary Interpretation

### Call: The first line just prints out the original call to the lm() function

### Residuals:
This is a summary of the residuals (the distance from the data to the fitted line)
Ideally, they should be symmetrically distributed around the line

### Coefficients:

- This first section tells us about the least-squares estimates for the fitted line<br/>The first line is for the intercept, and the second line is for the slope<br/>ex) size = 0.5813 + 0.7778 * weight

- Residual standard error is the square root of the denominator in the equation for F<br/>The standard error of the estimates and the “t value” are both provided to show you how the p-values were calculated

- Lastly, Pr(>|t|) parts are the p-values for the estimated parameters<br/>Generally speaking, we are usually not interested in the intercept, so it doesn’t matter what its p-value is<br/>However, we want the p-value for “weight” to be < 0.05<br/>That is, we want it to be statistically significant<br/>A significant p-value for weight means that it will give us a reliable guess of mouse size<br/>If you’re unable to read the actual p-value, but could, for some reason, see the star to its right, then these codes would give you a sense of what the p-value was

### Multiple R-squared is just R2 as we describe it in the StatQuest on Linear Regression
It means that weight can explain 61% of the variation in size. This is good!

### Generally speaking, the Adjusted R-squared is the R2 scaled by the number of parameters in the model

### The last line tells us whether the R2 is significant or not
- F-statistic: This is the value for F
- 1 and 7 DF : these are the degrees of freedom
- p-value : Here’s our p-value! Again this says that weight gives us a reliable estimate for size (0.01256 < 0.05)
