# Generate a fake dataset that we can use in the demonstration
# A matrix of data with 10 samples where we measured 100 genes in each sample
data.matrix <- matrix(nrow=100, ncol=10)

# name the samples
# The first 5 samples will be "wt" or "wild type" samples
# "wt" samples are the normal, every day samples
# The last 5 samples will be "ko" or "knock-out" samples
# These are samples that are missing a gene because we knocked it out
colnames(data.matrix) <- c(
  paste("wt", 1:5, sep=""),
  paste("ko", 1:5, sep="")
)

# name the genes
# Usually you'd have things like "Sox9" AND "Utx"
# but since this is a fake dataset, we have gene1, gene2, ..., gene100
rownames(data.matrix) <- paste("gene", 1:100, sep="")

# This is where we give the fake genes fake read counts
# rpois : used poisson distribution
for (i in 1:100) {
  wt.values <- rpois(5, lambda=sample(x=10:1000, size=1))
  ko.values <- rpois(5, lambda=sample(x=10:1000, size=1))
  
  data.matrix[i, ] <- c(wt.values, ko.values)
}

# First 6 rows in our data matrix
# NOTE : The samples are columns, and the genes are rows
head(data.matrix)

# Call prcomp() to do PCA on our data
# The goal is to draw a graph that shows how the samples are related (or not related) to each other
# NOTE : By default, prcomp() expects the samples to be rows and the genes to be columns
# Since the samples in our data matrix are columns, and the genes(variables) are rows
# ★ we have to transpose the matrix using the t() function
# If we don't transpose the matrix,
# we will ultimately get a graph that shows how the genes are related to each other
pca <- prcomp(t(data.matrix), scale=TRUE)

# ★ prcomp() returns three things: x, sdev, rotation

# x contains the principal components (PCs) for drawing a graph
# Here we are using the first two columns in x to draw a 2-D gaph
# that uses the first two PCs
# Remember! Since there are 10 samples, there are 10 PCs
# The first PC accounts for the most variation in the original data
# (the gene expression across all 10 samples),
# the 2nd PC accounts for the second most variation and so on
# To plot a 2-D PCA graph, we usually use the first 2 PCs
# However, sometimes we use PC2 and PC3
plot(pca$x[,1], pca$x[,2])

# Graph interpretation
# 5 of the samples are on one side of the graph
# and the other 5 samples are on the other side of the graph
# To get a sense of how meaningful these clusters are,
# let's see how much variation in the original data PC1 accounts for

# To do this, we use the square of "sdev"
# which stands for "standard deviation",
# to calculate how much variation in the original data each principal component accounts for
pca.var <- pca$sdev^2

# Since the percentage of variation that each PC accounts for is way more interesting
# than the actual value, we calculate the percentages
pca.var.per <- round(pca.var/sum(pca.var)*100, 1)

# Plot the percentages is easy with barplot()
barplot(pca.var.per, main="Scree Plot", xlab="Principal Component", ylab="Percent Variation")

# Graph interpretation
# PC1 accounts for almost all of the variation in the data!
# This means that there is a big difference between two clusters
# (5 on the left and 5 on the right)

# We can use ggplot2 to make a fancy PCA plot that looks nice
# and also provides us with tons of information
install.packages("ggplot2")
library(ggplot2)

# First, format the data the way ggplot2 likes it
# We made a data frame with one column with the sample ids,
# two columns for the X and Y coordinates for each sample
pca.data <- data.frame(Sample=rownames(pca$x),
                       X=pca$x[,1],
                       Y=pca$x[,2])

# Here's what the dataframe looks like
# We have one row per sample
# Each row has a sample ID and X/Y coordinates for that sample
pca.data

# The X-axis tells us what percentage of the variation in the original data that PC1 accounts for
# The Y-axis tells us what percentage of the variation in the original data that PC2 accounts for
# Now the samples are labeled, so we know which ones are on the left and the right
# ★ geom_text : plot the labels, rather than dots or some other shape
# ★ theme_bw : makes the graph's background white
ggplot(data=pca.data, aes(x=X, y=Y, label=Sample)) +
  geom_text() +
  xlab(paste("PC1 - ", pca.var.per[1], "%", sep=""))+
  ylab(paste("PC2 - ", pca.var.per[2], "%", sep=""))+
  theme_bw() +
  ggtitle("My PCA Graph")

# Lastly, let's look at how to use loading scores to determine
# which genes have the largest effect on where samples are plotted in the PCA plot
# ★ The prcomp() function calls the loading scores rotation
# There are loading scores for each PC
# Here we're just going to look at the loading scores for PC1
# since it accounts for 92% of the variation in the data
loading_scores <- pca$rotation[,1]

# Genes that push samples to the left side of the graph will have large negative values
# and genes that push samples to the right will have large positive values
# Since we're interested in both sets of genes,
# we'll use the abs() function to sort based on the number's magnitude rather than from high to low
gene_scores <- abs(loading_scores)

# Now we sort the magnitude of the loading scores, from high to row
gene_score_ranked <- sort(gene_scores, decreasing=TRUE)

# Now we get the names of top 10 genes with the largest loading score magnitudes
top_10_genes <- names(gene_score_ranked[1:10])
top_10_genes

# Show the scores (and +/- sign)
# Lastly, we can see which of these genes have positive loading scores, these push the "ko" samples to the right side of the graph
# then we see which genes have negative loading scores, these push the "wt" samples to the left side of the graph
pca$rotation[top_10_genes,1]
