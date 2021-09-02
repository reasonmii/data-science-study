
library(ggplot2)

# Generate some fake data
# The data will consist of a matrix with 10 columns,  corresponding to 10 samples,
# and 100 rows, corresponding to measurements from 100 genes
data.matrix <- matrix(nrow=100, ncol=10)

# The first 5 columns will be "wt" (or wild-type) samples
# and the last 5 columns will be "ko" (or knock-out) samples
colnames(data.matrix) <- c(
  paste("wt", 1:5, sep=""),
  paste("ko", 1:5, sep="")
)

# The genes will have really creative names like, "gene1" and "gene2"
rownames(data.matrix) <- paste("gene", 1:100, sep="")

# Generate the fake data
for(i in 1:100) {
  wt.values <- rpois(5, lambda=sample(x=10:1000, size=1))
  ko.values <- rpois(5, lambda=sample(x=10:1000, size=1))
  
  data.matrix[i, ] <- c(wt.values, ko.values)
}

head(data.matrix)


# =====================================================================================
# Just for comparison, do PCA on the dataset
# =====================================================================================

pca <- prcomp(t(data.matrix), scale=TRUE, center=TRUE)
pca.var <- pca$sdev^2
pca.var.per <- round(pca.var/sum(pca.var)*100, 1)
pca.var.per

pca.data <- data.frame(Sample=rownames(pca$x),
                       X=pca$x[,1],
                       Y=pca$x[,2])
pca.data

# The wild type is on the left side
# and the knock-out type is on the right side
# The x-axis, for PC1, accounts for 87.7% of the variation in the data
# The y-axis, for PC2, only accounts for 2.9% of the variation in the data
# This means most of the differences are between the WT and the KO samples
ggplot(data=pca.data, aes(x=X, y=Y, label=Sample))+
  geom_text()+
  xlab(paste("PC1 = ", pca.var.per[1], "%", sep=""))+
  ylab(paste("PC2 - ", pca.var.per[2], "%", sep=""))+
  theme_bw()+
  ggtitle("PCA Graph")


# =====================================================================================
# Create an MDS/PCoA plot to compare to this one!
# =====================================================================================

# Step1 : Create a distance matrix
# We do this with the dist() function
# Just like with PCA, we transpose the matrix so that the samples are rows
# We also center and scale the measurements for each gene (which are now the columns)
# Lastly, we tell the dist() function that we want it to create the matrix
# using the Euclidean distance metric
distance.matrix <- dist(scale(t(data.matrix), center=TRUE, scale=TRUE),
                        method="euclidean")

# ★ NOTE : The dist() function has 6 different distance metrics to choose from

# Step2 : Perform multi-dimensional scaling on the distance matrix using the cmdscale() function
# ★ cmdscald() stands for Classical Multi-Dimensional Scaling
# eig=TRUE : Tell cmdscale() that we want it to return the 'eigen values'
#            We use these to calculate how much variance in the distance matrix
#            each axis in the final MDS plot accounts for
# x.ret : We can also get cmdscale() to return the doubly centered
#         (both rows and columns are centered) version of the distance matrix
#         This is useful if you want to demonstrate how to do MDS using the eigen() function
#         instead of the cmdscale() function
mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)

# Step3 : Calculate the amount of variation each axis in the MDS plot
#         accounts for using the eigen values
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)
mds.var.per

# Step4 : Format the data for ggplot
mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2])

# Just like in the PCA graph, the wild-type samples are on the left side
# The knock-out samples are on the right side
# Just like in the PCA graph, the x-axis accounts for 87.7% of the variation in the data
# The y-axis only accounts for 2.9% of the variation in the data
ggplot(data=mds.data, aes(x=X, y=Y, label=Sample))+
  geom_text()+
  theme_bw()+
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep=""))+
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep=""))+
  ggtitle("MDS plot using Euclidean distance")
  

# =====================================================================================
# Summary
# Actually the PCA graph and MDS graph don't jsut look similar,
# they are exactly the same!
# ★ This is because we used the Euclidean metric to calculate the distance matrix
# =====================================================================================


# =====================================================================================
# Use different metric
# to calculate the distance matrix
# - Average of the absolute value of the log fold cahnge
# - For all you gene expression folks, this is what edgeR does
#   when you call the plotMDS() function
# =====================================================================================

# Calculate the log2 values of the measurements for each gene
log2.data.matrix <- log2(data.matrix)

# Since the average of absolute values of the log-fold change
# isn't one of the distance metrics built into the dist() function,
# we'll create our own distance matrix by hand
# In this step, we're just creating an empty matrix
log2.distance.matrix <- matrix(0,
                               nrow=ncol(log2.data.matrix),
                               ncol=ncol(log2.data.matrix),
                               dimnames=list(colnames(log2.data.matrix),
                                             colnames(log2.data.matrix)))

log2.distance.matrix

# Fill the matrix with the average of the absolute values of the log fold changes
for (i in 1:ncol(log2.distance.matrix)) {
  for(j in 1:i) {
    log2.distance.matrix[i, j] <-
      mean(abs(log2.data.matrix[,i] - log2.data.matrix[,j]))
  }
}

# Because the matrix would be symmetrical,
# we only have to calculate the values of the lower triangle
log2.distance.matrix

# Perform multi-dimensional scaling on our new distance matrix
# as.dist(log2.distance.matrix) : Convert the homemade matrix into a 'true' distance matrix,
#                                 so cmdscale() knows what it's working with
#                                 It only needs bottom triangle to be computed
mds.stuff <- cmdscale(as.dist(log2.distance.matrix),
                      eig=TRUE,
                      x.ret=TRUE)

# Calculate the ammount of variation each axis in the MDS plot accounts for
# using the eigen values
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)
mds.var.per

# Format the data for ggplot()
mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2])

# Create the graph
ggplot(data=mds.data, aes(x=X, y=Y, label=Sample))+
  geom_text()+
  theme_bw()+
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep=""))+
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep=""))+
  ggtitle("MDS plot using avg(logFC) as the distance")


# =====================================================================================
# Summary
# The 2 different MDS plots (one using the Euclidean distance,
# and the other usign the average absolute value of the log fold change) are similar,
# but not the same
# In the new graph, the x-axis accounts for more of the variation (99%)
# =====================================================================================

