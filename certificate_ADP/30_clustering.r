
#===============================================================================================================
# Clustering
#===============================================================================================================

### 1. Hierarchical Clustering

idx <- sample(1:dim(iris)[1], 40)
iris.s <- iris[idx,]
iris.s$Species <- NULL

# method : ward.D, average, median, centroid, etc.
hc <- hclust(dist(iris.s), method="ave")

# hang
# - The fraction of the plot height by which labels should hang below the rest of the plot
# - A negative value will cause the labels to hang down from 0 (-1로 설정 시 y=0부터 시작)
plot(hc, hang=-1, labels=iris$Species[idx])

### 2. K-means Clustering
data(iris)
newiris <- iris
newiris$Species <- NULL
kc <- kmeans(newiris, 3)

table(iris$Species, kc$cluster)
#             1  2  3
# setosa     50  0  0
# versicolor  0  2 48
# virginica   0 36 14

plot(newiris[c("Sepal.Length", "Sepal.Width")], col=kc$cluster)
