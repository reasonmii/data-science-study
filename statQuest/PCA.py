
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# There are a few Python PCA functions to choose from
# The one in sklearn is the most commonly used
from sklearn.decomposition import PCA

# Scale the data before performing PCA
from sklearn import preprocessing

# Generate an array of 100 gene names
# gene name : gene1, gene2, etc.
genes = ['gene' + str(i) for i in range(1,101)]

# Create arrays of sample names
# 5 wild type or "wt" samples
# 5 knock out or "ko" samples
wt = ['wt' + str(i) for i in range(1,6)]
ko = ['ko' + str(i) for i in range(1,6)]

# Create a dataframe
# ★ The stars "*" unack the "wt" and "ko" arrays
#    so that the column names are a single array that looks like this:
#    [wt1, wt2, wt3, wt4, wt5, ko1, ko2, ko3, ko4, ko5]
# ★ Without the stars, we would create an array of two arrays
#    and that wouldn't create 12 columns like we want:
#    [[wt1, wt2, wt3, wt4, wt5], [ko1, ko2, ko3, ko4, ko5]]
# The gene names are used for the index, which means that they are the equivalent of row names
data = pd.DataFrame(columns=[*wt,*ko], index=genes)

# Create the random data
# For each gene in the index (gene1, gene2, ...)
# create 5 values for the "wt" samples and 5 values for the "ko" samples
# The data comes from two poisson distributions
# For each gene, we select a new mean for the poisson distribution
# The means can vary between 10 and 1000
for gene in data.index:
    data.loc[gene, 'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
    data.loc[gene, 'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)

print(data.head())
print(data.shape)

# Center and scale the data
# After centering, the average value for each gene will be 0
# After scaling, the standard deviation for the values for each gene will be 1
# ★ Notice that we are passing in the transpose of our data
#    The scale function expects the samples to be rows instead of columns
# NOTE : We use samples as columns in this example
#        because that is often how genomic data is stored
#        If you have other data, you can store it however is easiest for you
#        There's no requirement that samples be rows or columns
#        ★ Just be aware that if it is columns, you'll need to transpose it before analysis
# NOTE : The code below is just one way to use sklearn to center and scale the data
#        so that the means for each gene are 0 and the standard deviation for each gene are 1
#        ★ Alternatively, we could have used: StandardScaler().fit_transform(data.T)
#           This method is more commonly used for machine learning
#           and that's what sklearn was designed to do
# NOTE : In sklearn, variation is calculated as
#           (measurements - mean)^2 / the number of measurements
#        In R using scale() or prcomp(), variation is calculated as
#           (measurements - mean)^2 / (the number of measurements - 1)
#           -> ★ This method results in larger, but unbiased, estimates of the variation
#        The good news is that these differences do not effect the PCA analysis
#        - The loading scores and the amount of variation per principal component will be the same, either way
#        The bad news is that these differences will have a minor effect the final graph
#        - This because the coordinates on the final graph come from multiplying the loading scores by the scaled values
scaled_data = preprocessing.scale(data.T)

# Create a PCA object
# Rather than just have a function that does PCA and returns results
# sklearn uses objects that can be trained using one dataset and applied to another dataset
pca = PCA()

# Since we're only using PCA to explore one dataset
# (and not using PCA in a machine learning setting),
# the additional steps are a little tedious,
# but they set us up for the machine learning topics

# This is where we do all of the PCA math
# i.e. Calculate loading scores and the variation each principal component accounts for
pca.fit(scaled_data)

# Generate coordinates for a PCA graph based on the loading scores and the scaled data
pca_data = pca.transform(scaled_data)

# Draw a graph
# We'll start with a scree plot to see how many principal components should go into the final plot

# First, calculate the percentage of variation that each principal component accounts for
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)

# Create labels for the scree plot
# These are "PC1", "PC2", etc
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

# Create bar plot
# Almost all of the variation is along the first PC,
# so a 2-D graph, using PC1 and PC2, should do a good job representing the original data
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# To draw a PCA plot, we'll first put the new coordinates,
# created by pca.transform(scaled.data), into a nice matrix
# whre the rows have sample labels and the columns have PC labels
pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

# This loop adds sample names to the graph
for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

# Display the graph
plt.show()

# Result
# The "wt" samples clustered on the left side, suggesting that they are correlated with each other
# The "ko" samples clustered on the right side, suggesting that they are correlated with each other
# And the separation of the two clusters along the x-axis suggests that
# "wt" samples are very different from "ko" samples

# Lastly, let's look at the loading scores for PC1 to determine which genes had the largest influence
# on separating the two clusters along the x-axis
# We'll start by creating a pandas "Series" object with the loading scores in PC1
# NOTE : The PCs are zero-indexed, so PC1 = 0
loading_scores = pd.Series(pca.components_[0], index=genes)

# Sort the loading scores based on their magnitude (absolute value)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

# Get the names of the top 10 indexes
top_10_genes = sorted_loading_scores[0:10].index.values

# Print out the top 10 gene names and their corresponding loading scores
# These values are super similar, so a lot of genes played a role in separating the samples,
# rather than just one or two
print(loading_scores[top_10_genes])
