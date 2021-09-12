
'''
In this lesson we will build a Support Vector Machine for classification using scikit-learn
and the Radial Basis Function (RBF) Kernel

Our training data set contains continuous and categorical data from the UCI Machine Learning Repository
to predict whether or not a person will default on their credit card

★ Support Vector Machines are one of the best machine learning methods
   when getting the correct answer is a higher priority than understanding why you get the correct answer
★ They work really well with relatively small datasets and they tend to work well "out of the box"
In other words, they do not require much optimization
'''

'''
Import the modules that will do all the work ==================================
'''

import pandas as pd   # to load and manipulate data and for One-Hot Encoding
import numpy as np    # data manipulation
import matplotlib.pyplot as plt  # for drawing graphs
import matplotlib.colors as colors
from sklearn.utils import resample  # downsample the dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale  # scale the center data
from sklearn.svm import SVC              # make a svm for classification
from sklearn.model_selection import GridSearchCV   # downsample the dataset
from sklearn.metrics import confusion_matrix       # create a confusion matrix
from sklearn.metrics import plot_confusion_matrix  # draw a confusion matrix
from sklearn.decomposition import PCA              # downsample the dataset


'''
Import the data ===============================================================

We will use the Credit Card Default dataset
This dataset will allow us to predict if someone will default on their credit card payment
based on their sex, age, and a variety of other metrics

When pandas (pd) reads in data, it returns a dataframe, which is a lot like a spreadsheet
The data are organized in rows and columns and each row can contain a mixture of text and numbers
The standard variable name for a dataframe is the initials df, and that is what we will use here
'''

# Encoding Error
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
#                  header=1,
#                  sep='\t')

df = pd.read_csv('default_of_credit_card_clients.txt',
                  # header=1,
                  sep='\t')

df.head()


'''
The columns are

- ID, The ID number assigned to each customer
- LIMIT_BAL, Credit limit
- SEX, Gender
- EDUCATION, Level of education
- MARRIAGE, Marital status
- AGE, Age
- PAY_, When the last 6 bills were payed
- BLL_AMT, What the last 6 bills were
- PAY_AMT, How much the last payements were
- default payment next month
'''

# The last column name, default payment next month, is long
# Let's change it to default
df.rename({'default payment next month' : 'DEFAULT'}, axis='columns', inplace=True)
df.head()

# The values in the ID column were randomly assigned, making it uniformative, so we can drop it
df.drop('ID', axis=1, inplace=True)  # ★ set axis=0 to remove row, axis1 to remove columns
df.head()


'''
Missing Data Part 1 : Identifying Missing Data ================================

Unfortunately, the biggest part of any data analysis project is making sure that the data is correctly formatted and fixing it when it is not
The first part of this process is identifying and dealing with Missing Data

Missing Data is simply a blank space, or a surrogate value like NA, that indicates that we failed to collect data for one of the features

There are two main ways to deal with missing data

1) We can remove the rows that contain missing data from the dataset
This is relatively easy to do, but it wastes all of the other values that we collected
How a big of a waste this is depends on how important this missing value is for classification

2) We can impute the values that are missing
In this context impute is just a fancy way of saying "we can make an educated guess about what the value should be"
ex) average, median, or use some other, more sophisticated approach to guess an appropriate value

In this section, we'll focus on identifying missing values in the dataset
'''

# First, let's see what sort of data in each column
df.dtypes

# We see that every column is int 64, this is good, since it tells us that they did not mix letters and numbers
# In other words, there are no NA values, or other character based place holders for missing data, in df

'''
That said, we should still make sure each column contains acceptable values
The list below describes what values are allowed in each column and was based on the column descriptions on the Credit Card Default webpage

1) LIMIT_BAL, The amount of available credit
   - Integer

2) SEX
   - Category
   1 = male, 2 = female

3) EDUCATION
   - Category
   1 = graduate school, 2 = university, 3 = high school, 4 = others

4) MARRIAGE
   - Category
   1 = Married, 2 = Single, 3 = Other

5) AGE
   - Integer

6) PAY_, When the last 6 bills were payed
   - Category
   -1 = Paid on time
   1 = Payment delayed by 1 month
   2 = payment delayed by 2 months
   ...
   8 = Payment delayed by 8 months
   9 = payment delayed by 9 months

7) BILL_AMT, What the last 6 bills were
   - Integer

8) PAY_AMT, How much the last payments were
   - Integer

9) DEFAULT, Whether or not a person defaulted on the next payment
   - Category
   0 = Did not default, 1 = Defaulted   
'''

# Let's start by making sure SEX only contains the numbers 1 and 2
df['SEX'].unique()
# array([2, 1], dtype=int64)

# Let's look at EDUCATION and make sure it only contains 1, 2, 3, and 4
df['EDUCATION'].unique()
# array([2, 1, 3, 5, 4, 6, 0], dtype=int64)
# It is possible that 0 represents missing data
# and 5 and 6 represent categories not mentioned in the specification, but that is just a guess

# Let's look at MARRIAGE and make sure it only contains 1, 2, 3
df['MARRIAGE'].unique()
# array([1, 2, 3, 0], dtype=int64)
# Like EDUCATION, MARRIAGE contains 0, which I'm guessing represents missing data

# In theory, I could pay a lot of money to get the article about this dataset and find out if 0 represents missing data or not
# But since this is a demo, we won't worry too much about being correct and see what happens when we treat 0 as missing data
# NOTE : I treid both ways and the model performs better when we treat 0 as missing data


'''
Missing Data Part 2 : Dealing with Missing Data ===============================

★ Since scikit-learn's classification trees do not support datasets with missing values,
we need to figure out what to do these question marks

We can either delete these patients from the training dataset, or impute values for the missing data
'''

len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)])

# So, only 68 rows have missing values
# Now, let's count the total number of rows in the dataset
len(df)

# So 68 of the 30,000 rows, or less than 1%, contain missing values
# Since that still leaves us with more data than we need for a Support Vector Machine,
# we will remove the rows with missing values, rather than try to impute their values
df_no_missing = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]

# Since df_no_missing has 68 fewer rows than the origianl df, it should have 29,932 rows
len(df_no_missing)

# Make sure EDUCATION and MARRIAGE no longer contains 0s by printing its unique values
df_no_missing['EDUCATION'].unique()  # array([2, 1, 3, 5, 4, 6], dtype=int64)
df_no_missing['MARRIAGE'].unique()   # array([1, 2, 3], dtype=int64)


'''
Downsample the data ===========================================================

Support Vector Machines are great with small datasets, but not awesome with large ones,
and this dataset, while not huge, is big enough to take a long time to optimize with Cross Validation

So we'll downsample both categories, customers who did and did not default, to 1,000 each
'''

# First, let's remind ourselves how many customers are in the dataset
len(df_no_missing)

# 29,932 samples is a relatively large number for a Support Vector Machine, so let's downsample
# ★ To make sure we get 1,000 of each category, we start by "splitting" the data into two dataframes,
# one for people that did not default and one for people that did
df_no_default = df_no_missing[df_no_missing['DEFAULT'] == 0]
df_default = df_no_missing[df_no_missing['DEFAULT'] == 1]

# Now downsample the dataset
df_no_default_downsampled = resample(df_no_default, replace=False, n_samples=1000, random_state=42)
len(df_no_default_downsampled)

df_default_downsampled = resample(df_default, replace=False, n_samples=1000, random_state=42)
len(df_default_downsampled)

# ★ Now let's merge the two downsampled datasets into a single dataframe
# and print out the total number of samples to make sure everything is hunky dory
df_downsample = pd.concat([df_no_default_downsampled, df_default_downsampled])
len(df_downsample)


'''
Format Data Part 1 : Split the Data into Dependent and Independent Variables ==

Now we're ready to format the data for making a Classification Tree

The first step is to split the data into two parts
1) The columns of data that we will use to make classifications : X (capital)
2) The column of data that we want to predict : y (lower case)

We will use the conventional notation of X (capital) and y (lower case)
In this case, we want to predict 'hd' (heart disease)

NOTE : The reason we deail with missing data before splitting it into X and y is that if we need to remove rows,
       splitting after ensures that each row in X correctly corresponds with the appropriate value in y
       
★ NOTE : In the code below we are using copy() to copy the data by value
By default, pandas uses copy by reference
Using copy() ensures that the original data df_no_missing is not modified when we modify X or y
In other words, if we make a mistake when we are formatting the columns for classification trees,
we can just re-copy df_no_missing, rather than reload the original data and remove the missing values etc.
'''

X = df_downsample.drop('DEFAULT', axis=1).copy()
# alternatively: X = df_downsample.iloc[:,:-1].copy)
X.head()

y = df_downsample['DEFAULT'].copy()
y.head()


'''
Format Data Part 2 : One-Hot Encoding =========================================

In order to use categorical data with scikit learn Support Vector Machines,
we have to use a trick that converts a column of categorical data into multiple columns of binary values
★ This trick is called One-Hot Encoding

What's wrong with treating categorical data like continuous data?

★ If we treat the categorical value, 1, 2, 3 and 4, like continuous data, then we would assume that 4 is more similar to 3
That means the support vector machine would be more likely to cluster the people with 4s and 3s together
than the people with 4s and 1s together

In contrast, if we treat these numbers like categorical data, then we treat each one as a separate category
that is no more or less similar to any of the other categories

Thus, the likelihood of clustering people with 4s with 3s is the same as clustering 4s with 1s,
and that approach is more reasonable


There are many different ways to do One-Hot Encoding in Python
★ Two of the more popular methods are ColumnTransformer() (from scikit-learn) and get_dummies() (from pandas)
and the both methods have pros and cons

ColumnTransformer() (from scikit-learn)
- It has a very cool feature where it creates a persistent function that can validate data that you get in the future.
  It remember the options (i.e. red, blue, orange)
  and later on when your Decision Tree is being used in a production system,
  if someone says their favorite color is 'orange', then it can throw an error or handle the situation in some other nice way.
- However, it turns your data into an array and looses all of the column names,
  making it harder to verify that your usage of ColumnTransformer() worked as you intended it to.

get_dummies() (from pandas)
- It leaves your data in a dataframe and retains the column names, making it much easier to verify that it worked as intended
- However, it does not have the persistent behavior that ColumnTransformer() has

★ So, for the sake of learning how One-Hot Encoding works, I prefer to use get_dummies()
   However, once you are comfortable with One-Hot Encoding, I encourage you to investigate using ColumnTransformer()
'''

# First, before we commit to converting columns with One-Hot Encoding,
# let's just see what happens when we convert MARRIAGE without saving the results
# This will make it easy to see how get_dummies() works
pd.get_dummies(X, columns=['MARRIAGE']).head()

# As we can see in the printout above, get_dummies() puts all of the columns it does not process on the left side
# and it puts MARRIAGE on the right side
# It also splits MARRIAGE into 3 columns

# Let's use it on the categorical columns and save the result
# NOTE : In a real situation, you should verify all 5 of these columns to make sure they only contain the accepted categories
X_encoded = pd.get_dummies(X, columns=['SEX', 'EDUCATION','MARRIAGE',
                                       'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'])

X_encoded.head()


'''
Format the Data Part 3 : Centering and Scaling ================================

★ The Raidal Basis Function (RBF) that we are using with our Support Vector Machine "assumes that the data are centered and scaled"
In other words, each column should have "a mean value = 0" and "a standard deviation = 1"
So we need to do this to both the training and testing datasets

★ NOTE : We split the data into training and testing datasets and then scale them separately to avoid "Data Leakage"
Data Leakage occurs when information about the training dataset corrupts or influences the training dataset
'''

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)


'''
Build a Preliminary Support Vector Machine ====================================

At long last, the data is correctly formatted for making a Support Vector Machine
So let's do it!
'''

clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scaled, y_train)

# We've built a Support Vector Machine
# Let's see how it performs on the Testing Dataset and draw a Confusion Matrix
plot_confusion_matrix(clf_svm,
                      X_test_scaled,
                      y_test,
                      values_format='d',
                      display_labels=["Did not default", "Defaulted"])

# Of the 257 people that did not default, 202 (79%) were correctly classified
# And of the 243 poeple that defaulted, 148 (61%) were correctly classified
# So the Support Vector Machine was not awesome
# ★ So let's try to improve predictions using Cross Validation to optimize the parameters


'''
Optimize Parameters with Cross Validation and GridSearchCV() ==================

Optimizing a Support Vector Machine is all about finding the best value for gamma,
and potentially, the regularization parameter, C

So let's see if we can find better parameters values using cross validation in hope that
we can improve the accuracy with the Testing Dataset

Since we have two parameters to optimize, we will use GridSearchCV()
★★ We specify a bunch of potential values for gamma and C,
    and GridSearchCV() tests all possible combinations of the parameters for us
'''

param_grid = [
    {'C' : [0.5, 1, 10, 100],  # NOTE: Values for C must be > 0
     'gamma' : ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
     'kernel' : ['rbf']},
    ]

# NOTE : We are including C=1 and gamma='scale'
# as possible choices since they are the default values

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',  # (accuracy is default scoring) Slightly improved, but hardly! C=1, gamma='scale'
    # scoring='balanced_accuracy',  # Slightly improved, but hardly! C=1, gamma=0.01
    # scoring='f1',           # Terrible! C=0.5, gamma=1
    # scoring='f1_micro',     # Slightly improved, but hardly! C=1, gamma=0.01
    # scoring='f1_macro',     # Same! C=1, gamma='scale' (these are the same as default values)
    # scoring='f1_weighted',  # Same! C=1, gamma='scale' (these are the same as default values)
    # scoring='roc_auc',      # Terrible! C=1, gamma=0.001
    # For more scoring metrics see:
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose=0   # NOTE : If you want to see what Grid Search is doing, set verbose=2    
    )

optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)
# {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
# And we see that the ideal value for C is 100, which means we will use regularization, and the ideal value for gamma is 0.001


'''
Building, Evaluating, Drawing and Interpreting ================================
the Final Support Vector Machine

Now that we have the ideal values for C and gamma, we can build the final Support Vector Machine
'''

clf_svm = SVC(random_state=42, C=100, gamma=0.001)
clf_svm.fit(X_train_scaled, y_train)

# Now let's draw another confusion matrix to see if the optimized support vector machine does better
plot_confusion_matrix(clf_svm,
                      X_test_scaled,
                      y_test,
                      values_format='d',
                      display_labels=["Did not default", "Defaulted"])

# The results from the optimized Support Vector Machine are just a little bit better than before
# 4 more people were correctly classified as not defaulting and only 1 more person was correctly classified as defaulting

# ★ In other words, the SVM was pretty good straight out of the box without much optimization
# This makes SVMs a great, quick and dirty method for relatively small datasets

# NOTE : Although classification with this dataset and an SVM is not awesome, it may be better than other methods
# We'd have to compare to find out

# ★★ The last thing we are going to do is draw a support vector machine decision boundary
# and discuss how to interpret it

# The first thing we need to do is count the number of columns in X
len(df_downsample.columns)  # 24


'''
★ So we see that there are 24 features, or columns, in X
★ This is a problem because it would require a 24-dimensional graph,
   one dimension per feature used to make predictions to plot the data in its raw form

If we wanted to, we could just pick two features at random to use as x and y-axis on our graph,
but instead, we will use PCA (Principal Component Analysis) to combine the 24 features
into 2 orthogonal meta-features that we can use as axes for a graph
※ PCA is a way to shrink a 24-dimensional graph into a 2-dimensional graph

However, before we shrink the graph, let's first determine how accurate the shrunken graph will be
If it's relatively accurate, then it makes sense to draw the 2-Dimensional graph
If not, the shrunken graph will not be very useful
★ We can determine the accuracy of the graph by drawing something called a "scree plot"
'''

# NOTE : By default, PCA() centers the data, but does not scale it
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var) + 1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var)
plt.tick_params(
    axis='x',           # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    top=False,          # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.show()

'''
The scree plot shows that the first principal component, PC1, accounts for a relatively large amount of variation in the raw data,
and this means that it will be a good candidate for the x-axis in the 2-dimensional graph

However, PC2 is not much different from PC3 and PC4, which doesn't bode well for dimension reduction
Since we don't have a choice, let's go with it, but don't be surprised if the final graph looks funky

Now we will draw the PCA graph
NOTE : This code is quite technical, but we don't have to type it in and there are comments that explain each step
'''

# First, let's optimize an SVM fit to PC1 and PC2
train_pc1_coords = X_train_pca[:, 0]
train_pc2_coords = X_train_pca[:, 1]

# NOTE
# pc1 contains the x-axis coordinates of the data after PCA
# pc2 contains the y-axis coordinates of the data after PCA

# Now center and scale the PCs
pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

# Now we optimize the SVM fit to the x and y-axis coordinates of the data after
# after PCA dimension reduction
param_grid = [
    {'C': [1, 10, 100, 1000],
     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
     'kernel': ['rbf']},
    ]

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    # NOTE : The default value for scoring results in worse performance
    # For more scoring metrics see:
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose=0
    # NOTE : If you want to see what Grid Search is doing, set verbose=2
    )

optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)
# {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}

# Now that we have the optimal values for C and gamma, let's draw the graph
clf_svm = SVC(random_state=42, C=1000, gamma=0.001)
clf_svm.fit(pca_train_scaled, y_train)

# Transform the test dataset with the PCA
# X_test_pca = pca.transform(X_test_scaled)
X_test_pca = pca.transform(X_train_scaled)
test_pc1_coords = X_test_pca[:, 0]
test_pc2_coords = X_test_pca[:, 1]

# Now create a matrix of points that we can use to show the decision regions
# The matrix will be a little bit larger than the transformed PCA points
# so that we can plot all of the PCA points on it without them being on the edge
x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1
y_min = test_pc2_coords.min() - 1
y_max = test_pc2_coords.max() + 1

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                     np.arange(start=y_min, stop=y_max, step=0.1))

# Now we will classify every point in that matrix with the SVM
# Points on one side of the classification boundary will get 0, and points on the other side will get 1
Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))

# Right now, Z is just a long array of lots of 0s and 1s,
# which reflects how each point in the mesh was classified
# We use reshape() so that each classification (0 or 1) corresponds to a specific point in the matrix
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 10))
# Now we will use contourf() to draw a filed contour plot, using the matrix values and classifications
# The contours will be filled according to the predicted classifications (0s and 1s) in Z
ax.contourf(xx, yy, Z, alpha=0.1)

# Now create custom colors for the actual data points
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])
# Now draw the actual data points
# - these will be colored by their known (not predicted) classifications
# NOTE : setting alpha=0.7 lets us see if we are covering up a point
# scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_test,
scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_train,
                     cmap=cmap,
                     s=100,
                     edgecolors='k',  # 'k' = black
                     alpha=0.7)

# Now create a legend
legend = ax.legend(scatter.legend_elements()[0],
                   scatter.legend_elements()[1],
                   loc="upper right")
legend.get_texts()[0].set_text("No Default")
legend.get_texts()[1].set_text("Yes Default")

# Now add axis labels and titles
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Decision surface using the PCA transformed/projected features')
# plt.savefig('svm_default.png')
plt.show()


'''
The pink part of the graph is the area where all datapoints will be predicted to have "not defaulted"
The yellow part of the graph is the area where all datapoints will be predicted to "have defaulted"

The dots are datapoints in the trainnig dataset and are color coded by their known classifications
- red is for those that did not default and green is for those that defaulted

NOTE : The results are show the training data, not the testing data and thus, do not match the confusion matrices that we generated

★ Also, remember that for this picture, we only fit the SVM to the first two principal components instead of all the data
   and thus, this is only an approximation of the true classifier

★ Lastly, because the scree plot showed that PC2 was not very different from PC3 or PC4,
   this is not a very good approximation
'''


'''
In conclusion we...
- Loaded the Data from a File
- Identified and Dealt with Missing Data
- Downsampling Data
- Formatted the Data for a Support Vector Machine using One-Hot Encoding
- Built a Support Vector Machine for Classification
- Optimized the Support Vector Machine with Cross Validation
- Built, Drew, Interpreted and Evaluated the Final Support Vector Machine
'''
