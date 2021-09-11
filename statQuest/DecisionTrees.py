
'''
Classification Trees are an exceptionally useful machine learning method
when you need to know how the decision are being made

For example, if you have to justify the predictions to your boss,
Classification Trees are a good method because each step in the decision making process is easy to understand
'''

'''
Import the modules that will do all the work ==================================

conda update --all
conda install scikit-learn=0.22.1
'''

import pandas as pd   # to load and manipulate data and for One-Hot Encoding
import numpy as np    # to calculate the mean and standard deviation
import matplotlib.pyplot as plt  # to draw graphs
from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree              # to draw a classification tree
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.model_selection import cross_val_score  # for cross validation
from sklearn.metrics import confusion_matrix         # to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix    # to draw a confusion matrix


'''
Import the data ===============================================================

UCI Machine Learning Repository - Heart Disease Dataset
Predict if someone has heart disease based on their sex, age, blood pressure and a variety of other metrics

When pandas (pd) reads in data, it returns a dataframe, which is a lot like a spreadsheet
The data are organized in rows and columns and each row can contain a mixture of text and numbers
The standard variable name for a dataframe is the initials df, and that is what we will use here
'''

# Download the data directly from UCI
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
                 header=None)

df.head()

# Change the column numbers to column names
# We see that instead of nice column names, we just have column numbers
# Since nice column names would make it easier to know how to format the data
df.columns = ['age','sex','cp','restbp','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','hd']
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
'''

# See what sort of data in each column
# dtypes tell us the data type for each column
df.dtypes

# We see that they are almost all float64
# However, two columns, 'ca' and 'thal', have the object type and one column, 'hd' has int64
# The fact that the 'ca' and 'thal' columns have object data types suggests there is something funny going on in them
# ★ object datatypes are used when there are mixtures of things, like a mixture of numbers and letters
# In theory, both ca and thal should just have a few values representing different categories
# So let's investigate what's going on by printing out their unique values

df['ca'].unique()
# array(['0.0', '3.0', '2.0', '1.0', '?'], dtype=object)
# We see that 'ca' contains numbers and question marks
# The question marks represent missing data

df['thal'].unique()
# array(['6.0', '3.0', '7.0', '?'], dtype=object)
# 'thal' also contains a mixture of numbers and question marks, which represent missing values


'''
Missing Data Part 2 : Dealing with Missing Data ===============================

★ Since scikit-learn's classification trees do not support datasets with missing values,
we need to figure out what to do these question marks

We can either delete these patients from the training dataset, or impute values for the missing data
'''

# print the number of rows that contain missing values
#
# loc[], short for "location", let us specify which rows we want
# and so we say we want any row with '?' in column 'ca'
# OR any row with '?' in column 'thal'
#
# len(), short for "length", prints out the number of rows
len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')])

# Since only 6 rows have missing values, let's look at them
df.loc[(df['ca'] == '?') | (df['thal'] == '?')]

# Now let's count the number of rows in the full dataset
len(df)

# So 6 of 303 rows, or 2%, contain missing values
# Since 297 is plenty of data to build a classification tree, we will remove the rows with missing values, rather than impute their values
# NOTE : Imputing missing values is a big topic that we will tackle in another webinar

# use loc[] to select all rows that do not contain missing values
# and save them in a new dataframe called "df_no_missing"
df_no_missing = df.loc[(df['ca'] != '?') & (df['thal'] != '?')]

# Since df_no_missing has 6 fewer rows than the original df, it should have 297 rows
len(df_no_missing)

# Make sure 'ca' and 'thal' no longer contains question marks by printing its unique values
df_no_missing['ca'].unique()    # array(['0.0', '3.0', '2.0', '1.0'], dtype=object)
df_no_missing['thal'].unique()  # array(['6.0', '3.0', '7.0'], dtype=object)

# ★ NOTE : 'ca' and 'thal' still have the objet data type. That's OK.


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

# Make a new copy of the columns used to make predictions
X = df_no_missing.drop('hd', axis=1).copy()  # alternatively: X = df_no_missing.iloc[:,:-1]
X.head()

# Make a new copy of the column of data we want to predict
y = df_no_missing['hd'].copy()
y.head()


'''
Format Data Part 2 : One-Hot Encoding =========================================

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

What the data should be
1) age
   - Float
2) sex
   - Category
   0 = female, 1 = male
3) cp, chest pain
   - Category
   1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic
4) restbp, resting blood pressure (in mm Hg)
   - Float
5) chol, serum cholesterol in mg/dl
   - Float
6) fbs, fasting blood sugar 
   - Category
   0 = >= 120mg/dl, 1 = < 120mg/dl
7) restecg, resting electrocardiographic results
   - Category
   1 = normal, 2 = having ST-T wave abnormality, 3 = showing probable or define left ventricular hypertrophy
8) thalach, maximum heart rate achieved
   - Float
9) exang, exercise induced angina
   - Category
   0 = no, 1 = yes
10) oldpeak, ST depression induced by exercise relative to rest
    - Float
11) slope, the slope of the peak exercise ST segment
    - Category
    1 = unsloping, 2 = flat, 3 = downsloping
12) ca, number of major vessels (0-3) colored by fluoroscopy
    - Float
13) thal, thalium heart scan
    - Category
    3 = normal (no cold spots),
    6 = fixed defect (cold spots during rest and exercise),
    7 = reversible defect (when cold spots only appear during exercise)
'''

# Just to review, let's look at the data types in X to remember how python is seeing the data right now
X.dtypes

# ★★ scikit learn Decision Trees relatively support continuous data
# They do not natively support categorical data, like chest pain (cp)
# Thus, in order to use categorical data with scikit learn Decision Trees,
# ★ we have to use a trick that converts a column of categorical data into multiple columns of binary values
# ★ This trick is called One-Hot Encoding

# What's wrong with treating categorical data like continuous data?
# ★ If we treat the categorical value, 1, 2, 3 and 4, like continuous data, then we would assume that 4 is more similar to 3
# That means the decision tree would be more likely to cluster the patients with 4s and 3s together
# than the patients with 4s and 1s together
# In contrast, if we treat these numbers like categorical data, then we treat each one as a separate category
# that is no more or less similar to any of the other categories
# Thus, the likelihood of clustering patients with 4s with 3s is the same as clustering 4s with 1s,
# and that approach is more reasonable

# ★ First, you need to verify the column only contains the accepted categories
# If needed, convert the columns that contain categorical and integer data into the correct datatypes

X['cp'].unique()   # array([1., 4., 3., 2.])
# 'cp' only contains the values it is supposed to contain
# So we will convert it, using One-Hot Encoding into a series of columns that only contains 0s and 1s

# For this tutorial, we will use get_dummies() to do One-Hot Encoding
pd.get_dummies(X, columns=['cp']).head()

# ★ It puts all of the columns it does not process in the front and it puts 'cp' at the end
# It also split cp into 4 columns, cp_1.0, cp_2.0, cp_3.0, cp_4.0

# Do it for other columns
X['restecg'].unique() # array([2., 0., 1.])
X['slope'].unique()   # array([3., 2., 1.])
X['thal'].unique()    # array(['6.0', '3.0', '7.0'], dtype=object)

X_encoded = pd.get_dummies(X, columns=['cp',
                                       'restecg',
                                       'slope',
                                       'thal'])

X_encoded.head()

# Since 'sex', 'fbs', and 'exang' only have 2 categories, and they only contain 0s and 1s
# we don't have to do anything special to them
# Use unique() to verify this
X['sex'].unique()   # array([0., 1.])
X['fbs'].unique()   # array([0., 1.])
X['exang'].unique() # array([0., 1.])

# In this tutorial, we're only making a tree that does simple classification
# and only care if someone has heart disease or not
# so we need to convert all numbers > 0 to 1
y_not_zero_index = y > 0  # ★ get the index for each non-zero value for y
y[y_not_zero_index] = 1   # set each non-zero value in y to 1
y.unique()                # verify that y only contains 0 and 1


'''
Build a Preliminary Classification Tree =======================================

At long last, the data are correctly formatted for making a Classification Tree
Now we simply split the data into training and testing sets and build the tree
'''

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

# Create a decision tree and fit it to the training data
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

# NOTE : We can plot the tree and it is huge!
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt,
          filled=True,
          rounded=True,
          class_names=["No HD", "Yes HD"],
          feature_names=X_encoded.columns);

# OK, we've built a Classification Tree for classification
# Let's see how it performs on the Testing Dataset by running the Testing Dataset down the tree
# and drawing a Confusion Matrix
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=["Does not have HD", "Has HD"])

# Of the 31 + 11 = 42 people that do not have HD, 31 (74%) were correctly classified
# Of the 7 + 26 = 33 people that have HD, 26 (79%) were correctly classified

# It may have overfit the training dataset
# So let's prune the tree
# ★ Pruning, in theory, should solve the overfitting problem and give us better results


'''
Cost Complexity Pruning Part 1 : Visualize alpha ==============================

Decision Trees are notorious for being overfit to the Training Dataset
and there are lots of parameters, like max_depth and min_samples, that are desinged to reduce overfitting

★ However, pruning a tree with 'cost complexity pruning' can simplify the whole process
of finding a smaller tree that improves the accuracy with the Testing Dataset

Pruning a decision tree is all about finding the right value for the pruing parameter, alpha,
which controls how little or how much pruning happens

★ One way to find the optimal value for alpha is to plot the accuracy of the tree as a function of different values
We'll do this for both the Training Dataset and the Testing Dataset

First let's extract the different values of alpha that are available for this tree
and build a pruned tree for each value for alpha

NOTE : We omit the maximum value for alpha with 'ccp_alphas = ccp_alphas[:-1]'
       because it would prune all leaves, leaving us with only a root instead of a tree
'''

# Determine values for alpha
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)

# Extract different values for alpha
ccp_alphas = path.ccp_alphas

# Exclude the maximum value for alpha
ccp_alphas = ccp_alphas[:-1]

# Create an array that we will put decision trees into
clf_dts = []

# ★ Create one decision tree per value for alpha and store it in the array
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)

# Graph the accuracy of the trees using Training Dataset and the Testing Dataset as a function of alpha
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()

# The accuracy for the Testing Dataset hits its maximum value when alpha is about 0.016
# After this value for alpha, the accuracy of the Traiing Dataset drops off and that suggests we should set ccp_alpha=0.016

# NOTE
# ★ When we apply Cost Complexity Pruning to a Classification Tree, values for alpha go from 0 to 1
#    because GINI scores go from 0 to 1,
# ★ In contrast, values for alpha for a Regression Tree can be much larger
#    since the sum of squared residuals can, in theory, go from 0 to positive infinity


'''
Cost Complexity Pruning Part 2 : Cross Validation For Finding the Best Alpha ==

Let's demonstrate that different training and testing datasets result in trees with different accuracies when we set cp_alpha=0.016
We will do this by using the cross_val_score() function to generate different training and testing datasets
and then train and test the tree with those datasets
'''

# Create the tree with ccp_alpha=0.016
clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)

# ★ Now use 5-fold cross validation create 5 different training and testing datasets that are then used to train and test the tree
# NOTE: We use 5-fold because we don't have tons of data
scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
df = pd.DataFrame(data={'tree': range(5), 'accuracy':scores})

df.plot(x='tree', y='accuracy', marker='o', linestyle='--')

# The graph shows taht using different Training and Testing data with the same alpha resulted in different accuracies,
# ★ suggesting that alpha is sensitive to the datasets
# So, instead of picking a single Training dataset and single Testing dataset,
# ★★ let's use cross validation to find the optimal value for ccp_alpha

# Create an array to store the result of each fold during cross validation
alpha_loop_values = []

# For each candidate value for alpha, we will run 5-fold cross validation
# Then we will store the mean and standard deviation of the scores (the accuracy) for each call
# to cross_val_score in alpha_loop_values
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

# Now we can draw a graph of the means and standard deviations of the scores
# for each candidate value for alpha
alpha_results = pd.DataFrame(alpha_loop_values,
                             columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',       # errorbar sizes
                   marker='o',
                   linestyle='--')

# Using cross validation, we can see that, over all, instead of setting ccp_alpha=0.016,
# we need to set it to something closer to 0.014
# We can find the exact value with:
alpha_results[(alpha_results['alpha'] > 0.014)
              &
              (alpha_results['alpha'] < 0.015)]

# Now let's store the ideal value for alpha so that we can use it to build the best tree
ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)&(alpha_results['alpha'] < 0.015)]['alpha']
ideal_ccp_alpha
# result : 20    0.014225

# NOTE : At this point Python thinks that ideal_ccp_alpha is a 'series', which is a type of array
# We can tell because when we printed it out, we got two bits of stuff
# The first one was 20, which is the index in the series, the second one, 0.014225, is the value we want
# ★ So we can convert this from a 'series' to a 'float' with the following command:
ideal_ccp_alpha = float(ideal_ccp_alpha)
ideal_ccp_alpha   # 0.014224751066856332


'''
Building, Evaluating, Drawing, and Interpreting the Final Classification Tree =

Now that we have the ideal value for alpha
we can build the final Classification Tree by setting ccp_alpha=ideal_ccp_alpha
'''

# Build and train a new decision tree, only this time use the optimal value for alpha
clf_dt_pruned = DecisionTreeClassifier(random_state=42,
                                       ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

# Now let's draw another confusion matrix to see if the pruned tree does better
plot_confusion_matrix(clf_dt_pruned, X_test, y_test,
                      display_labels=["Does not have HD", "Has HD"])

# We see that the pruned tree is better at classifying patients than the full sized tree
# Of the 34 + 8 = 42 people without HD, 34 (81%) were correctly classified
# This is an improvement over the full sized tree, which only correctly classified 31 (74%) of the patients without HD
# Of the 5 + 28 = 33 people with HD, 28 (85%) were correctly classified
# This is an improvement over the full sized tree, which only correctly classified 26 (79%) of the patients with HD

# ★ The last thing we are going to do is draw the pruned tree and discuss how to interpret it
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["No HD", "Yes HD"],
          feature_names=X_encoded.columns);

'''
Now let's discuss how to interpret the tree
In each one we have:
    
    The variable (column name) and the threshold for splitting the observations
    For example, in the tree's root, we use 'ca' to split the observations
    All observations with ca <= 0.5 go to the left and ca > 0.5 go to the right
    
    gini is the gini index or score for that node
    
    samples tell us how many samples are in that node
    
    value tells us how many samples in the node are in each category
    In this example, we have two categories, No and Yes, referring to whether or not a patient has HD
    ★ The number of patients with No comes first because the categories are in alphabetical order
    Thus, in the root, 118 patients have No and 104 patients have Yes
    
    class tells us whichever category is represented most in the node
    In the root, since 118 people have No and only 104 people have Yes, class is set to No

The leaves are just like the nodes, except that they do not contain a variable and threshold for splitting the observations

★ Lastly, the nodes and leaves are colored by the class
In this case No is different shades of orange-ish and Yes is different shades of blue
The darker the shade, the lower the gini score, and that tells us how much the node or leaf is skewed towards one class
'''


'''
In conclusion =================================================================
1) Imported Data
2) Identified and Dealt with Missing Data
3) Formatted the Data for Decision Trees using One-Hot Encoding
4) Built a Preliminary Decision Tree for Classification
5) Pruned the Decision Tree with Cost Complexity Pruning
6) Built, Drew, Interpreted and Evaluated the Final Decision Tree
'''
