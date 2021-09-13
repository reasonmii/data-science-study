'''
XGBoost is an exceptionally useful machine learning method when you don't want to sacrifice the ability
to correctly classify observations but you still want a model that is fairly easy to understand and interpret
'''

'''
Import the modules that will do all the work ==================================
conda update --all
conda install scikit-learn=0.22.1
conda install -c conda-forge xgboost

To draw the tree
conda install graphviz python-graphviz
'''

import pandas as pd   # to load and manipulate data and for One-Hot Encoding
import numpy as np    # to calculate the mean and standard deviation
import xgboost as xgb # XGBoost stuff
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer # for scoring
from sklearn.model_selection import GridSearchCV     # cross validation
from sklearn.metrics import confusion_matrix         # to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix    # to draw a confusion matrix

'''
Import the data ===============================================================
dataset from IBM Bese Sample - Telco Churn dataset
Predict whether or not a customer will stop using a company's service
In business lingo, this is called Customer Churn

When pandas (pd) reads in data, it returns a dataframe, which is a lot like a spreadsheet
The data are organized in rows and columns and each row can contain a mixture of text and numbers
The standard variable name for a dataframe is the initials df, and that is what we will use here
'''

df = pd.read_csv('Telco_customer_churn.csv')
df.head()

# The last four variables contain exit interview information and should not be used for prediction,
# so we will remove them
df.drop(['Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'],
        axis=1, inplace=True) # set axis=0 to remove rows, axis=1 to remove columns
df.head()

# Some of the other columns only contain a single value, and will not be useful for classification
# For example
df['Count'].unique()   # array([1], dtype=int64)
df['Country'].unique() # array(['United States'], dtype=object)
df['State'].unique()   # array(['California'], dtype=object)

df['City'].unique()
# array(['Los Angeles', 'Beverly Hills', 'Huntington Park', ..., 'Standish',
#        'Tulelake', 'Olympic Valley'], dtype=object)

# We will also remove 'CustomerID' because it is different for every customer and useless for classification
# Lastly, we will drop 'Lat Long' because there are separate columns for 'Latitude' and 'Longitude'
df.drop(['CustomerID', 'Count', 'Country', 'State', 'Lat Long'],
        axis=1, inplace=True) # set axis=0 to remove rows, axis=1 to remove columns
df.head()

# Although it is OK to have whitespace in the city names in City for XGBoost and classification
# ★ we can't have nay whitespace if we want to draw a tree
# So let's take care of that now by replacing the white space in the city names with an underscore character _
#
# We can easily remove whitespaces from all values, not just city names,
# but we will wait to do that until after we have identified missing values
df['City'].replace(' ', '_',
                   regex=True,   # regular expression
                   inplace=True)
df.head()

df['City'].unique()[0:10]

# ★ Eliminate the whitespace in the column names, so we'll replace it with underscores
df.columns = df.columns.str.replace(' ', '_')
df.head()


'''
Missing Data Part 1 : Identifying Missing Data ================================

Unfortunately, the biggest part of any data analysis project is making sure that the data is correctly formatted and fixing it when it is not
The first part of this process is identifying and dealing with Missing Data

Missing Data is simply a blank space, or a surrogate value like NA, that indicates that we failed to collect data for one of the features

★★ One thing that is relatively unique about XGBoost is that it has default behavior for missing data
★ So all we have to do is identify missing value and make sure they are set to 0

In this section, we'll focus on identifying missing values in the dataset
'''

# See what sort of data is in each column
df.dtypes

# A lot of columns are object, and this is OK
# because, as we saw above when we ran head() there were a lot of text responses, like Yes and No
# However, let's verify that we are getting what we expect

df['Phone_Service'].unique()
# array(['Yes', 'No'], dtype=object)

# So, Phone_Service has type 'object' because it contains text and it only contains two values, 'Yes' and 'No'
# So this is good
# Now, in practice, we would check every other column, and I did this,
# but right now we will focus on one specific column that looks like it could be a problem: Total_Charges

# If we look at the output from head(), Total_Charges looks like it contains numbers, not text,
# but the 'object' datatype suggests that it contains more than just numbers
# If we try the trick of printing out the unique values
# we see that there are too many values to print and what little we see looks like numbers
df['Total_Charges'].unique()
# array(['108.15', '151.65', '820.5', ..., '7362.9', '346.45', '6844.5'],
#       dtype=object)

# However, if we try to convert the column to numeric values, we get an error
# df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])

# This error, however, tells us what the problem is
# There are blank spaces, " ", in the data
# So we need to deal with those


'''
Missing Data Part 2 : Dealing with Missing Data, XGBoost Style ================

One thing that is relatively unique about XGBoost is that it determines default behavior for missing data
So all we have to do is identify missing values and make sure they are set to 0

However, before we do that, let's see how many rows are missing data
If it's a lot, then we might have a problem on our hands that is bigger than what XGBoost can deal with on its own
If it's not that many, we can just set them to 0
'''

len(df.loc[df['Total_Charges'] == ' '])  # 11

# Since only 11 rows have missing values, let's look at them
df.loc[df['Total_Charges'] == ' ']

# We see that all 11 people with Total_Charges == ' ' have just signed up, because 'Tenure_Months' is 0
# These people also all have Churn_Value set to 0 because they just signed up
# So we have a few choices here, we can set Total_Charges to 0 for these 11 people or we can remove them
# In this example, we'll try setting it to 0
df.loc[(df['Total_Charges'] == ' '), 'Total_Charges'] = 0  # df.loc[ "row" , "column" ]

# Now let's verify that we modified Total Charges correctly by looking at every who had Tenure Months set to 0
df.loc[df['Tenure_Months'] == 0]

# ★ NOTE : Total_Charges still have the object data type
# ★ That is no good because XGBoost only allows int, float or boolean data types
# We can fix this by converting it with to_numeric()
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])
df.dtypes

# ★ Replace all of the other whitespaces in all of the columns with underscores
# ★ We are only doing this so we can draw a picture of the one of the XGBoost trees
df.replace(' ', '_', regex=True, inplace=True)
df.head()


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

X = df.drop('Churn_Value', axis=1).copy()
# alternatively: X = df_no_missing.iloc[:, :-1]
X.head()

y = df['Churn_Value'].copy()
y.head()


'''
Format Data Part 2 : One-Hot Encoding =========================================

★XGBoost natively support continuous data, it does not natively support categorical data
Thus, in order to use categorical data with XGBoost,
★ we have to use a trick that converts a column of categorical data into multiple columns of binary values
★ This trick is called One-Hot Encoding

What's wrong with treating categorical data like continuous data?
★ If we treat the categorical value, 1, 2, 3 and 4, like continuous data, then we would assume that 4 is more similar to 3
That means the XGBoost tree would be more likely to cluster the people with 4s and 3s together
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

# Just to review, let's look at the data types in X to remember how python is seeing the data right now
X.dtypes

# First, before we commit to coverting columns with One-Hot Encoding
# let's see what happens when we convert Payment_Method without saving the results
pd.get_dummies(X, columns=['Payment_Method']).head()

# As we can see, get_dummies() puts all of thecolumns it does not process in the front
# and it puts the values for Payment_Mthod at the end, split into 4 columns, just like we expected it

# ★ NOTE : In a real situation, you should verify all these columns to make sure they only contain the accepted categories
X_encoded = pd.get_dummies(X, columns=['City', 'Gender', 'Senior_Citizen', 'Partner', 'Dependents', 'Phone_Service',
                                       'Multiple_Lines', 'Internet_Service', 'Online_Security', 'Online_Backup',
                                       'Device_Protection', 'Tech_Support', 'Streaming_TV', 'Streaming_Movies',
                                       'Contract', 'Paperless_Billing','Payment_Method'])

X_encoded.head()

# Let's verify that y only contains 0s and 1s with unique()
y.unique()


'''
Build a Preliminary Support Vector Machine ====================================

At long last, the data is correctly formatted for making an XGBoost Model

★ NOTE : Because XGBoost uses Sparse Matrices, it only keeps track of the 1s and it doesn't allocate memory for the 0s
'''

# Let's observe that this data is imbalanced by dividing the number of people who left the company, where y=1
sum(y) / len(y)   # 0.2653698707936959

# So only 27% of the people in the dataset left the company
# Because of this, when we split the data into training and testing, we will split using stratification
# ★ in order to maintain the same percentage of people who left the company in both the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)

# ★ Now let's verify that using 'stratify' worked as expected
sum(y_train) / len(y_train)  # 0.2654297614539947
sum(y_test) / len(y_test)    # 0.26519023282226006

# 'stratify' worked as expected and both y_train and y_test have the same percentage of people that left the company
# Now let's build the preliminary model
# ★ NOTE : Instead of determining the optimal number of trees with cross validation,
#           we will use early stopping to stop building trees when they no longer improve the situation

# binary:logistic
# This is for classification
#
# missing=None (default)
# This is to tell XGBoost what character we're using for the missing value
# 'missing=0', it means we set missing value to 0 (not allocate memory
# 'missing=?', it means we changed the missing value to whatever we want
clf_xgb = xgb.XGBClassifier(objective='binary:logistic',
                            missing=0,  # default value for missing value
                            seed=42)

clf_xgb.fit(X_train,
            y_train,
            verbose=True,                # detailed logging information (tell you what you're doing)
            # ★★ When the model meets the point where they no longer improve the situation, it will build only 10 more trees
            # If none of that 10 trees show any improvement, then it will stop
            early_stopping_rounds=10,
            eval_metric='aucpr',         # Evaluate with 'AUC'
            eval_set=[(X_test, y_test)]) # Evaluate with testing set

# [0]	validation_0-aucpr:0.64036
# [1]	validation_0-aucpr:0.64404
# [2]	validation_0-aucpr:0.65124
# [3]	validation_0-aucpr:0.65051
# [4]	validation_0-aucpr:0.65321
# [5]	validation_0-aucpr:0.64854
# [6]	validation_0-aucpr:0.65459
# [7]	validation_0-aucpr:0.65895
# [8]	validation_0-aucpr:0.65746
# [9]	validation_0-aucpr:0.65850
# [10]	validation_0-aucpr:0.66217
# [11]	validation_0-aucpr:0.66527
# [12]	validation_0-aucpr:0.66322
# [13]	validation_0-aucpr:0.66310
# [14]	validation_0-aucpr:0.66000
# [15]	validation_0-aucpr:0.66027
# [16]	validation_0-aucpr:0.65781
# [17]	validation_0-aucpr:0.65593
# [18]	validation_0-aucpr:0.65738
# [19]	validation_0-aucpr:0.65829
# [20]	validation_0-aucpr:0.65683

# OK, we've built an XGBoost model for classification
# Let's see how it performs on the Testing Dataset by running the Testing Dataset down the model and drawing a Confusion Matrix
plot_confusion_matrix(clf_xgb,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Did not leave", "Left"])

'''
In the confusion matrix, we see that of the 1294 people that did not leave, 1166 (90%) were correctly classified
And of 467 people that left the company, 256 (55%) were correctly classified

So the XGBoost model was not awesome
Part of the problem is that our data is imbalanced, which we saw earlier
and we see this in the confusion matrix with the top row showing 1262 people that did not default
and the bottom row showing 467 people who did

Because people leaving costs the company a lot of money,
we would like to capture more of the people that left

★ The good news is that XGBoost has a parameter, scale_pos_weight, that helps with imbalanced data
So let's try to improve predictions using Cross Validation to optimize the parameters
'''

'''
Optimize Parameters using Cross Validation and GridSearch() ===================

XGBoost has a lot of hyperparameters, parameters that we have to manual configure and are not determined by XGBoost itself,
- max_depth : the maximum tree depth
- learning_rate : the learning rate
- eta : gamma, the parameter that encourages pruning
- reg_lambda : lambda, the regularization parameter

So let's try to find the optimal values for these hypoerparameters in hopes that we can improve the accuracy with the Testing Dataset

★ NOTE : Since we have many hyperparameters to optimize, we will use GridSearchCV()
We specify a bunch of potential values for the hyperparameters and GridSearchCV() tests all possible combinations of the parameters for us

★ When data are imbalanced, the XGBoost manual says...
If you care only about the overall performance metric (AUC) of your prediction
    * Balance the positive and negative weights via "scale_pos_weight"
    * Use AUC for evaluation
'''

# I ran GridSearchCV sequentially on subsets of parameter options, rather than all at once
# in order to optimize parameters in a short period of time
# If not, it takes about 10 minutes to run

# ROUND 1
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0],
    'scale_pos_weight': [1, 3, 5] # XGBoost recommends sum(negative instances) / sum(positive instances)
    }
# Output : max_depth 4, learning_rate 0.1, gamma 0.25, reg_lambda 10.0, scale_pos_weight 3
# ★ Because 'learning_rate' and 'reg_lambda' were at the ends of their range, we will continue to explore

# ROUND 2
param_grid = {
    'max_depth': [4],
    'learning_rate': [0.1, 0.5, 1],
    'gamma': [0.25],
    'reg_lambda': [10.0, 20, 100],
    'scale_pos_weight': [3]
    }
# Output : learning_rate 0.1, reg_lambda 10.0

# To speed up cross validation, and to further prevent overfitting
# ★ we are only using a random subset of the data (90%)
# ★ and are only using a random subset of the features (columns) (50%) per tree
optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=0,   # If you want to see what Grid Search is doing, set verbose = 2
    n_jobs=10,
    cv=3)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False)

print(optimal_params.best_params_)

# So, after testing all possible combinations of the potential parameter values with Cross Validation,
# we see that we should set gamma=0.25, learn_rate=0.1, max_depth=4, and reg_lambda=10


'''
Building, Evaluating, Drawing and Interpreting the Optimized XGBoost Model ====

Now that we have the ideal parameter values, we can build the final XGBoost model
'''

clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic',
                            max_depth=4,
                            learn_rate=0.1,
                            gamma=0.25,
                            reg_lambda=10.0,
                            scale_pos_weight=3,
                            subsample=0.9,
                            colsample_bytree=0.5)

clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

# Now let's draw another confusion matrix to see if the optimized XGBoost model does better
plot_confusion_matrix(clf_xgb,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Did not leave", "Left"])

'''
We see that the optimized XGBoost model is a lot better at identifying people that left the company
Of the 467 people that left the company, 389 (83%), were correcly identified
Before optimization, we only correctly identified 256 (55%)

However, this improvement was at the expense of not being able to correctly classify as many people that did not leave
Before optimization, we correctly identified 1166 (90%) people that did not leave
Now we correctly classify 931 (80%)

That said, this trade off may be better for the company
because now it can focus resources on the people that leave if that will help them retain them
'''

# The last thing we are going to do is draw the first XGBoost Tree and discuss how to interpret it
# ★ If we want to get information, like gain and cover etc, at each node in the first tree, we just build the "first" tree,
# otherwise we'll get the average over all of the trees
clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic',
                            max_depth=4,
                            learn_rate=0.1,
                            gamma=0.25,
                            reg_lambda=10.0,
                            scale_pos_weight=3,
                            subsample=0.9,
                            colsample_bytree=0.5,
                            # ★ 'n_estimators' tells XGBoost, we only want to build 'one' tree
                            # We set this to 1 so we can get gain, cover etc.
                            n_estimators=1)

clf_xgb.fit(X_train, y_train)

'''
Now print out the weight, gain, cover, etc. for the tree

weight = number of times a feature is used in a branch or root across all trees
gain = the average gain across all splits that the feature is used in
cover = the average coverage across all splits a feature is used in

total_gain = the total gain across all splits the feature is used in
total_cover = the total coverage across all splits the feature is used in

NOTE : Since we only built one tree, 'gain = total_gain' and 'cover = total_cover'

★ conda install graphviz python-graphviz
'''

bst = clf_xgb.get_booster()

for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

node_params = {'shape': 'box', # make the nodes fancy
               'style': 'filled, rounded',
               'fillcolor': '#78cbe'}

leaf_params = {'shape': 'box',
               'style': 'filled',
               'fillcolor': '#e48038'}

# ★ NOTE : num_trees is NOT the number of trees to plot, but the specific tree you want to plot
# The default value is 0, but I'm setting it just to show it in action since it is counter-intuitive
# xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10")
xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10",
                condition_node_params=node_params,
                leaf_node_arams=leaf_params)

# If you want to save the figure
# graph_daga = xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10",
#                              condition_node_params=node_params,
#                              leaf_node_arams=leaf_params)
#
# Save as PDF
# graph_data.view(filename='xgboost_tree_customer_churn')

'''
How to interpret the XGBoost Tree

The variable (column name) and the threshold for splitting the observations
For example, inthe tree's root, we use Contract_Month-to-month to split the observations
All observations with Contract_Month-to-month < 1 go to the left
and all observations with Contract_Month-to-month >= go to the right

Each branch either says yes or no and some also say missing
- yes and no refer to whether the threshold in the node above it is true or not
- missing is the default option if there is missing data

leaf tells us the output value for each leaf
'''

'''
In conclusion we
1) Loaded the Data From a File
2) Identified and Dealt with Missing Data
3) Formatted the Data for XGBoost using One-Hot Encoding
4) Built an XGBoost Model for Classification
5) Optimize the XGBoost Parameters with Cross Validation and GridSearch()
6) Built, Drew, Interpreted and Evaluated the Optimized XGBoost Model
'''
