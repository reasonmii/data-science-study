'''
Bike Sharing Demand

dataset : bike_sharing_train.csv
https://www.kaggle.com/c/bike-sharing-demand/overview
https://www.kaggle.com/c/bike-sharing-demand/

Score : 0.44
'''

import pandas as pd
import numpy as np

df = pd.read_csv('bike_sharing_train.csv')

df.head()
df.info()
df.describe()

X = df[df.columns[0:9]]
y = df[['count']]


# Datetime ===================================================================
# Type : object -> datetime64
X['datetime'] = pd.to_datetime(X['datetime'])
X['year'] = X['datetime'].dt.year
X['month'] = X['datetime'].dt.month
X['day'] = X['datetime'].dt.day
X['hour'] = X['datetime'].dt.hour
X = X.drop('datetime', axis=1)


# Train Test Split ===========================================================

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_tr.shape, y_tr.shape, X_val.shape, y_val.shape)


# Modeling ===================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

lr = LogisticRegression(random_state=42)
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
ada = AdaBoostRegressor(random_state=42)
gbr = GradientBoostingRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)

models = [lr, dt, rf, ada, gbr, xgb]


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

for model in models:
    name = model.__class__.__name__
    model.fit(X_tr, y_tr)
    scores = cross_val_score(model, X=X_tr, y=y_tr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    mse = (-1) * np.mean(scores)
    print("Model %s : RMSE %.4f" % (name, np.sqrt(mse)))
    print("Train : %.4f - Val : %.4f" % (model.score(X_tr, y_tr), model.score(X_val, y_val)))
# Model LogisticRegression : RMSE 220.4131
# Train : 0.0223 - Val : 0.0188
# Model DecisionTreeRegressor : RMSE 63.4305
# Train : 1.0000 - Val : 0.8957
# Model RandomForestRegressor : RMSE 44.5454
# Train : 0.9920 - Val : 0.9455
# Model AdaBoostRegressor : RMSE 104.5038
# Train : 0.6525 - Val : 0.6541
# Model GradientBoostingRegressor : RMSE 67.4345
# Train : 0.8595 - Val : 0.8527
# Model XGBRegressor : RMSE 43.2483
# Train : 0.9819 - Val : 0.9498


# Hyperparameter Tuning ======================================================

params = {
    'n_estimators' : [50, 100, 200],
    'max_depth' : [3, 5, 7],
    'colsample_bytree' : [0.7, 0.9],
    'random_state' : [42]
    }

from sklearn.model_selection import GridSearchCV
xgb_search = GridSearchCV(xgb, param_grid = params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_search.fit(X_tr, y_tr)
mse = (-1) * xgb_search.best_score_
print("Best Params : {}".format(xgb_search.best_params_))
print("Best RMSE : {}".format(np.sqrt(mse)))
# Best Params : {'colsample_bytree': 0.9, 'max_depth': 7, 'n_estimators': 100, 'random_state': 42}
# Best RMSE : 43.108408614341556


# Final ======================================================================

# ★ objective='count:poisson'
# - poisson regression for count data, output mean of Poisson distribution
# - This would prevent the 'negative' output values
model = XGBRegressor(n_estimators=200, max_depth=3, colsample_bytree=0.7, random_state=42, objective='count:poisson')
model.fit(X_tr, y_tr)
print("Train : %.4f - Val : %.4f" % (model.score(X_tr, y_tr), model.score(X_val, y_val)))
# Train : 0.9285 - Val : 0.9149


# Apply to test set ==========================================================

test = pd.read_csv('bike_sharing_test.csv')
test_org = test.copy()

test['datetime'] = pd.to_datetime(test['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test = test.drop('datetime', axis=1)

test_org['count'] = model.predict(test)
test_org = test_org[['datetime', 'count']]
test_org.head()

test_org.to_csv('bike_sharing_test_rst.csv', index=False)

