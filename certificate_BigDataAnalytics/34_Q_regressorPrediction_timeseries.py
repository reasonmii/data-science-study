
import pandas as pd
import numpy as np

train = pd.read_csv('bike_sharing_train.csv')
test = pd.read_csv('bike_sharing_test.csv')


# EDA ========================================================================

# Null 체크
train.isnull().sum()
test.isnull().sum()

# hour 컬럼 추가
train['hour'] = train['datetime'].str[11:13]
test['hour'] = train['datetime'].str[11:13]

# X, y 구성
X_train = train.drop(['datetime', 'casual', 'registered', 'count'], axis=1)
y_train = train['count']

X_test = test.drop(['datetime'], axis=1)
y_test = test['datetime']


# OneHot Encoding ============================================================

X_train[['season', 'weather']] = X_train[['season', 'weather']].astype('str')
X_test[['season', 'weather']] = X_train[['season', 'weather']].astype('str')

X_train = pd.get_dummies(X_train, columns=['hour', 'season', 'weather'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['hour', 'season', 'weather'], drop_first=True)

print(X_train.head())
print(X_train.info())


# Train Test Split ===========================================================

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


# MinMaxScaler ===============================================================

from sklearn.preprocessing import MinMaxScaler
cols = ['temp', 'atemp', 'humidity', 'windspeed']
for col in cols:
    scaler = MinMaxScaler()
    scaler.fit(X_tr[[col]])
    X_tr[[col]] = scaler.transform(X_tr[[col]])
    X_val[[col]] = scaler.transform(X_val[[col]])
    X_test[[col]] = scaler.transform(X_test[[col]])


# Model Selection ============================================================
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
# Model LogisticRegression : RMSE 153.5342
# Train : 0.0581 - Val : 0.0193
# Model DecisionTreeRegressor : RMSE 111.8584
# Train : 0.9990 - Val : 0.6674
# Model RandomForestRegressor : RMSE 81.4802
# Train : 0.9719 - Val : 0.8088
# Model AdaBoostRegressor : RMSE 144.9797
# Train : 0.3746 - Val : 0.3584
# Model GradientBoostingRegressor : RMSE 91.7642
# Train : 0.7598 - Val : 0.7358
# Model XGBRegressor : RMSE 80.6644
# Train : 0.9263 - Val : 0.8198


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
# Best Params : {'colsample_bytree': 0.7, 'max_depth': 3, 'n_estimators': 200, 'random_state': 42}
# Best RMSE : 77.34240929977935


# Final ======================================================================

# ★ objective='count:poisson'
# - poisson regression for count data, output mean of Poisson distribution
# - This would prevent the 'negative' output values
model = XGBRegressor(n_estimators=200, max_depth=3, colsample_bytree=0.7, random_state=42, objective='count:poisson')
model.fit(X_tr, y_tr)
print("Train : %.4f - Val : %.4f" % (model.score(X_tr, y_tr), model.score(X_val, y_val)))
# Train : 0.8352 - Val : 0.8082


# Apply to test set ==========================================================

X_test.info()

test['count'] = model.predict(X_test)
test.head()

test = test[['datetime', 'count']]
test.head()

test.to_csv('bike_sharing_test_rst.csv', index=False)

