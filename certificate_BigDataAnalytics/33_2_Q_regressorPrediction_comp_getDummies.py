
import numpy as np
import pandas as pd

data = pd.read_csv('insurance.csv')

data.head()
data.info()

X = data[data.columns[0:6]]
y = data[['charges']]

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size = 0.3, random_state=42)

print(X_tr.shape, y_tr.shape)
print(X_val.shape, y_val.shape)


# MinMaxScaler ===============================================================
from sklearn.preprocessing import MinMaxScaler
cols = ['age', 'bmi', 'children']
for col in cols:
    scaler = MinMaxScaler()
    scaler.fit(X_tr[[col]])
    X_tr[[col]] = scaler.transform(X_tr[[col]])
    X_val[[col]] = scaler.transform(X_val[[col]])

# one-hot encoding ===========================================================
X_tr = pd.get_dummies(X_tr, columns = ['sex', 'smoker', 'region'], drop_first=True)
X_val = pd.get_dummies(X_val, columns = ['sex', 'smoker', 'region'], drop_first=True)

X_tr.head()
X_tr.info()


# Model Selection ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

lr = LogisticRegression(random_state=42)
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
abr = AdaBoostRegressor(random_state=42)
gbr = GradientBoostingRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)

models = [lr, dt, rf, abr, gbr, xgb]

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

for model in models:
    name = model.__class__.__name__
    scores = cross_val_score(model, X=X_tr, y=y_tr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    mse = (-1) * np.mean(scores)
    print("Model %s : RMSE %.4f" %(name, np.sqrt(mse)))


# Hyperparameter Tuning ======================================================

print(help(GradientBoostingRegressor))
print(dir(GradientBoostingRegressor))

gbr_params = {
    'learning_rate' : [0.01, 0.05, 0.1],  # rf에서는 이 parameter만 제외
    'n_estimators' : [50, 100, 300],
    'max_depth' : [3,5,7,9],
    'min_samples_split' : [2,3,4,5],
    'min_samples_leaf' : [1, 3, 5],
    'random_state' : [42]
    }

from sklearn.model_selection import GridSearchCV
gbr_search = GridSearchCV(gbr, param_grid = gbr_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gbr_search.fit(X_tr, y_tr)
mse = (-1) * gbr_search.best_score_
rmse = np.sqrt(mse)

print("Best Parameter : {}".format(gbr_search.best_params_))
print("Best score : {}".format(rmse))
# Best Parameter : {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 100, 'random_state': 42}
# Best score : 4648.821860525881

# Validation set =============================================================

gbr_final = GradientBoostingRegressor(**gbr_search.best_params_)
gbr_final.fit(X_tr, y_tr)
gbr_pred = gbr_final.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, gbr_pred))
print("RMSE of GBR : %.4f" % (rmse))
# RMSE of GBR : 4336.9589

