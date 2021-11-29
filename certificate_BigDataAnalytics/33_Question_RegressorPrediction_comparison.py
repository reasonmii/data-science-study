
import pandas as pd
import numpy as np

data = pd.read_csv('insurance.csv')

print(data.head())
# print(data.info())
# print(data.describe())

print(data.isnull().sum())

X = data.drop(['charges'], axis=1)
y = data[['charges']]

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


# MinMaxScaler ===============================================================
from sklearn.preprocessing import MinMaxScaler
cols = ['age', 'bmi', 'children']
for col in cols:
    scaler = MinMaxScaler()
    scaler.fit(X_tr[[col]])
    X_tr[col] = scaler.transform(X_tr[[col]]).flatten()
    X_val[col] = scaler.transform(X_val[[col]]).flatten()


# one-hot encoding w/ 2 category =============================================
from sklearn.preprocessing import OneHotEncoder

# Sex : 여성 = 0, 남성 = 1 (정수형)
onehot_sex = OneHotEncoder()
sex_train = X_tr[['sex']]
sex_test = X_val[['sex']]

onehot_sex.fit(sex_train)
sex_train_onehot = onehot_sex.transform(sex_train).toarray()
sex_test_onehot = onehot_sex.transform(sex_test).toarray()

X_tr['sex'] = sex_train_onehot[:, 1].astype(np.uint8)
X_val['sex'] = sex_test_onehot[:, 1].astype(np.uint8)
X_tr.head()

# Smoker : no = 0, yes = 1 (정수형)
onehot_smoker = OneHotEncoder()
smoker_train = X_tr[['smoker']]
smoker_test = X_val[['smoker']]

onehot_smoker.fit(smoker_train)
smoker_train_onehot = onehot_smoker.transform(smoker_train).toarray()
smoker_test_onehot = onehot_smoker.transform(smoker_test).toarray()

X_tr['smoker'] = smoker_train_onehot[:,1].astype(np.uint8)
X_val['smoker'] = smoker_test_onehot[:,1].astype(np.uint8)
X_tr.head()


# one-hot encoding w/ multi categories =======================================

# LavelEncoder
# southeast, northeast, northwest, southwest -> 1, 2, 3, 4
# 1차원 배열이 input으로 들어가야 함
from sklearn.preprocessing import LabelEncoder
label_region = LabelEncoder()
label_region.fit(X_tr['region'])

X_tr['region'] = label_region.transform(X_tr['region'])
X_val['region'] = label_region.transform(X_val['region'])
X_tr.head()

onehot_region = OneHotEncoder()
region_train = X_tr[['region']]
region_test = X_val[['region']]

onehot_region.fit(region_train)
region_train_onehot = onehot_region.transform(region_train).toarray()
region_test_onehot = onehot_region.transform(region_test).toarray()

X_tr['region_1'] = region_train_onehot[:, 1].astype(np.uint8)  # northwest
X_tr['region_2'] = region_train_onehot[:, 2].astype(np.uint8)  # southeast
X_tr['region_3'] = region_train_onehot[:, 3].astype(np.uint8)  # southwest

X_val['region_1'] = region_test_onehot[:, 1].astype(np.uint8)  # northwest
X_val['region_2'] = region_test_onehot[:, 2].astype(np.uint8)  # southeast
X_val['region_3'] = region_test_onehot[:, 3].astype(np.uint8)  # southwest

X_tr.drop(['region'], axis=1, inplace=True)
X_val.drop(['region'], axis=1, inplace=True)

X_tr.head()

print(X_tr.shape, y_tr.shape)
print(X_val.shape, y_val.shape)


# Model Selection ============================================================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
ada = AdaBoostRegressor(random_state=42)
gbr = GradientBoostingRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)

models = [lr, dt, rf, ada, gbr, xgb]

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# ★ neg_mean_squared_error 사용 이유
#    sklearn의 교차검증은 scoring 값이 클수록 좋다고 생각하는데,
#    사실 mse 값이 작을수록 좋은 모델이기 때문
# ★ n_jobs : Number of jobs to run in parallel (default = None)
#             -1은 전부를 의미
for model in models:
    name = model.__class__.__name__    
    scores = cross_val_score(model, X=X_tr, y=y_tr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)    
    mse = (-1) * np.mean(scores)
    print("Model %s - RMSE : %.4f" %(name, np.sqrt(mse)))
# Model LinearRegression - RMSE : 6208.6892
# Model DecisionTreeRegressor - RMSE : 7003.4460
# Model RandomForestRegressor - RMSE : 5153.1456 -> 선택
# Model AdaBoostRegressor - RMSE : 5426.7206
# Model GradientBoostingRegressor - RMSE : 4836.0392 -> 선택
# Model XGBRegressor - RMSE : 5429.8235


# Hyperparameter Tuning ======================================================
from sklearn.model_selection import GridSearchCV

rf_params = {
    'n_estimators' : [100, 200, 300, 400],
    'max_depth' : [3, 5, 7, 9],
    'min_samples_split': [2,3,4,5],
    'min_samples_leaf' : [1,2,3,5],
    'random_state' : [42]
    }

rf_search = GridSearchCV(rf, param_grid=rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_search.fit(X_tr, y_tr)
best_mse = (-1) * rf_search.best_score_
best_rmse = np.sqrt(best_mse)

print("Best score : {}, Best params : {}".format(round(best_rmse, 4), rf_search.best_params_))


gbr_params = {
    'learning_rate' : [0.01, 0.05, 0.1],
    'n_estimators' : [50, 80, 100, 200, 300],
    'max_depth' : [3, 5, 7, 9],
    'min_samples_split' : [2,3,4,5],
    'min_samples_leaf' : [1,2,3,4],
    'random_state' : [42]
    }

gbr_search = GridSearchCV(gbr, param_grid=gbr_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gbr_search.fit(X_tr, y_tr)
best_mse = (-1) * gbr_search.best_score_
best_rmse = np.sqrt(best_mse)

print("Best score : {}, Best params : {}".format(round(best_rmse, 4), gbr_search.best_params_))


# Test set ===================================================================

rf_final = RandomForestRegressor(**rf_search.best_params)
rf_final.fit(X_tr, y_tr)
rf_pred = rf_final.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
print("RMSE of RandomForest : %.4f" % (rf_rmse))

gbr_final = GradientBoostingRegressor(**gbr_search.best_params)
gbr_final.fit(X_tr, y_tr)
gbr_pred = gbr_final.predict(X_val)
gbr_rmse = np.sqrt(mean_squared_error(y_val, gbr_pred))
print("RMSE of GBR : %.4f" % (gbr_rmse))

