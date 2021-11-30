
import numpy as np
import pandas as pd

train = pd.read_csv('house_price_train.csv')

#print(train.head())
print(train.info())
#print(train.describe())
#print(train.isnull().sum())
#print(train.columns.tolist())

X = train.drop(columns=['Id', 'SalePrice'])
y = train[['Id', 'SalePrice']]

print(X.info())
#print(X_trainain.describe())

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# 데이터프레임에 모든 열 이름 표시
pd.set_option('display.max_columns', None)
display(X_train.head(3))
display(y_train.head(3))

y_train['SalePrice'].hist()

# Null 값 확인
print(X_train.isnull().sum().sort_values(ascending=False)[:20])
print(X_val.isnull().sum().sort_values(ascending=False)[:20])

# Preprocessing =================================================================

X_train = X_train.select_dtypes(exclude=['object'])
X_val = X_val.select_dtypes(exclude=['object'])

# colsample_bytree : Subsample ratio of columns when constructing each tree
from xgboost import XGBRegressor
model = XGBRegressor(n_estimator=100, max_depth=5, colsample_bytree=0.9)
model.fit(X_train, y_train['SalePrice'])
pred = model.predict(X_val)
 
# squared=False : squared 값 출력
from sklearn.metrics import mean_squared_error
print("RMSE : ", mean_squared_error(y_val['SalePrice'], pred, squared=False))

output = pd.DataFrame({'Id': y_val['Id'], 'SalePrice': pred})
output.head()

output.to_csv('house_price_rst.csv', index='False')
