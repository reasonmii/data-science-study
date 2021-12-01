
'''
백화점 고객의 1년 간 구매 데이터를 활용
- 데이터 전처리
- Feature Engineering
- 모델링 (분류 알고리즘 사용)
- 하이퍼파라미터 튜닝 (초매개변수 최적화)
- 모형 앙상블
- csv제출

유의사항
- 수험번호.csv 파일이 만들어지도록 코드를 제출함
- 제출한 모델의 성능은 ROC-AUC 평가지표에 따라 채점함

dataset : https://www.dataq.or.kr/
          (공지사항 - 759번 제2회 빅데이터분석기사 실기 안내 - 첨부파일)
          https://www.dataq.or.kr/www/board/view.do
'''

import pandas as pd
import numpy as np

df_X = pd.read_csv('X_train.csv', encoding='euc-kr')
df_y = pd.read_csv('y_train.csv', encoding='euc-kr')
test = pd.read_csv('X_test.csv', encoding='euc-kr')
test_2 = test.copy()


# 데이터 전처리 ================================================================

# print(df_X.head())
X = df_X.drop('cust_id', axis=1)
y = df_y.drop('cust_id', axis=1)
test2 = test.drop('cust_id', axis=1)

X['환불금액'] = X['환불금액'].fillna(0)
test2['환불금액'] = test2['환불금액'].fillna(0)
# print(X.isnull().sum())

# Label Encoding (범주형 변수 레이블인코딩)
from sklearn.preprocessing import LabelEncoder
cols = ['주구매상품', '주구매지점']
for col in cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test2[col] = le.fit_transform(test2[col])
# print(X.head())

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
cols = ['총구매액', '최대구매액', '환불금액', '내점일수', '내점당구매건수', '구매주기']
for col in cols:
    scaler = MinMaxScaler()
    scaler.fit(X_tr[[col]])
    X_tr[[col]] = scaler.transform(X_tr[[col]])
    X_val[[col]] = scaler.transform(X_val[[col]])
    test2[[col]] = scaler.transform(test2[[col]])
    

# Modeling ===================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

rf = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)
xgb = XGBClassifier(random_state=42)

models = [rf, svc, xgb]

for model in models:
    name = model.__class__.__name__
    model.fit(X_tr, y_tr)
    print(name, " : ", model.score(X_tr, y_tr), model.score(X_val, y_val))
# RandomForestClassifier  :  0.9992857142857143 0.6271428571428571
# SVC  :  0.6239285714285714 0.6242857142857143
# XGBClassifier  :  0.9825 0.6142857142857143


# Tuning =====================================================================

print(help(RandomForestClassifier))

params = {'n_estimators':[30, 50, 100],
         'max_features':['auto', 'sqrt', 'log2'],
         'random_state':[42]}

from sklearn.model_selection import GridSearchCV

rf_search = GridSearchCV(rf, param_grid=params, cv=5)
rf_search.fit(X_tr, y_tr)
print("Best params : {}".format(rf_search.best_params_))
print("Best score : {}".format(rf_search.best_score_))
# Best params : {'max_features': 'auto', 'n_estimators': 30, 'random_state': 42}
# Best score : 0.6378571428571428

final = RandomForestClassifier(n_estimators=30, max_features='auto', random_state=42)
final.fit(X_tr, y_tr)
print(final.score(X_tr, y_tr), final.score(X_val, y_val))
# 0.9985714285714286 0.6171428571428571


# CSV 제출 ====================================================================

pred = final.predict(test2)
output = pd.DataFrame({'custid': test['cust_id'], 'gender': pred})

output.to_csv('X_test_rst.csv', index=False)

