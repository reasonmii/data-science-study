'''
생존여부 예측모델 만들기
학습용 데이터 (Xtrain, ytrain)을 이용하여 생존 예측 모형을 만든 후,
이를 평가용 데이터(X_test)에 적용하여 얻은 예측값을
다음과 같은 형식의 CSV파일로 생성하시오
(제출한 모델의 성능은 accuracy 평가지표에 따라 채점)

(가) 제공 데이터 목록
- y_train: 생존여부(학습용)
- Xtrian, Xtest : 승객 정보 (학습용 및 평가용)

(나) 데이터 형식 및 내용
- y_trian (712명 데이터)

시험환경 세팅은 예시문제와 동일한 형태의
Xtrain, ytrain, X_test 데이터를 만들기 위함임

유의사항
- 성능이 우수한 예측모형을 구축하기 위해서는
  적절한 데이터 전처리, 피처엔지니어링, 분류알고리즘,
  하이퍼파라미터 튜닝, 모형 앙상블 등이 수반되어야 한다.
- 수험번호.csv파일이 만들어지도록 코드를 제출한다.
- 제출한 모델의 성능은 accuracy로 평가함

csv 출력형태
- index는 없고 PassengerId, Survived 만 있는 형태

dataset : titanic_train, titanic_test
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6769
https://www.kaggle.com/rahulsah06/titanic
'''

import pandas as pd
import numpy as np

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
answer = pd.read_csv('../input/titanic/gender_submission.csv')

test2 = test.drop(['PassengerId', 'Name'], axis=1)

# print(X.shape)
# print(X.head())
# print(X.isnull().sum())

train['Cabin'] = train['Cabin'].fillna('null')
train['Embarked'] = train['Embarked'].fillna('null')
train['Age'] = train['Age'].fillna(train['Age'].mean())

test2['Age'] = test2['Age'].fillna(test2['Age'].mean())
test2['Fare'] = test2['Fare'].fillna(0)

X = train.drop(['PassengerId', 'Name', 'Survived'], axis=1)
y = train['Survived']

# print(X.head())
# print(X.info())
# print(X.isnull().sum())
# print(test2.isnull().sum())

# Error : Encoders require their input to be uniformly strings or numbers. Got ['float', 'str']
# -> 중간에 서로 다른 type이 있어서 발생
# -> ★ .astype(str) 으로 해결
from sklearn.preprocessing import LabelEncoder
cols = ['Sex', 'Ticket', 'Cabin', 'Embarked']
for col in cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test2[col] = le.fit_transform(test2[col].astype(str))

from sklearn.preprocessing import MinMaxScaler
cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in cols:
    scaler = MinMaxScaler()
    scaler.fit(X[[col]])
    X[[col]] = scaler.transform(X[[col]])
    test2[[col]] = scaler.transform(test2[[col]])

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
# print(X.head())
# print(X.info())

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
    print(name, model.score(X_tr, y_tr), model.score(X_val, y_val))

# RandomForestClassifier 0.9985955056179775 0.8100558659217877
# SVC 0.6699438202247191 0.664804469273743
# XGBClassifier 0.9971910112359551 0.8212290502793296

from sklearn.model_selection import GridSearchCV
params = {'n_estimators':[50, 100],
         'random_state':[42],
         'max_depth':[3, 5],
         'learning_rate':[0.1, 1]}

xgb_search = GridSearchCV(xgb, param_grid = params, cv=5)
xgb_search.fit(X_tr, y_tr)
print("Best Params : {}".format(xgb_search.best_params_))
print("Best Score : {}".format(xgb_search.best_score_))
# Best Params : {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'random_state': 42}
# Best Score : 0.8371023342854329
    
final = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, learning_rate=0.1)
final.fit(X_tr, y_tr)

pred = final.predict(test2)

output = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':pred})
output.to_csv('titanic_rst.csv', index=False)

print(output.head())

# 실제 결과확인
print(final.score(test2, answer['Survived']))
# 0.7368421052631579


