# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가


# 데이터 불러오기 =============================================
import pandas as pd

X_test = pd.read_csv("data/X_test.csv")
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# EDA =======================================================

# 데이터확인
print(X_train.shape)
print(X_train.head())
print(y_train['gender'].value_counts())

# null 확인
print(X_train.isnull().sum())
print(X_test.isnull().sum())

# 데이터 전처리 ===============================================

# null 변환
X_train['환불금액'] = X_train['환불금액'].fillna(0)
X_test['환불금액'] = X_test['환불금액'].fillna(0)

# ID, object 컬럼 삭제
X_train = X_train.drop(['cust_id'], axis=1)
cust_id = X_test.pop('cust_id')   # test는 이후에 다시 합쳐야 하니 따로 저장해 두기

# 혹은 object 컬럼 label encoding
from sklearn.preprocessing import LabelEncoder
cols = ['주구매상품', '주구매지점']
for col in cols:
	le = LabelEncoder()
	X_train[col] = le.fit_transform(X_train[col])
	X_test[col] = le.fit_transform(X_test[col])
	
# 모델링 & 하이퍼파라미터 튜닝 & 앙상블 ==========================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

X_tr, X_val, y_tr, y_val = train_test_split(X_train,
					    y_train['gender'],
					    stratify=y_train['gender'],
					    test_size=0.3,
				    	    random_state=42)

model = LogisticRegression()
model.fit(X_tr, y_tr)
print('LogisticRegression : ', round(model.score(X_val, y_val)*100, 2))

model = KNeighborsClassifier()
model.fit(X_tr, y_tr)
print('KNeighborsClassifier : ', round(model.score(X_val, y_val)*100, 2))

model = SVC()
model.fit(X_tr, y_tr)
print('SVC : ', round(model.score(X_val, y_val)*100, 2))

model = DecisionTreeClassifier()
model.fit(X_tr, y_tr)
print('DecisionTreeClassifier : ', round(model.score(X_val, y_val)*100, 2))

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_tr, y_tr)
print('RandomForestClassifier : ', round(model.score(X_val, y_val)*100, 2))

model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
model.fit(X_tr, y_tr)
print('XGBClassifier : ', round(model.score(X_val, y_val)*100, 2))

# 모델 선택 ==================================================

# GridSearch -> 너무 시간이 오래 걸려서 에러 발생 (1분 이상)
# from sklearn.model_selection import GridSearchCV
# param_grid = {'n_estimators': range(100, 1000, 250),
#   						'max_features': ['auto']}
# grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
# grid_search.fit(X_tr, y_tr)

# print("Best Parameter ", grid_search.best_params_)
# print("Best Score ", grid_search.best_score_)
# print("Val set Score ", grid_search.score(X_val, y_val))

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train['gender'])
pred = model.predict(X_test)
pred_prob = model.predict_proba(X_test)

print(pred_prob)
print(print(pd.DataFrame(pred).value_counts()))

# CSV =======================================================
# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'cust_id': X_test.cust_id, 'gender': pred}).to_csv('003000000.csv', index=False)

output = pd.DataFrame({'cust_id': cust_id, 'gender': pred_prob[:,1]})
output.to_csv('20211131.csv', index=False)

