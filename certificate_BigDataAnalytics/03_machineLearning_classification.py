
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# 1. 분석 데이터 검토
# =================================================================

data = pd.read_csv('breast-cancer-wisconsin.csv', encoding='utf-8')

data.describe()
data.info()

# 유방암 환자 : 239명
data['Class'].value_counts(sort = False)

print(data.shape)
# (683, 11)


# =================================================================
# 2. 특성(X)과 레이블(y) 나누기
# =================================================================

# 방법1) 특성 이름으로 특성 데이터셋(X) 나누기
X1 = data[['Clump_Thickness', 'Cell_Size', 'Cell_Shape',
           'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei',
           'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']]

# 방법2) 특성 위치값으로 특성 데이터셋(X) 나누기
X2 = data[data.columns[1:10]]  # 1~9열

# 방법3) loc 함수로 특성 데이터셋(X) 나누기
# 단, 불러올 특성이 연달아 있어야 함
X3 = data.loc[:, 'Clump_Thickness':'Mitoses']

# (683, 9)
print(X1.shape)
print(X2.shape)
print(X3.shape)

y = data[['Class']]

# (683, 1)
print(y.shape)


# =================================================================
# 3. train-test 데이터셋 나누기
# =================================================================

from sklearn.model_selection import train_test_split

# ★ stratify = y : y레이블의 범주비율에 맞게 분류
X_train, X_test, y_train, y_test = train_test_split(X1, y, stratify=y, random_state=42)

print(y_train.mean())
print(y_test.mean())
# Class    0.349609
# dtype: float64
# Class    0.350877
# dtype: float64


# =================================================================
# 4. 정규화
# =================================================================

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

# train data 정규화 ===============================================

scaler_minmax.fit(X_train)  # 모델 정규화 훈련
X_scaled_minmax_train = scaler_minmax.transform(X_train)

scaler_standard.fit(X_train)
X_scaled_standard_train = scaler_standard.transform(X_train)

# 기술통계량 확인
# 모든 값 min : 0, max : 1 으로 변환됨
pd.DataFrame(X_scaled_minmax_train).describe()
pd.DataFrame(X_scaled_standard_train).describe()

# test data 정규화 ================================================

X_scaled_minmax_test = scaler_minmax.transform(X_test)
pd.DataFrame(X_scaled_minmax_test).describe()

X_scaled_standard_test = scaler_minmax.transform(X_test)
pd.DataFrame(X_scaled_standard_test).describe()

# 특정 컬럼 정규화 ================================================
from sklearn.preprocessing import minmax_scale
data['Normal_Nucleoli'] = minmax_scale(data['Normal_Nucleoli'])


# =================================================================
# 5. 모델 학습
# =================================================================

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# 모델훈련
# model.fit(X, y)
model.fit(X_scaled_minmax_train, y_train)

# 모델예측
# 1) 범주 : model.predict(X)
# 2) 확률 : model.predict_proba(X)
pred_train = model.predict(X_scaled_minmax_train)

# 모델정확도
# model.score(X, y)
model.score(X_scaled_minmax_train, y_train)
# 0.97265625

pred_test = model.predict(X_scaled_minmax_test)
model.score(X_scaled_minmax_test, y_test)
# 0.9590643274853801

# 모델평가지표 1 ===================================================
# Confusion Matrix 혼동행렬 =======================================
from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)

confusion_test = confusion_matrix(y_test, pred_test)
print("검증데이터 오차행렬:\n", confusion_test)

from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.97      0.98      0.98       333
#            1       0.97      0.95      0.96       179
#
#     accuracy                           0.97       512
#    macro avg       0.97      0.97      0.97       512
# weighted avg       0.97      0.97      0.97       512

from sklearn.metrics import classification_report
cfreport_test = classification_report(y_test, pred_test)
print("분류예측 레포트:\n", cfreport_test)
# 분류예측 레포트:
#                precision    recall  f1-score   support
#
#            0       0.98      0.95      0.97       111
#            1       0.92      0.97      0.94        60
#
#     accuracy                           0.96       171
#    macro avg       0.95      0.96      0.96       171
# weighted avg       0.96      0.96      0.96       171

# 모델평가지표 2 ===================================================
# ROC =============================================================
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.decision_function(X_scaled_minmax_test))
roc_auc = metrics.roc_auc_score(y_test, model.decision_function(X_scaled_minmax_test))
roc_auc
# 0.9923423423423423

plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)' % roc_auc)
plt.plot([0,1],[1,1],'y--')
plt.plot([0,1],[0,1],'r--')
plt.legend(loc='lower right')
plt.show()


# =================================================================
# 6. 예측값 병합 및 저장
# =================================================================
prob_train = model.predict_proba(X_scaled_minmax_train)
y_train[['y_pred']] = pred_train
y_train[['y_prob0', 'y_prob1']] = prob_train
y_train
#      Class  y_pred   y_prob0   y_prob1
# 131      0       0  0.981014  0.018986
# 6        0       0  0.768191  0.231809
# 0        0       0  0.966431  0.033569
# 269      0       0  0.988880  0.011120
# 56       1       1  0.203161  0.796839

prob_test = model.predict_proba(X_scaled_minmax_test)
y_test[['y_pred']] = pred_test
y_test[['y_proba0', 'y_proba1']] = prob_test
y_test
#      Class  y_pred  y_proba0  y_proba1
# 541      0       0  0.955893  0.044107
# 549      0       0  0.970887  0.029113
# 318      0       0  0.943572  0.056428
# 183      0       0  0.979370  0.020630
# 478      1       1  0.001305  0.998695

# pd.concat([데이터셋1, 데이터셋2, ..., 데이터셋n], option)
# option : axis=1 변수 병합 (열추가)
#          axis=0 케이스 병합 (행추가)
Total_test = pd.concat([X_test, y_test], axis=1)
Total_test

# CSV 파일로 내보내기
Total_test.to_csv('classification_test.csv')

