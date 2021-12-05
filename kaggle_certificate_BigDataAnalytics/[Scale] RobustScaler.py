
'''
RobustScaler
이상치가 있을 때 강력한 scaler

https://wikidocs.net/89704
'''

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

# 수치형 변수
cols = ['Age', 'AnnualIncome', 'FamilyMembers', 'ChronicDiseases']

# train 데이터에만 fit_transform 사용
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])

