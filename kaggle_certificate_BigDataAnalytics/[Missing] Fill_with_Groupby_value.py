
'''
'f4'컬럼의 값이 'ESFJ'인 데이터를 'ISFJ'로 대체하고,
'city'가 '경기'이면서 'f4'가 'ISFJ'인 데이터 중
'age'컬럼의 최대값을 출력하시오

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6705
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

# 결측치 80% 이상
print(df.isnull().sum()/df.shape[0])

# 해당 컬럼 삭제
df = df.drop(['f3'], axis=1)
df.head()

# city 변수 확인
print(df['city'].unique())
s = df[df['city'] == '서울']['f1'].median()
b = df[df['city'] == '부산']['f1'].median()
d = df[df['city'] == '대구']['f1'].median()
k = df[df['city'] == '경기']['f1'].median()

# f1 결측치 city별 중앙값으로 대체
df['f1'] = df['f1'].fillna(df['city'].map({'서울':s, '부산':b, '대구':d, '경기':k}))

# f1 평균
print(df['f1'].mean())
# 65.52
