'''
빅데이터분석기사 실기 3회 기출문제

작업형 1-1
데이터셋에서 결측치 데이터 제거 후 앞에서부터 70%로만 데이터 구성
특정 컬럼의 1사분위 값 구하기

dataset : basic1.csv
'''

import pandas as pd

# 시험 환경 setting
df = pd.read_csv('basic1.csv')
df = df.drop('f3', axis = 1)
df.head()
print(df.shape)    # (100, 8)

# 문제풀이
df = df.dropna()   # (69, 7)
df = df.iloc[:int(len(df)*0.7), :]   # (48, 7)

print(int(df['age'].quantile(0.25)))
