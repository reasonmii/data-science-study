
'''
고객과 잘 맞는 타입 추천
basic1 데이터 중 'f4'를 기준으로 basic3 데이터 'f4'값을 병합하고,
병합한 데이터에서 r2결측치를 제거한 다음,
앞에서 부터 20개 데이터를 선택하고 'f2'컬럼 합을 구하시오

dataset : basic1.csv, basic3.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6868
'''

import pandas as pd
import numpy as np

df1 = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
df2 = pd.read_csv('../input/bigdatacertificationkr/basic3.csv')

# print(help(pd.merge))
df = pd.merge(left=df1, right=df2, how='left', on='f4')

print(df.head())
print(df.info())

print(df.isnull().sum())
df = df.dropna(subset=['r2'])

print(df[:20]['f2'].sum())
# 15
