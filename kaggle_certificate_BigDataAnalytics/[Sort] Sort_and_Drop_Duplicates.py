'''
f1의 결측치를 채운 후
age 컬럼의 중복 제거 전과 후의 중앙값 차이를 구하시오
- 결측치는 f1의 데이터 중 10번째 큰 값으로 채움
- 중복 데이터 발생 시 뒤에 나오는 데이터를 삭제함
- 최종 결과값은 절대값으로 출력

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6934
'''

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('basic1.csv')
df.head()

df.isnull().sum()

# 정렬
top10 = df['f1'].sort_values(ascending=False).iloc[9]
print(top10)

# 결측치 채우기
df['f1'] = df['f1'].fillna(top10)

# 중복제거 전
print(df.shape)
med_bf = df['f1'].median()

# 중복제거 후
df_2 = df.drop_duplicates(subset=['age'])
print(df_2.shape)
med_af = df_2['f1'].median()

# 차이
print(abs(med_bf - med_af))


