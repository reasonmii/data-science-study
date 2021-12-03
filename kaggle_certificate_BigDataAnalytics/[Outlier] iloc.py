
'''
문제1
- 데이터셋(basic1.csv)의 'f5' 컬럼을 기준으로 상위 10개의 데이터를 구하고,
- 'f5' 컬럼 10개 중 최소값으로 데이터를 대체한 후,
- 'age' 컬럼에서 80 이상인 데이터의'f5 컬럼 평균값 구하기

문제2
- 데이터셋(basic1.csv)의 앞에서 순서대로 70% 데이터만 활용해서,
- 'f1'컬럼 결측치를 중앙값으로 채우기 전후의 표준편차를 구하고
- 두 표준편차 차이 계산하기 (표본표준편차 기준)

문제3
- 데이터셋(basic1.csv)의 'age'컬럼의 이상치를 더하시오!
- 단, 평균으로부터 '표준편차*1.5'를 벗어나는 영역을 이상치라고 판단함

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6743
'''

import pandas as pd
import numpy as np

# 문제1 =======================================================================
df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

df = df.sort_values('f5', ascending=False)
df['f5'][:10] = df['f5'][:10].min()
print(df[df['age'] >= 80]['f5'].mean())
# 62.497747125217394


# 문제2 =======================================================================
df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

df = df.iloc[: int(df.shape[0]*0.7)]   # 상위 70%

std_bf = df['f1'].std()
df['f1'] = df['f1'].fillna(df['f1'].median())
std_af = df['f1'].std()
print(abs(std_bf - std_af))
# 3.2965018033960725


# 문제3 =======================================================================
df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

age_max = df['age'].mean() + df['age'].std() * 1.5
age_min = df['age'].mean() - df['age'].std() * 1.5
print(df[(df['age'] > age_max) | (df['age'] < age_min)]['age'].sum())
# 473.5



