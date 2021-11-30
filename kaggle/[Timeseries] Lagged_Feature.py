
'''
주어진 데이터(basic2.csv)에서 새로운 컬럼(1일 이전 시차 컬럼)을 만들고,
Events가 1이면서 Sales가 1000000이하인 조건에 맞는 새로운 컬럼 합을 구하시오

데이터셋 : basic2.csv
'''

import pandas as pd
import numpy as np

df = pd.read_csv('basic2.csv')

# PV 값을 1일씩 (다음 행으로) 미루기
# -> 맨 첫번째 값이 결측값이 됨
df['prev_PV'] = df['PV'].shift(1)
df.head()

# 결측값이 있는 경우 바로 다음 값으로 채우기
df['prev_PV'] = df['prev_PV'].fillna(method='bfill')
df.head()

# Events = 1, Sales <= 1000000
print(df[(df['Events'] == 1) & (df['Sales'] <= 1000000)]['prev_PV'].sum())
# 1894876.0

