
'''
주어진 데이터에서 2022년 5월 sales컬럼의 중앙값을 구하시오

dataset : basic2.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6836
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic2.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df.head()

print(df[(df['year'] == 2022) & (df['month'] == 5)]['Sales'].median())
# 1477685.0

