
'''
주어진 데이터에서 2022년 월별 Sales 합계 중 가장 큰 금액과
2023년 월별 Sales 합계 중 가장 큰 금액의 차이를 절대값으로 구하시오
단, Events컬럼이 '1'인 경우 80%의 Salse값만 반영함
(최종값은 소수점 반올림 후 정수 출력)

dataset : basic2.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6838
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month

def event_sales(x):
    if x['Events'] == 1:
        x['Sales'] = x['Sales'] * 0.8
    else:
        x['Sales'] = x['Sales']
    return x

df = df.apply(lambda x: event_sales(x), axis=1)


max_2022 = df[df['year'] == 2022].groupby('month')['Sales'].sum().max()
max_2023 = df[df['year'] == 2023].groupby('month')['Sales'].sum().max()

print(int(round(abs(max_2022 - max_2023),0)))
# 42473436
