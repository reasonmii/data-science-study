'''
주어진 데이터에서 2022년 5월 주말과 평일의 sales 컬럼
평균값 차이를 구하시오 (소수점 둘째자리까지 출력, 반올림)

dataset : basic2.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6837
'''

import pandas as pd
import numpy as np

df = pd.read_csv('basic2.csv')

df['Date'] = pd.to_datetime(df['Date'])

# 5월 주말
sales_wknd = np.mean(df[(df['Date'].dt.year==2022) & (df['Date'].dt.month == 5) & (df['Date'].dt.weekday >= 5)]['Sales'])

# 5월 평일
sales_wkdy = np.mean(df[(df['Date'].dt.year==2022) & (df['Date'].dt.month == 5) & (df['Date'].dt.weekday < 5)]['Sales'])

# 차이
round(abs(sales_wknd - sales_wkdy),2)


