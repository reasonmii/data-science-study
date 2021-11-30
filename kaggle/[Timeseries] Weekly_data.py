
'''
주어진 데이터(basic2.csv)에서 주 단위 Sales의 합계를 구하고,
가장 큰 값을 가진 주와 작은 값을 가진 주의 차이를 구하시오(절대값)

https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6930
'''

import pandas as pd
import numpy as np

df = pd.read_csv('basic2.csv', parse_dates=['Date'], index_col=0)

# 주 단위 W
# 2주 단위 2W
# 월 단위 M
df_w = df.resample('W').sum()
df_w

sales_max = df_w['Sales'].max()
sales_min = df_w['Sales'].min()

abs(sales_max - sales_min)
# 91639050


