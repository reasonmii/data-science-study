
'''
'age' 컬럼의 IQR방식을 이용한 이상치 수와
표준편차*1.5방식을 이용한 이상치 수 합을 구하시오

IQR방식
Q1 - 1.5 * IQR, Q3 + 1.5 * IQR에서 벗어나는 영역을 이상치라고 판단
(Q1은 데이터의 25%, Q3는 데이터의 75% 지점임)

표준편차 1.5방식
평균으로부터 '표준편차 1.5'를 벗어나는 영역을 이상치라고 판단함

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6979
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

q1 = df['age'].quantile(0.25)
q3 = df['age'].quantile(0.75)
iqr = q3 - q1
out1 = df[(df['age'] < q1 - 1.5 * iqr) | (df['age'] > q3 + 1.5 * iqr)]

m = df['age'].mean()
s = df['age'].std() * 1.5
out2 = df[(df['age'] < m - s) | (df['age'] > m + s)]

print(len(out1) + len(out2))
# 14
