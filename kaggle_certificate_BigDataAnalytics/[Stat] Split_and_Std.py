'''
첫번째 데이터 부터 순서대로 50:50으로 데이터를 나누고,
앞에서 부터 50%의 데이터(이하, A그룹)는 'f1'컬럼을 A그룹의 중앙값으로 채우고,
뒤에서부터 50% 데이터(이하, B그룹)는 'f1'컬럼을 B그룹의 최대값으로 채운 후,
A그룹과 B그룹의 표준편차 합을 구하시오
단, 소수점 첫째자리까지 구하시오 (둘째자리에서 반올림)

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6979
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
df_len = int(df.shape[0]/2)

# 50:50 나누기 방법1
a = df[:df_len]
b = df[df_len:]

# 50:50 나누기 방법2
a = df.head(df_len)
b = df.tail(df_len)

a['f1'] = a['f1'].fillna(a['f1'].median())
b['f1'] = b['f1'].fillna(b['f1'].max())

print(round(a['f1'].std() + b['f1'].std(),1))
# 42.0
