'''
mtcars 데이터셋(mtcars.csv)의 qsec 컬럼을
최소최대 척도(Min-Max Scale)로 변환한 후
0.5보다 큰 값을 가지는 레코드 수를 구하시오

dataset : mtcars.csv
https://www.kaggle.com/ruiromanini/mtcars
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/mtcars/mtcars.csv')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df[['qsec']])
df[['qsec2']] = scaler.transform(df[['qsec']])

print(len(df[df['qsec2'] > 0.5]))
# 9



