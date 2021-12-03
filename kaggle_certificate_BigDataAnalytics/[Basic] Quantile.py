
'''
주어진 데이터에서 'f5'컬럼을 min-max 스케일 변환한 후,
상위 5%와 하위 5% 값의 합을 구하시오

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6750
'''

import numpy as np
import pandas as pd

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df[['f5']])
df[['f5']] = scaler.transform(df[['f5']])

print(df['f5'].quantile(0.05) + df['f5'].quantile(0.95))


