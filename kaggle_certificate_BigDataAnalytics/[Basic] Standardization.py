'''
주어진 데이터에서 'f5'컬럼을 표준화
(Standardization (Z-score Normalization))하고
그 중앙값을 구하시오

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6707
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[['f5']])
df['f5'] = scaler.transform(df[['f5']])
df['f5'].median()
# 0.260619629559015

