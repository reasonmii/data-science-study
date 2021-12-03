
'''
데이터에서 IQR을 활용해 'Fare'컬럼의 이상치를 찾고,
이상치 데이터의 여성 수를 구하시오

dataset : titanic_train.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6666
'''

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/titanic/train.csv')

q1 = df['Fare'].quantile(0.25)
q3 = df['Fare'].quantile(0.75)
iqr = q3 - q1

out = df[(df['Fare'] < q1 - iqr * 1.5) | (df['Fare'] > q3 + iqr * 1.5)]

print(len(out[out['Sex'] == 'female']))
# 70

