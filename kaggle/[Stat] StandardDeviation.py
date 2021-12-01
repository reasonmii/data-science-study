'''
주어진 데이터 중 basic1.csv에서 'f4' 컬럼 값이
'ENFJ'와 'INFP'인 'f1'의 표준편차 차이를 절대값으로 구하시오

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6670
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
print(abs(df[(df['f4'] == 'ENFJ')]['f1'].std() - df[(df['f4'] == 'INFP')]['f1'].std()))
# 5.859621525876811

