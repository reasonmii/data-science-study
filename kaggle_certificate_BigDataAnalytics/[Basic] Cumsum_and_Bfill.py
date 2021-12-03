'''
주어진 데이터 셋에서 'f2' 컬럼이 1인 조건에 해당하는
데이터의 'f1'컬럼 누적합을 계산한다.
이때 발생하는 누적합 결측치는 바로 뒤의 값을 채우고,
누적합의 평균값을 출력한다.
(단, 결측치 바로 뒤의 값이 없으면 다음에 나오는 값을 채워넣는다)

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6706
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

f1_cs = df[df['f2'] == 1]['f1'].cumsum()
f1_cs = f1_cs .fillna(method='bfill')

print(f1_cs.mean())
# 980.3783783783783
