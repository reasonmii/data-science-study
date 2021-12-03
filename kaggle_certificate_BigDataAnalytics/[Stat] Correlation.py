
'''
주어진 데이터에서 상관관계를 구하고,
quality와의 상관관계가 가장 큰 값과, 가장 작은 값을 구한 다음 더하시오

dataset : winequality-red.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6755
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

# -1 : quality와 quality 상관관계 제거
df = df.corr()['quality'][:-1]
print(df.min() + df.max())
# 0.08560854373713056
