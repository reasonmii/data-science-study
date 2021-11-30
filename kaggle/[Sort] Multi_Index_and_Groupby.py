'''
city와 f4를 기준으로 f5의 평균값을 구한 다음,
f5를 기준으로 상위 7개 값을 모두 더해 출력하시오
(소수점 둘째자리까지 출력)

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6789
'''

import numpy as np
import pandas as pd

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

# f5 평균값
df = df.groupby(['city', 'f4'])[['f5']].mean()
print(df)

# f5 기준 상위 7개
df = df.reset_index().sort_values('f5', ascending=False).head(7)
print(round(df['f5'].sum(), 2))
# 643.68

