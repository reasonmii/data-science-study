'''
상위 10개, 하위 10개 차이
주어진 데이터에서 상위 10개 국가의 접종률 평균과
하위 10개 국가의 접종률 평균을 구하고, 그 차이를 구해보세요
(이상치 - 100%가 넘는 접종률 제거, 소수 첫째 자리까지 출력)

dataset : covid-vaccination-vs-death_ratio.csv
https://www.kaggle.com/sinakaraji/covid-vaccination-vs-death
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6751
'''


import pandas as pd
import numpy as np

df = pd.read_csv('../input/covid-vaccination-vs-death/covid-vaccination-vs-death_ratio.csv')

df.groupby('country').max()

# 이상치 제거
df = df[df['ratio'] <= 100]

# 다른방법 : head(10), tail(10)
max10 = df.sort_values('ratio', ascending=False)['ratio'][:10].mean()
min10 = df.sort_values('ratio', ascending=True)['ratio'][:10].mean()

print(max10 - min10)
# 97.94521254721668
