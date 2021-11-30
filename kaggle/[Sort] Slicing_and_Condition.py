
'''
주어진 데이터 셋에서 age컬럼 상위 20개의 데이터를 구한 후
f1의 결측치를 중앙값으로 채운다.
그리고 f4가 ISFJ와 f5가 20 이상인 f1의 평균값을 출력하시오!

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6804
'''

import pandas as pd
import numpy as np

df = pd.read_csv('basic1.csv')

# age 상위 20개
df = df.sort_values('age', ascending=False).reset_index(drop=True)
df = df[:20]
print(df)

# f1 결측값 : 중앙값으로 채우기
df['f1'] = df['f1'].fillna(df['f1'].median())

# 최종 값
df[(df['f4'] == 'ISFJ') & (df['f5'] >= 20)]['f1'].mean()
# 73.875


