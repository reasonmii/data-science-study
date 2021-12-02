'''
'f4'컬럼을 기준 내림차순 정렬과 'f5'컬럼기준 오름차순 정렬을 순서대로 다중 조건 정렬하고
앞에서부터 10개의 데이터 중 'f5'컬럼의 최소값 찾고,
이 최소값으로 앞에서 부터 10개의 'f5'컬럼 데이터를 변경함
그리고 'f5'컬럼의 평균값을 계산함
단 소수점 둘째자리까지 출력(셋째자리에서 반올림)

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6979
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
df = df.sort_values(['f4', 'f5'], ascending=(False, True))
df['f5'][:10] = df['f5'].head(10).min()
print(round(df['f5'].mean(),2))
# 53.63
