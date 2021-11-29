'''
나이 구간 나누기
basic1 데이터 중 'age'컬럼 이상치를 제거하고,
동일한 개수로 나이 순으로 3그룹으로 나눈 뒤 각 그룹의 중앙값을 더하시오
- 이상치는 음수(0포함), 소수점 값
'''

import numpy as np
import pandas as pd

data = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

data.head()
print('Original : ', data.shape)

# age : 0 이하인 row 제거
data = data[~(data['age'] <= 0)]
print('0 이하 제거 후 : ', data.shape)

# age : 소수점 row 제거
data = data[(data['age'] == round(data['age'],0))]
print('소수점 제거 후 : ', data.shape)

data.head()

# age 3분할
data['range'] = pd.qcut(data['age'], q=3, labels=['group1', 'group2', 'group3'])
data['range'].value_counts()

g1_med = data[data['range'] == 'group1']['age'].median()
g2_med = data[data['range'] == 'group2']['age'].median()
g3_med = data[data['range'] == 'group3']['age'].median()

print(g1_med + g2_med + g3_med)
