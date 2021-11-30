
'''
주어진 데이터에서 20세 이상인 데이터를 추출하고
'f1'컬럼을 결측치를 최빈값으로 채운 후,
f1 컬럼의 여-존슨과 박스콕스 변환 값을 구하고,
두 값의 차이를 절대값으로 구한다음 모두 더해
소수점 둘째 자리까지 출력(반올림)하시오

A Box Cox transformation is a transformation
of non-normal dependent variables into a normal shape. 

The Yeo-Johnson transformation can be thought
of as an extension of the Box-Cox transformation.
It handles both positive and negative values,
whereas the Box-Cox transformation only handles positive values.
Both can be used to transform the data so as to improve normality.

dataset : basic1.csv
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6742
'''

import pandas as pd
import numpy as np

from sklearn.preprocessing import power_transform

df = pd.read_csv('basic1.csv')
df.head()

df = df[df['age'] >= 20]

# 최빈값으로 결측치 채우기
df['f1'] = df['f1'].fillna(df['f1'].mode()[0])

# f1컬럼의 Yeo-Johnson & Box–Cox 반환 값 구하기
df['y'] = power_transform(df[['f1']], standardize=False)  # default=Yeo-Johnson
df['b'] = power_transform(df[['f1']], standardize=False, method='box-cox')

# Box-Cox 값을 구할 수 있는 다른 방법
from scipy import stats
x = stats.boxcox(df['f1'])

# 두 값의 차이 구하기
print(round(sum(np.abs(df['y'] - df['b'])),2))
# 39.17


