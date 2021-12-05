'''
작업형 1-3
결측치 비중이 가장 큰 컬럼명
'''

# 시험 환경 setting
import pandas as pd
df = pd.read_csv('basic1.csv')

df = df.isnull().sum()
# id       0
# age      0
# city     0
# f1      31
# f2       0
# f3      95
# f4       0
# f5       0

df.sort_values(ascending=False).index[0]
