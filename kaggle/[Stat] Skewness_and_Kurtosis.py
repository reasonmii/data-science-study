
'''
주어진 데이터 중 train.csv에서 'SalePrice' 컬럼의
왜도와 첨도를 구한 값과,
'SalePrice'컬럼을 스케일링(log1p)로 변환한 이후
왜도와 첨도를 구해 모두 더한 다음 소수점 2째자리까지 출력하시오

dataset : House Prices - Advanced Regression Technique
https://www.kaggle.com/agileteam/bigdatacertificationkr/tasks?taskId=6669
'''

import pandas as pd
import numpy as np

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

sk_bf = df['SalePrice'].skew()
kt_bf = df['SalePrice'].kurt()

# 로그변환
# log1p : 자연로그 log(1 + x) 값
df['SalePrice2'] = np.log1p(df['SalePrice'])

sk_af = df['SalePrice2'].skew()
kt_af = df['SalePrice2'].kurt()

print(round(sk_bf + kt_bf + sk_af + kt_af, 2))
# 9.35




