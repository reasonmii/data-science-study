# =================================================================
# 배열 생성
# np.arange : 1차원 배열
# =================================================================

import numpy as np

v1 = np.array([1,2,3,4])
print(v1)

# 연속되거나 일정한 규칙을 가진 숫자
v1 = np.arange(5)
print(v1)          # [0 1 2 3 4]

# 데이터 형태 지정
v2 = np.arange(1, 10, 2, dtype=int)
print(v2)          # [1 3 5 7 9]

v3 = np.arange(3.5, 10.5, 2, dtype=float)
print(v3)          # [3.5 5.5 7.5 9.5]

# 제곱값
v4 = np.arange(1, 10, 2)**2
print(v4)          # [1  9 25 49 81]

# 세제곱값
v5 = np.arange(1, 10, 2)**3
print(v5)          # [1  27 125 343 729]


# =================================================================
# 행렬 생성
# np.reshape(차원, 행, 렬, order='C' or 'F')
# order = 'C' : 값을 행부터 채워 넣음 (default)
# order = 'F' : 값을 열부터 채워 넣음
# =================================================================

v1 = np.arange(12)
print(v1)
# [ 0  1  2  3  4  5  6  7  8  9 10 11]

v2 = v1.reshape(2, 6)
print(v2)
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]]

v3 = v1.reshape(2, 6, order='F')
print(v3)
# [[ 0  2  4  6  8 10]
#  [ 1  3  5  7  9 11]]

# 행렬의 연산 ------------------------------------------------------
# np.add (변수명, 변수명)
# np.subtract(변수명, 변수명)
# np.multiply(변수명, 변수명)
# np.dot(변수명, 변수명) : 행렬의 연산

v1 = np.arange(1,5).reshape(2,2)
print(v1)
# [[1 2]
#  [3 4]]

print(np.add(v1, v1))
# [[2 4]
#  [6 8]]

print(np.subtract(v1, v1))
# [[0 0]
#  [0 0]]

print(np.multiply(v1, v1))
# [[ 1  4]
#  [ 9 16]]

print(np.dot(v1, v1))
# [[ 7 10]
#  [15 22]]

# 다차원 배열 만들기 -----------------------------------------------
# 다차원 배열은 직접 눈으로 데이터를 일일이 확인하기 어려운 경우가 많음
# 최대값, 최소값 : np.amax(변수명), np.amin(변수명)
# 데이터 타입 : 변수명.dtype
# 행, 열, 차원 확인 : 변수명.shape

v1 = np.arange(12).reshape(2,3,2,order='F')
print(v1)

# [[[ 0  6]
#   [ 2  8]
#   [ 4 10]]
#
#  [[ 1  7]
#   [ 3  9]
#   [ 5 11]]]

v2 = np.arange(3.5, 10.5, 2, dtype=float)
print(v2)
# [3.5 5.5 7.5 9.5]

v3 = np.arange(1,5).reshape(2,2)
print(v3)
# [[1 2]
#  [3 4]]

print(np.amax(v3))  # 4
print(np.amin(v4))  # 1
print(v2.dtype)     # float64
print(v3.shape)     # (2, 2)


# =================================================================
# pandas
# 자료구조, 데이터 분석처리를 위한 핵심 패키지
# 크게 Series와 DataFrame 형태로 나눌 수 있음
# =================================================================

import pandas as pd
from pandas import Series, DataFrame


# =================================================================
# Series : index, value 형태
# index는 0, 1, 2, 3 ... 으로 자동 생성
# =================================================================

# series 설정
a = Series([1,3,5,7])
print(a)
# 0    1
# 1    3
# 2    5
# 3    7
# dtype: int64

print(a.index)   # RangeIndex(start=0, stop=4, step=1)
print(a.values)  # [1 3 5 7]

# index 변경
a2 = pd.Series([1,3,5,7], index=['a','b','c','d'])
print(a2)
# a    1
# b    3
# c    5
# d    7
# dtype: int64


# =================================================================
# DataFrame과 데이터 파일 불러오기
# DataFrame : 2차원 행렬구조
# csv 파일 불러오기 : pd.read_csv('파일명.csv')
# 한글이 깨질 때 : pd.read_csv('파일명', encoding='euc-kr')
# =================================================================

# csv 파일을 pandas DataFrame 형태로 불러오기
df = pd.read_csv('EX_GrapeData.csv')
df


# =================================================================
# DataFrame 확인하기
# =================================================================

# 변수명.head(불러올 행 개수)
# 변수명.tail(불러올 행 개수)
# default : 5개
df.head()
df.tail()

# 특정한 연속 행 확인
df[1:5]    # 1 ~ 4행 출력
df[:3]     # 처음 ~ 2행 출력
df[60:]    # 60 ~ 마지막 행 출력

# 열 불러오기 ------------------------------------------------------
# 방법1 : 데이터셋명[['열 이름']]
# 방법2 : 데이터셋명[데이터셋명.columns[['열 번호']]]
# 방법3 : 데이터셋명.loc[:, 첫 열 이름 : 끝 열 이름]

# series 형태
df['price']

# DataFrame 형태
df[['price']]

# 0번, 2번, 4번 열 출력
df[df.columns[[0,2,4]]]

df.loc[:, 'size':'price']

# 특정 행과 열을 모두 지정 ------------------------------------------
# 1~6 행, 0~1열
df.iloc[1:7, 0:2]

# 특정 값을 지정해서 불러오기 : at ----------------------------------
df.at[5, 'price']   # '129'


# =================================================================
# 데이터 변환하기
# insert, delete, rename
# =================================================================

# 전체 복사 (백업)
df_columns = df.copy()

# DataFrame의 열(변수) 이름 확인
df_columns.columns
# Index(['continent', 'brand', 'size', 'period', 'price'], dtype='object')

# DataFrame 열 필터링
df_columns = df_columns[['size','period','price']]
df_columns.head()

# 변수이름 변경 : period -> time
df_columns.rename(columns={'period':'time'}, inplace=True)
df_columns.columns
# Index(['size', 'time', 'price'], dtype='object')

# 열 추가
df_columns['growth'] = df_columns['size']/df_columns['time']
df_columns.columns
# Index(['size', 'time', 'price', 'growth'], dtype='object')

# 열 삭제
del df_columns['growth']
df_columns.columns
# Index(['size', 'time', 'price'], dtype='object')


# =================================================================
# 데이터 케이스 추출
# AND : &
# OR : |
# =================================================================

df_continent_brand = df[(df['continent'] == 1) & (df['brand'] == 1)]
df_continent_brand.head()

df_over_size_period = df[(df['size'] >= 10) & (df['period'] >= 30)]
df_over_size_period.head()


# =================================================================
# 코딩 변경
# 기존의 값을 변경해야 하는 경우
# =================================================================

# 기존 값 : brand 1,2,3
# 수정방향 : brand 1, 2 -> brand 1 / brand 3-> brand 2

# brand 범주와 수 확인하기
# 데이터셋명['변수명'].value_counts()
df['brand'].value_counts()
# 2    24
# 1    23
# 3    16
# Name: brand, dtype: int64

# 방법1 : replace 이용
recode_brand = {"brand": {1:1, 2:1, 3:2}}
df_recode1 = df.replace(recode_brand)
df_recode1.head()

df_recode1['brand'].value_counts()
# 1    47
# 2    16
# Name: brand, dtype: int64

# 방법2 : 함수로 정의
# 코딩 변경 함수 정의
def brand_groups(series):
    if series == 1:
        return 1
    elif series == 2:
        return 1
    elif series == 3:
        return 2

df['re_brand'] = df['brand'].apply(brand_groups)
df.head()

df['re_brand'].value_counts()
# 1    47
# 2    16
# Name: re_brand, dtype: int64


# =================================================================
# pandas와 numpy 전환
# pandas : 데이터를 먼저 눈으로 확인하기 위해 많이 사용
# numpy : 머신러닝, 딥러닝에서 많이 사용
# =================================================================

import pandas as pd
df = pd.read_csv('EX_GrapeData.csv')
df.head()

# pandas를 numpy로 변환
# 변수명이 사라짐
df_num = df.to_numpy()
df_num

# numpy를 pandas로 변환
# 변수명은 0, 1, 2, ... 로 자동 생성
df_pd = pd.DataFrame(df_num)
df_pd

# 변수명 변경
df_pd2 = pd.DataFrame(data = df_num, columns = ['continent','brand','size','period','price'])
df_pd2.head()
