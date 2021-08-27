

'''참고강의
Inflearn 파이썬 무료강의 (기본편) - 6시간 뒤면 나도 개발
'''

''' numpy
매우 빠르게 데이터 처리
'''
import numpy as np

data = [[1,2,3], [4,5,6], [7,8,9]]

a = np.array(data)
a

a.dtype     # dtype('int32')

# Change type from int to float
a = a.astype(np.float32)    # 방법1
a = a.astype('float32')     # 방법2
a           # dtype=float32

# array([1, 3, 5, 7, 9])
np.arange(1, 10, 2)

# reshape(row, column)
np.arange(1,10).reshape(3,3)
#array([[1, 2, 3],
#       [4, 5, 6],
#       [7, 8, 9]])

np.arange(1, 13).reshape(3,2,2)
#array([[[ 1,  2],
#        [ 3,  4]],
#
#       [[ 5,  6],
#        [ 7,  8]],
#
#       [[ 9, 10],
#        [11, 12]]])

np.nan * 10    # nan

a = np.arange(1,10).reshape(3, 3)
a[0][1] = np.nan         # Error : integer에서는 nan 사용 불가

# linespace(start, end, 몇 개를 생성할지)
np.linspace(1,10,20)
array([ 1.        ,  1.47368421,  1.94736842,  2.42105263,  2.89473684,
#        3.36842105,  3.84210526,  4.31578947,  4.78947368,  5.26315789,
#        5.73684211,  6.21052632,  6.68421053,  7.15789474,  7.63157895,
#        8.10526316,  8.57894737,  9.05263158,  9.52631579, 10.        ])


       
### -------------------------------------- < Calculate >

data = np.arange(1,10).reshape(3,3)
data

# 각 원소들끼리의 계산
data + data    # sum of each element
data * data
data / data

# 행렬의 곱을 계산하고 싶을 때
np.dot(data, data)
data@data



### -------------------------------------- < 차원 >

### 0차원 : Scala
a = np.array(1)
a         # array(1)
a.shape   # ()
a.ndim    # 차원 : 0


### 1차원 : vector
a = np.array([1,2,3])
a         # array([1, 2, 3])
a.shape   # (3,)
a.ndim    # 차원 : 1


### 2차원 : Matrix
a = np.array([[1,2,3],[4,5,6]])
a
#array([[1, 2, 3],
#       [4, 5, 6]])
a.shape   # (2, 3)
a.ndim    # 차원 : 2

# 주의 : 아래 형태도 [[]] 이기 때문에 2차원 임
a = np.array([[1]])


### matrix 유형
np.arange(12).reshape(2,3,2)  # fill with the number 0~11
np.ones(12).reshape(2,3,2)    # fill with ones
np.zeros(12).reshape(2,3,2)   # fill with zeros

np.eye(3)                     # 단위 행렬
#array([[1., 0., 0.],
#       [0., 1., 0.],
#       [0., 0., 1.]])

np.zeros([3,4])
np.zeros([3,4,2])

# empty : 0에 가까운 값으로 채우기
f = np.empty([2,3])
f
array([[ 1.        ,  1.47368421,  1.94736842],
       [ 9.05263158,  9.52631579, 10.        ]])

# np.full((row, column), 채우고 싶은 숫자)
np.full((3,4), 1000)

np.linspace(2,10,6)
#array([ 2. ,  3.6,  5.2,  6.8,  8.4, 10. ])


### 3차원 이상의 다차원 행렬 : Tensor
a = np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]])
a
#array([[[ 1,  2],
#        [ 3,  4],
#        [ 5,  6]],
#
#       [[ 7,  8],
#        [ 9, 10],
#        [11, 12]]])
a.shape   # (2, 3, 2)
a.ndim    # 차원 : 3
     

### 집게함수
a = np.arange(10).reshape(2, 5)
#array([[0, 1, 2, 3, 4],
#       [5, 6, 7, 8, 9]])
a[0][0] = 100
#array([[100,   1,   2,   3,   4],
#       [  5,   6,   7,   8,   9]])
a[0,1] = 1000
#array([[ 100, 1000,    2,    3,    4],
#       [   5,    6,    7,    8,    9]])

np.sum(a)         # 합 : 1144
np.mean(a)        # 평균 : 114.4
np.median(a)      # 중앙값 : 6.5
np.std(a)         # 표준편차 : 296.5485457728633
np.var(a)         # 분산 : 87941.04

sum(a)            # array([ 105, 1006,    9,   11,   13])
np.sum(a, axis = 0)    # sum each column : array([ 105, 1006,    9,   11,   13])
np.sum(a, axis = 1)    # sum each row : array([1109,   35])

np.max(a)      # 1000
np.min(a)      # 2



''' pandas
'''
import numpy as np
import pandas as pd

data = np.arange(0, 50, 10)


### -------------------------------------- < pd.Series >
# index 자동 설정 : 0 ~ 4
pd.Series(data)
#0     0
#1    10
#2    20
#3    30
#4    40
#dtype: int32

# 명시적 index : index 직접 설정하기
a = pd.Series(data, index = ['a', 'b', 'c', 'd', 'e'])
a
#a     0
#b    10
#c    20
#d    30
#e    40
#dtype: int32

a['b']
a.loc['b']   # 우리가 명시한 index로 value 찾기 : 10
a.iloc[1]    # 순서로 value 찾기 : 10


### -------------------------------------- < 산술연산 >
a + 10
a - 10
a * 10
a ** 2
a / 5     # float로 바뀜
a // 5    # 같은 나누기인데 int형으로 반환
a % 3

a > 15
#a    False
#b    False
#c     True
#d     True
#e     True
#dtype: bool

a[a > 15]
#c    20
#d    30
#e    40
#dtype: int32


### -------------------------------------- < 집계함수 >

# 더하기
a.add(100)
#a    100
#b    110
#c    120
#d    130
#e    140
#dtype: int32

# 빼기
a.sub(100)
#a   -100
#b    -90
#c    -80
#d    -70
#e    -60
#dtype: int32

# 곱하기
a.mul(100)
a       0
#b    1000
#c    2000
#d    3000
#e    4000
#dtype: int32

# 나누기
a.div(100)
#a    0.0
#b    0.1
#c    0.2
#d    0.3
#e    0.4
#dtype: float64

# 나머지
a.mod(3)
#a    0
#b    1
#c    2
#d    0
#e    1
#dtype: int32

# 최솟값
a.min()      # 0
a.max()      # 40
a.sum()      # 100
a.mean()     # 20.0
a.median()   # 20.0
a.std()      # 15.811388300841896
a.var()      # 250.0



### -------------------------------------- < DataFrame >
# Excel 같은 것

rawData = np.random.randint(50, 100, size=(4,3))
rawData
#array([[97, 78, 76],
#       [77, 63, 97],
#       [55, 96, 81],
#       [93, 71, 70]])

df = pd.DataFrame(rawData,
                  index = ['1반', '2반', '1반', '2반'],
                  columns = ['국','영','수'])
df
#     국   영   수
#1반  97  78  76
#2반  77  63  97
#1반  55  96  81
#2반  93  71  70

df['국']     # df[0] -> Error
#1반    97
#2반    77
#1반    55
#2반    93
#Name: 국, dtype: int32


### 행 별 평균값 추가하기
# 소수점 2자리까지 출력
df['평균'] = round((df['국'] + df['영'] + df['수'])/3, 2)
df
#     국   영   수     평균
#1반  97  78  76  83.67
#2반  77  63  97  79.00
#1반  55  96  81  77.33
#2반  93  71  70  78.00

df['na'] = np.nan
df
#     국   영   수     평균  na
#1반  97  78  76  83.67    NaN
#2반  77  63  97  79.00    NaN
#1반  55  96  81  77.33    NaN
#2반  93  71  70  78.00    NaN


### 컬럼 삭제
del df['na']
df
#     국   영   수     평균
#1반  97  78  76  83.67
#2반  77  63  97  79.00
#1반  55  96  81  77.33
#2반  93  71  70  78.00

df = df.drop(['평균'], axis = 'columns')
df
#     국   영   수
#1반  97  78  76
#2반  77  63  97
#1반  55  96  81
#2반  93  71  70


### 조건에 만족하는 열만 추출
df[df.평균 > 78]
#     국   영   수     평균
#1반  97  78  76  83.67
#2반  77  63  97  79.00

### column과 row를 바꾸기
df.T
#     1반    2반    1반    2반
#국  97.0  77.0  55.0  93.0
#영  78.0  63.0  96.0  71.0
#수  76.0  97.0   NaN  70.0



### -------------------------------------- < 결측값 처리 >
df = df.astype('float64')
df['수'][2] = np.nan
df
#       국     영     수
#1반  97.0  78.0  76.0
#2반  77.0  63.0  97.0
#1반  55.0  96.0   NaN
#2반  93.0  71.0  70.0

# 결측값이 있는 raw or column 삭제
# 단, 원본이 바뀌지는 않음
df.dropna(axis = 0)       # delete the row
df.dropna(axis = 1)       # delete the column
df.dropna(axis = 0, inplace = True)  # 원본을 바꾸고 싶은 경우
#       국     영     수
#1반  97.0  78.0  76.0
#2반  77.0  63.0  97.0
#2반  93.0  71.0  70.0

# 결측값을 원하는 값으로 바꾸기
df.fillna('hello')
#       국     영      수
#1반  97.0  78.0     76
#2반  77.0  63.0     97
#1반  55.0  96.0  hello
#2반  93.0  71.0     70
df.fillna(0)           # 0으로 바꾸기
df.fillna(df.mean())   # 평균 값으로 바꾸기



### -------------------------------------- < MultiIndex >

### 행 index 추가하기
df.index = [['1학년','1학년','2학년','2학년'],['1반','2반','1반','2반']]
#           국     영     수
#1학년 1반  97.0  78.0  76.0
#     2반  77.0  63.0  97.0
#2학년 1반  55.0  96.0   NaN
#     2반  93.0  71.0  70.0

### 열 index 추가하기
df.columns = [['언어','언어','수리'],['국','영','수']]
df
#          언어          수리
#           국     영     수
#1학년 1반  97.0  78.0  76.0
#     2반  77.0  63.0  97.0
#2학년 1반  55.0  96.0   NaN
#     2반  93.0  71.0  70.0

df['언어']
#           국     영
#1학년 1반  97.0  78.0
#     2반  77.0  63.0
#2학년 1반  55.0  96.0
#     2반  93.0  71.0

df['언어']['국']
#1학년  1반    97.0
#      2반    77.0
#2학년  1반    55.0
#      2반    93.0
#Name: 국, dtype: float64

### index location 0번째 행
df.iloc[0]
#언어  국    97.0
#    영    78.0
#수리  수    76.0
#Name: (1학년, 1반), dtype: float64

# 1학년 값만 가져오기
df.loc['1학년']
df.loc['1학년'].loc['1반']



### -------------------------------------- < Data 사전분석 >

# basic information
# 용량 너무 크면 불필요한 컬럼 삭제할 것 (나중에 데이터 분석 시 힘듦)
df.info()
#<class 'pandas.core.frame.DataFrame'>
#MultiIndex: 4 entries, (1학년, 1반) to (2학년, 2반)
#Data columns (total 3 columns):
#(언어, 국)    4 non-null float64
#(언어, 영)    4 non-null float64
#(수리, 수)    3 non-null float64
#dtypes: float64(3)
#memory usage: 248.0+ byte

df.head()
df.tail()

df.dtypes

df.describe()
#              언어                    수리
#               국          영          수
#count   4.000000   4.000000   3.000000
#mean   80.500000  77.000000  81.000000
#std    19.070046  14.071247  14.177447
#min    55.000000  63.000000  70.000000
#25%    71.500000  69.000000  73.000000
#50%    85.000000  74.500000  76.000000
#75%    94.000000  82.500000  86.500000
#max    97.000000  96.000000  97.000000


### 결측치 여부 확인
df.isnull()
#           언어            수리
#            국      영      수
#1학년 1반  False  False  False
#    2반  False  False  False
#2학년 1반  False  False   True
#    2반  False  False  False

# 결측치가 몇 개 있는지
df.isnull().sum()
#언어  국    0
#    영    0
#수리  수    1
#dtype: int64



### -------------------------------------- < 값의 연결 >

a = pd.DataFrame(np.arange(1,10).reshape(3,3))
a
#   0  1  2
#0  1  2  3
#1  4  5  6
#2  7  8  9

b = pd.Series(np.arange(10, 40, 10))
b
#0    10
#1    20
#2    30
#dtype: int32

### b를 column으로 추가
pd.concat([a, b], axis = 1)
#   0  1  2   0
#0  1  2  3  10
#1  4  5  6  20
#2  7  8  9  30

# b열의 컬럼명을 기존 dataframe a 컬럼명과 이어지게 하기
pd.concat([a, b], axis = 1, ignore_index = True)
#   0  1  2   3
#0  1  2  3  10
#1  4  5  6  20
#2  7  8  9  30

### b를 row로 추가
a.append(b, ignore_index = True) 
#    0   1   2
#0   1   2   3
#1   4   5   6
#2   7   8   9
#3  10  20  30

