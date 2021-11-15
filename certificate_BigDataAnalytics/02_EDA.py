
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('EX_CEOSalary.csv', encoding='utf-8')

data.info()
data.head()


# =================================================================
# 1. 단변량 데이터 탐색
# =================================================================

# 범주형 변수 ======================================================

# category 확인
data['industry'].value_counts()

# 1,2,3,4 -> 각 이름으로 변경
data['industry'] = data['industry'].replace([1,2,3,4],['Service','IT','Finance','Others'])
data['industry'].value_counts()

# Plot 시각화
# % matplotlib inline
data['industry'].value_counts().plot(kind="pie")
data['industry'].value_counts().plot(kind="bar")

# 숫자형 변수 ======================================================

# 기술통계량 descriptive statistics
# 확인사항 : 평균과 median의 차이가 크지 않은지 (이상치)
data.describe()

# 자료분포 확인
# 왜도(skewness) = 0 : 정규분포 (대칭)
# 왜도 > 2 : 어느 한 쪽으로 치우쳐져 있다는 의미
# 왜도 > 0 : 우측꼬리분포 (왼쪽에 자료가 더 많고 이상치는 큰 값이 많음)
# 왜도 < 0 : 좌측꼬리분포 (오른쪽에 자료가 더 많고 이상치는 작은 값이 많음)
data.skew()
# salary    6.904577
# sales     5.035336
# roe       1.572126
# dtype: float64

# 첨도(kurtosis) = 0 : 정규분포와 같은 높이
# 첨도↑ : 자료가 중심에 몰려 있음
data.kurtosis()
# salary    58.971229
# sales     33.115193
# roe        3.797366
# dtype: float64

# 히스토그램 =======================================================
# bins : 구간너비
# figsize : 도표 크기

# 데이터셋 내 모든 numerical 변수의 히스토그램
data.hist(bins=50, figsize=(20,15))

# 특정 변수의 히스토그램
data['salary'].hist(bins=50, figsize=(20,15))
data['sales'].hist(bins=50, figsize=(20,15))


# =================================================================
# 2. 이변량 데이터 탐색
# =================================================================

# 상관계수 =========================================================
# method : pearson (default), spearman, kendall
data.corr()

data.corr(method="pearson")
data.corr(method="spearman")
data.corr(method="kendall")

# 산점도 Scatter plot ==============================================
# plt.scatter(X, y)

plt.scatter(data['sales'], data['salary'], alpha=0.5)
plt.show()

plt.scatter(data['roe'], data['salary'], alpha=0.5)
plt.show()

# group by, describe ===============================================
# data에서 category 변수인 'industry' 별로 salary 기술통계량 계산
# count, mean, std, min, 25%, 50%, 75%, max
data.groupby('industry')[['salary']].describe()


# =================================================================
# 3. 이상치 처리
# =================================================================

# Box-whisker plot ================================================
# return_type : axes, dict, both

data.boxplot(column='salary', return_type='both')
data.boxplot(column='sales', return_type='both')
data.boxplot(column='roe', return_type='both')

# 1) salary 변수 이상치 처리 =======================================
Q1_salary = data['salary'].quantile(q=0.25)
Q3_salary = data['salary'].quantile(q=0.75)
IQR_salary = Q3_salary - Q1_salary
IQR_salary

data_IQR = data[(data['salary'] < Q3_salary + IQR_salary * 1.5) &
                (data['salary'] > Q1_salary - IQR_salary * 1.5)]

data_IQR['salary'].hist()

# histogram
data_IQR.hist(bins=50, figsize=(20,15))

# corr
data_IQR.corr()

# scatter plot
plt.scatter(data_IQR['sales'], data_IQR['salary'], alpha=0.5)
plt.show()

plt.scatter(data_IQR['roe'], data_IQR['salary'], alpha=0.5)
plt.show()

# 2) sales 변수 이상치 처리 ========================================
Q1_sales = data['sales'].quantile(0.25)
Q3_sales = data['sales'].quantile(0.75)
IQR_sales = Q3_sales - Q1_sales
IQR_sales

data_IQR = data[(data['salary'] > Q1_salary - IQR_salary * 1.5) &
                (data['salary'] < Q3_salary + IQR_salary * 1.5) &
                (data['sales'] > Q1_sales - IQR_sales * 1.5) &
                (data['sales'] < Q3_sales + IQR_sales * 1.5)]

# histogram
data_IQR.hist(bins=50, figsize=(20,15))

# corr
data_IQR.corr()


# =================================================================
# 4. 변수 변환
# =================================================================

# 1) log 변환 =====================================================
# -> 이상치가 중앙으로 이동함으로써 이상치 영향 제거 (정규분포에 가까워짐)
data['log_salary'] = np.log(data['salary'])
data['log_sales'] = np.log(data['sales'])
data['log_roe'] = np.log(data['roe'])
data.head()

data.hist(bins=50, figsize=(20,15))

# 상관계수 향상
data.corr()

# 2) 제곱근 변환 ==================================================
# -> 이상치 영향 제거 (정규분포에 가까워짐)
data['sqrt_salary'] = np.sqrt(data['salary'])
data['sqrt_sales'] = np.sqrt(data['sales'])
data['sqrt_roe'] = np.sqrt(data['roe'])
data.head()

data.hist(bins=50, figsize=(20,15))

# 상관계수 향상
data.corr()


# =================================================================
# 5. 결측치 처리
# =================================================================

data = pd.read_csv('Ex_Missing.csv')
data

# 1) 결측치 확인 ==================================================

# 결측이면 True, 결측이 아니면 False 반환
pd.isnull(data)
data.isnull()

# 결측이면 False, 결측이 아니면 True 반환
pd.notnull(data)
data.notnull()

# 전체 데이터에서 변수 별 결측값 개수 확인
data.isnull().sum()

# 특정 변수의 결측값 개수 확인
data['salary'].isnull().sum()

# 변수 별 결측이 아닌 개수 확인
data.notnull().sum()

# 특정 변수의 결측이 아닌 개수 확인
data['salary'].notnull().sum()

# 행 단위로 결측값 개수 구하기
data.isnull().sum(1)

# 행 단위로 결측값 개수 구해서 새로운 변수 생성하기
data['missing'] = data.isnull().sum(1)
data

# 변수 삭제
del data['missing']

# 행 단위로 실측값 개수 구해서 새로운 변수 생성
data['valid'] = data.notnull().sum(1)
data

# 2) 결측값 제거 : dropna() =======================================

# 결측값이 있는 행/열 제거
# axis=0 : 행
data_del_row = data.dropna(axis=0)
data_del_row

# axis=1 : 열
data_del_col = data.dropna(axis=1)
data_del_col

# 특정 변수에서 결측값이 있는 행 제거
data[['salary']].dropna()
data[['salary', 'sales', 'roe', 'industry']].dropna()

# 특정 변수에서 결측값이 있는 열 제거
data[['salary', 'sales', 'roe', 'industry']].dropna(axis=1)

# 3) 결측값 대체 ==================================================

# 3-1) 특정 값으로 대체

# 0으로 대체
data_0 = data.fillna(0)
data_0

# 'missing'으로 대체
data_missing = data.fillna("missing")
data_missing

# 해당 변수에서 결측치의 바로 앞 값으로 대체
# method = 'ffill' or method = 'pad'
data_ffill = data.fillna(method='ffill')
data_ffill

data_ffill = data.fillna(method='pad')
data_ffill

# 해당 변수에서 결측치의 바로 다음 값으로 대체
# method = 'bfill' or method = 'backfill'
data_bfill = data.fillna(method='bfill')
data_bfill

data_bfill = data.fillna(method='backfill')
data_bfill

# 3-2) 변수별 평균으로 대체
data_mean = data.fillna(data.mean())
data_mean

data_median = data.fillna(data.median())
data_median

data_max = data.fillna(data.max())
data_max

data_min = data.fillna(data.min())
data_min

# 데이터 내 전체 결측값을 특정 변수의 평균값으로 대체
data_other_mean = data.fillna(data.mean()['salary'])
data_other_mean

# 3-3) 다른 변수 값으로 대체
# 새로운 변수 생성 : sales가 null이 아니면 sales, null이면 salary
data2 = data.copy()
data2['sales_new'] = np.where(pd.notnull(data2['sales']) == True,
                              data2['sales'], data2['salary'])
data2

# 3-4) 그룹 평균 값으로 대체
# industry의 category에 따라 전체 변수의 값을 평균 값으로 대체
data.groupby('industry').mean()
#                salary         sales    roe  valid
# industry                                         
# 1         1013.666667  14981.224975  13.64    3.4
# 2         1077.400000   3158.425049  15.80    3.6

# lambda 함수 사용
# apply()를 이용해 집단별 평균 적용
fill_mean_func = lambda g: g.fillna(g.mean())
data_group_mean = data.groupby('industry').apply(fill_mean_func)
data_group_mean
#                  salary         sales        roe  industry  valid
# industry                                                         
# 1        0  1095.000000  27595.000000  14.100000         1      4
#          1  1013.666667   9958.000000  10.900000         1      3
#          2  1013.666667   6125.899902  23.500000         1      3
#          3   578.000000  16246.000000   5.900000         1      4
#          4  1368.000000  14981.224975  13.800000         1      3
# 2        5  1145.000000   3158.425049  20.000000         2      3
#          6  1078.000000   2266.699951  16.400000         2      4
#          7  1094.000000   2966.800049  16.299999         2      4
#          8  1237.000000   4570.200195  10.500000         2      4
#          9   833.000000   2830.000000  15.800000         2      3

# 집단 별로 분석가가 설정한 특정 값으로 대체
fill_values = {1: 1000, 2:2000}
fill_func = lambda d: d.fillna(fill_values[d.name])

data_group_value = data.groupby('industry').apply(fill_func)
data_group_value

# 변수 별로 다른 대체방법을 한 번에 적용
# salary : 보간법 (interpolate)
# ★ 보간법 : missing data 전 값과 다음 값 사이의 중간값
# sales : 평균
# roe : missing
missing_fill_val = {'salary': data.salary.interpolate(),
                    'sales': data.sales.mean(),
                    'roe': 'missing'}

print(missing_fill_val)

data_multi = data.fillna(missing_fill_val)
data_multi


# =================================================================
# 6. 데이터정제 실전과제
# =================================================================

data = pd.read_csv('house_raw.csv')

data.head()
data.describe()
data.info()

data.hist(bins=50, figsize=(20,15))

# 1) 선형회귀 적용 (정제 전 데이터) =================================
X = data[data.columns[0:5]]
y = data[["house_value"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()
scaler_minmax.fit(X_train)
X_scaled_minmax_train = scaler_minmax.transform(X_train)
X_scaled_minmax_test = scaler_minmax.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_scaled_minmax_train, y_train)

pred_train = model.predict(X_scaled_minmax_train)
model.score(X_scaled_minmax_train, y_train)
# 0.5463729131516732

pred_test = model.predict(X_scaled_minmax_test)
model.score(X_scaled_minmax_test, y_test)
# -2.8220648010161544
# -> 음수가 나온 이유 : 이상치 때문

# 2) 데이터 정제를 위한 세부 검토 ===================================

# bedrooms : 0.6 미만 데이터
data_bedroom = data[data['bedrooms'] < 0.6]
data_bedroom['bedrooms'].hist(bins=50, figsize=(20,15))

# bedrooms : 0.6 이상 데이터
# -> 14개만 있음 (제거해도 큰 문제가 없을 것으로 보임)
data_bedroom2 = data[data['bedrooms'] >= 0.6]
print(data_bedroom2['bedrooms'].value_counts())
data_bedroom2

# households : 10 미만 데이터
data_households = data[data['households'] < 10]
data_households['households'].hist(bins=100, figsize=(20,15))

# households : 10 이상 데이터
# -> 22개만 있음 (제거해도 큰 문제가 없을 것으로 보임)
data_households2 = data[data['households'] >= 10]
print(data_households2['households'].value_counts())
data_households2

# rooms : 20개 미만
data_room = data[data['rooms'] < 20]
data_room['rooms'].hist(bins=100, figsize=(20,15))

# rooms : 20개 이상
# -> 적은 데이터는 않지만 전체 데이터 수가 워낙 커서 제거해도 괜찮을 것으로 보임
data_room2 = data[data['rooms'] >= 20]
print(data_room2['rooms'].value_counts())
data_room2

# 3) 정제 데이터셋 생성 ============================================

new_data = data[(data['bedrooms'] < 0.5) &
                (data['households'] < 7) &
                (data['rooms'] < 12)]

new_data.describe()
new_data.hist(bins=50, figsize=(20,15))

# 4) 선형회귀 적용 (정제 후 데이터) ================================

X = new_data[new_data.columns[0:5]]
y = new_data[['house_value']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()

scaler_minmax.fit(X_train)
X_scaled_minmax_train = scaler_minmax.transform(X_train)
X_scaled_minmax_test = scaler_minmax.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_scaled_minmax_train, y_train)

pred_train = model.predict(X_scaled_minmax_train)
print("훈련데이터 정확도 : ", model.score(X_scaled_minmax_train, y_train))
# 훈련데이터 정확도 :  0.5706921210926263

pred_test = model.predict(X_scaled_minmax_test)
print("검증데이터 정확도 : ", model.score(X_scaled_minmax_test, y_test))
# 검증데이터 정확도 :  0.5826083517811866

# 정제된 최종 테이블을 CSV 파일로 저장
new_data.to_csv('house_price.csv', index=False)
