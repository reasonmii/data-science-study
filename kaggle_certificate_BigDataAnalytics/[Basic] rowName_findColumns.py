'''
빅데이터분석기사 실기 3회 기출문제

작업형 1-2
2000년 평균보다 큰값 개수
'''

# 시험 환경 setting
import random
box1 = []
box2 = []
box3 = []
for i in range(1, 100):
    num1 = random.randint(1, 200)
    num2 = random.randint(1, 200)
    num3 = random.randint(1, 200)
    box1.append(num1)
    box2.append(num2)
    box3.append(num3)

df = pd.DataFrame({
    1999 : box1,
    2000 : box2,
    2001 : box3
    }).T

df
#        0    1    2    3    4    5    6   ...   92   93   94   95   96   97   98
# 1999   94  131   12  137   39   73   35  ...  165  177   41  172   75  166   28
# 2000  167   67  141  141  198   45   84  ...   30  102  140    9  141   35   75
# 2001  148  155   96   63  120  177  183  ...  128    8  129   84   89  165  112

# 문제풀이
m = df.loc[2000].mean() 
print(sum(df.loc[2000, :] > m))  # True 개수
