
# =================================================================
# Association Rule
# = Apriori Algorithm
# = Market Basket Analysis (장바구니 분석)
# - 대용량의 트랜잭션 데이터(거래뎅터)로부터 'X이면 Y이다' 형식의
#   연관관계 발견하는 기법
# - 어떤 두 아이템 집합이 빈번히 발생하는가를 알려줌
# - 연관규칙을 수행하기 위해서는 거래데이터(transaction data)
#   형식으로 되어 있어야 함
#
# 지지도 Support
# 전체 거래 건수 중 X와 Y를 모두 포함하는 거래 건수 비율
# = X와 Y를 모두 포함하는 거래 수 / 전체 거래 수
# = n(X∩Y) / N
#
# 신뢰도 Confidence
# 항목집합 X를 포함하는 거래 중 항목집합 Y도 포함하는 거래 비율
# = X와 Y를 모두 포함하는 거래 수 / X가 포함된 거래 수
# = n(X∩Y) / n(X)
#
# 향상도 Lift
# 항목집합 X가 주어지지 않았을 때 항목집합 Y의 확률 대비
# 항목집합 X가 주어졌을 때 항목집합 Y의 확률 증가 비율
# = 연관규칙의 신뢰도 / 지지도
# = c(X->Y) / s(Y)
# 1) > 1 : 우수함
# 2) = 1 : X와 Y는 독립임
# =================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Market_Basket.csv', header=None)
data.head()

# transaction에 data에 있는 제품목록을 2차원 array로 담기
transactions = []
for i in range(data.shape[0]):
    transactions.append([str(data[j][i])
                         for j in range(data.shape[1] - data.isnull().sum(axis=1)[i])])

transactions


# apriori 모델 적용 ================================================

# 연관규칙분석 모듈은 사이킷런에 없고, 별도의 패키지로 구성되어 있음
# apyori 패키지 설치
!pip install apyori
from apyori import apriori
# Hyper Parameter
# - min_support : 최소 지지도 (기준값 이상만 제시)
# - min_confidence : 최소 신뢰도 (기준값 이상만 제시)
# - min_lift : 최소 향상도 (기준값 이상만 제시)

rules = apriori(transactions,
                min_support=0.015,
                min_confidence=0.2,
                min_lift=1,
                min_length=1)

results = list(rules)
results

# 총 78개의 규칙이 있음
print(len(results))

# 좀 더 깔끔하게 pandas로 결과 보기
df = pd.DataFrame(results)
df

df.to_csv('apriori_result.csv')

# 인덱스 6번 ~ 19번까지 규칙과
# 이때의 제품(item) 및 지지도(support) 확인
print(df.iloc[6:19][['items', 'support']])
#                              items   support
# 6             (burgers, spaghetti)  0.021464
# 7                     (eggs, cake)  0.019064
# 8             (cake, french fries)  0.017864
# 9            (mineral water, cake)  0.027463
# 10               (cake, spaghetti)  0.018131
# 11        (mineral water, chicken)  0.022797
# 12            (spaghetti, chicken)  0.017198
# 13               (eggs, chocolate)  0.033196
# 14           (chocolate, escalope)  0.017598
# 15       (chocolate, french fries)  0.034395
# 16  (frozen vegetables, chocolate)  0.022930
# 17        (ground beef, chocolate)  0.023064
# 18               (chocolate, milk)  0.032129

# 연관품목 시각화
# 78개 규칙 중 74개만 뽑아 그래프로 표현
ar = (df.iloc[1:74]['items'])
ar
# 1                       (burgers, eggs)
# 2               (burgers, french fries)
# 3                  (burgers, green tea)
# 4                       (burgers, milk)
# 5              (burgers, mineral water)
               
# 69    (mineral water, whole wheat rice)
# 70               (spaghetti, olive oil)
# 71                (pancakes, spaghetti)
# 72                  (spaghetti, shrimp)
# 73                (spaghetti, tomatoes)
# Name: items, Length: 73, dtype: object

# 파이썬의 대표적인 네트워크 시각화 패키지인 networkx 이용 ===========
# 도표의 기본틀은 matplotlib 사용
# 표현 스타일은 ggplot 사용

import networkx as nx

plt.style.use('ggplot')
plt.figure(figsize=(9,6))
G = nx.Graph()
G.add_edges_from(ar)
pos = nx.spring_layout(G)    # 에러 발생 -> 원인 파악 필요
nx.draw(G, pos, font_size=16,
        with_labels=False,
        edge_color='green',
        node_size = 800,
        node_color=['red','green','blue','cyan','orange','magenta'])
for p in pos:
    pos[p][1] += 0.07
nx.draw_networkx_labels(G, pos)
plt.show()

