
# =================================================================
# DBSCAN
# Density-based spatial clustering of applications with noise
# - 밀도기반 클러스터링 기법
# - 케이스가 집중되어 있는 밀도(density)에 초점을 두어
#   밀도가 높은 그룹을 클러스터링 하는 방식
# - 중심점을 기준으로 특정한 반경 이내에 케이스가 n개 이상 있을 경우
#   하나의 군집을 형성하는 알고리즘
# - 이상값(멀리 떨어진 데이터)을 탐지하는 데에 많이 활용
#
# 데이터 케이스(포인트)는 3가지로 분류
# - Core point
#   epsilon 반경 내에 최소점(minPts) 이상을 갖는 점
# - Border point
#   Core point의 epsilon 반경 내에 있으나,
#   그 자체로는 최소점(minPts)을 갖지 못하는 점
# - Noise point
#   Core point도 아니고 Border point도 아닌 점
#
# sklearn.cluster.DBSCAN
# - eps (epsilon) : 근접 이웃점을 찾기 위해 정의 내려야 하는 반경 거리
# - min_samples (minPts = minimum amount of points)
#   하나의 군집을 형성하기 위해 필요한 최소 케이스 수
# - metric : 개체 간 거리 (default : euclidean)
# =================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = pd.read_csv('iris.csv')
iris_data = iris[iris.columns[0:4]]
iris_data.head()

# DBSCAN 모델 적용
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
dbscan

dbscan.fit(iris_data)
dbscan.labels_
# array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#         0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,
#         1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,
#        -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1,  1,
#         1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1,
#         1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1, -1, -1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
#       dtype=int64)
# -> 2개의 군집(0, 1)으로 나타난 것
#    - 모델 기준에서 이상치로 파악된 것은 '-1'로 표현
#    - DBSCAN은 반경과 최소표본수를 어떻게 잡는가에 따라
#      -1 값이 많아질 수도, 적어질 수도 있음
#    - -1 데이터를 보고 이상치로 볼 수 있는지 확인하면서 하이퍼파라미터 조정 필요

pred = dbscan.fit_predict(iris_data)
pred = pd.DataFrame(pred)
pred.columns = ['predict']
pred.head()
#    predict
# 0        0
# 1        0
# 2        0
# 3        0
# 4        0

match_data = pd.concat([iris, pred], axis=1)
match_data.head()

# 교차표
cross = pd.crosstab(match_data['class'], match_data['predict'])
cross
# predict          -1   0   1
# class                      
# Iris-setosa       1  49   0
# Iris-versicolor   6   0  44
# Iris-virginica   10   0  40
# -> setosa에서 1개의 이상치
#    versicolor에서 6개의 이상치
#    virginica에서 10개의 이상치
# -> 반경을 넓힐 필요 있음

# 정규화도 필요
# 표준화한 데이터로 모델 재수행하면 이상치가 더 적어질 수 있음

# 시각화
# 변수가 4개지만 4차원 공간에 표현이 불가능
# -> '2차원'으로 차원 축소 필요
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(iris_data)
pca_2d = pca.transform(iris_data)

for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.show()

