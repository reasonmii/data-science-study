
# =================================================================
# 군집분석 Cluster Analysis
# - 개체들의 특성을 대표하는 몇 개의 변수들을 기준으로
#   몇 개의 그룹(군집)으로 세분화
# - 유사성 : 개체 간의 거리 사용
#   ex) 유클리디안 거리 : sqrt(X좌표간의 차이^2 + y좌표간의 차이^2)
#
# sklearn.cluster.KMeans
# - n_clusters : 군집 수
#   경험과 주관으로 결정할 수도 있고
#   통계적 기준으로 최적의 군집수를 찾을 수도 있음
# =================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('plasma')
from sklearn.cluster import KMeans

data = pd.read_csv('Mall_Customers.csv', encoding='utf-8')

data.info()
data.describe()
data.head()

X = data.iloc[:, [3,4]]
X.head()

# 통계 기준으로 최적의 군집수 찾기
# 군집수를 1~21개까지 늘려보면서 kmeans.inertia_ 값 확인
# - 군집의 중심과 각 케이스(개체) 간 거리 계산
# - 값이 작을수록 군집이 잘 형성되었다는 의미
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_transform(X)
    wcss.append(kmeans.inertia_)

wcss

# 이를 꺾은 선 도표로 보면 군집수가 늘어날수록 수치가 작아짐
# 그러나 무조건 작아진다고 좋은 것은 아님
# 크게 감소하다가 변화가 없는 지점에서 최적의 군집수 K 결정
plt.figure()
plt.plot(range(1,21), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 위 결과를 바탕으로 k=5 결정
k=5
kmeans = KMeans(n_clusters=k)
y_kmeans = kmeans.fit_predict(X)
y_kmeans
# array([3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
#        3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 0,
#        3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2,
#        0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
#        1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
#        1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
#        1, 2])

# 결과를 기존 data set과 합치기
# 변수명 : Group
Group_cluster = pd.DataFrame(y_kmeans)
Group_cluster.columns=['Group']
full_data = pd.concat([data, Group_cluster], axis=1)
full_data
#       ID  Gender  Age  Income  Spend  Group
# 0      1    Male   19      15     39      3
# 1      2    Male   21      15     81      4
# 2      3  Female   20      16      6      3
# 3      4  Female   23      16     77      4
# 4      5  Female   31      17     40      3

# .cluster_centers_ : 각 좦의 중심점 결과
# 군집0 : 소득 55, 지출 50 -> 평균수준의 집단
# 군집1 : 소득 88, 지출 17 -> 많이 벌고 덜 쓰는 저축형 집단
# 군집4 : 소득 87, 지출 82 -> 번 만큼 쓰는 집단
# 이처럼 직관적으로 명확한 분리가 되어야 제대로 된 군집 수
kmeans_pred = KMeans(n_clusters=k, random_state=42).fit(X)
kmeans_pred.cluster_centers_
# array([[55.2962963 , 49.51851852],
#        [88.2       , 17.11428571],
#        [26.30434783, 20.91304348],
#        [25.72727273, 79.36363636],
#        [86.53846154, 82.12820513]])

# 소득과 지출이 (100, 50), (30, 80)인 사람이 속하는 군집 알아보기
# 새로운 데이터가 들어오면 학습된 모델(kmeans_pred)로 예측(predict)
# -> 각각 4, 3 군집으로 분류됨
kmeans_pred.predict([[100, 50], [30, 80]])
# array([4, 3])

# 군집에 이름 붙이기 : Cluster 1, Cluster 2, Cluster 3 ...
labels = [('Cluster ' + str(i+1)) for i in range(k)]
labels
# ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

# 2차원 도표에 산점도 scatter plot 그리기
# for문으로 각 개체만큼 하나씩 y_kmeans에 분석하여 저장된 좌표값 찍음
# 라벨에 따라 색을 달리하기 위해 label = labels[i] 설정
X = np.array(X)
plt.figure()
for i in range(k):
    plt.scatter(X[y_kmeans == i, 0],
                X[y_kmeans == i, 1],
                s = 20,
                c = cmap(i/k),   # 색 결정
                label=labels[i])

# 각 군집의 중심점만 찍은 scatter plot
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=100,
            c='black',
            label='Centroids',
            marker='X')
plt.xlabel('Income')
plt.ylabel('Spend')
plt.title('Kmeans cluster plot')
plt.legend()
plt.show()

# 각 개체의 좌표와 군집 중심의 좌표를 한 도표에 표현
plt.figure()
for i in range(k):
    plt.scatter(X[y_kmeans == i, 0],
                X[y_kmeans == i, 1],
                s = 20,
                c = cmap(i/k),
                label=labels[i])
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=100,
            c='black',
            label='Centroids',
            marker='X')
plt.xlabel('Income')
plt.ylabel('Spend')
plt.title('Kmeans cluster plot')
plt.legend()
plt.show()


# =================================================================
# K-mean Clustering
# 비계층적 군집분석
# - 분석속도가 빠름, 군집 형성, 유연하게 다른 군집으로 재군집화 가능
# =================================================================

# 군집분석의 계산과정
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# sepal_length, sepal_width : 꽃받침의 길이와 넓이
# petal_length, petal_width : 꽃잎의 길이와 넓이
# class : 꽃의 종류 (3종류)
iris = pd.read_csv('iris.csv')
iris.head()

# class 변수 우선 drop
# 군집분석은 레이블이 있으면 안 되기 때문
# 추후 분류가 잘 되었는지 확인하는 차원에서 사용할 예정
x_iris = iris.drop(['class'], axis=1)
y_iris = iris['class']
x_iris.head()
y_iris.head()

# 기술통계를 보면 4개 변수 단위가 다름
# 평균, 최대값, 최소값 차이 큰 편
# -> 정규화 필요
x_iris.describe()
#        sepal_length  sepal_width  petal_length  petal_width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.054000      3.758667     1.198667
# std        0.828066     0.433594      1.764420     0.763161
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(x_iris)
X_scale = scale.transform(x_iris)
pd.DataFrame(X_scale).head()
#           0         1         2         3
# 0 -0.900681  1.032057 -1.341272 -1.312977
# 1 -1.143017 -0.124958 -1.341272 -1.312977
# 2 -1.385353  0.337848 -1.398138 -1.312977
# 3 -1.506521  0.106445 -1.284407 -1.312977
# 4 -1.021849  1.263460 -1.341272 -1.312977

K = range(1, 10)
KM = [KMeans(n_clusters=k).fit(X_scale) for k in K]
centroids = [k.cluster_centers_ for k in KM]

# cdist 거리계산 : euclidean
D_k = [cdist(x_iris, centrds, 'euclidean') for centrds in centroids]
D_k

# cIdx : 계산한 거리 중 최소값(np.argmin)
# dist : 최소값
# avgWithinSS : 군집의 중심과 개체들 간 최소 거리들의 평균
cIdx = [np.argmin(D, axis=1) for D in D_k]
dist = [np.min(D, axis=1) for D in D_k]
avgWithinSS = [sum(d) / X_scale.shape[0] for d in dist]

# wcss : 개체들 간 거리를 제곱하여 더한 값
# tss : 전체 개체들 간 거리들을 제곱하여 개체수로 나눈 것
# bss : 둘의 차이
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(X_scale)**2)/X_scale.shape[0]
bss = tss-wcss

# 최적의 군집수를 구하기 위해 군집 수에 따른
# 평균 군집 내 거리(avgWithinSS) Plot 그리기
# elbow curve - Avg. within-cluster sum of squares
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
# 결과 : 군집 수를 4 or 6 정도가 적합해 보임

# 객체들 간 분산비율을 군집수를 달리하면서 보기
# elbow curve - percentage of variance explained
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
# 결과 : 군집 수를 4 or 6 정도가 적합해 보임

# 특정 데이터의 고유값(eigen-value) 계산하기
# 분석에 필수적이지는 않지만 군집에서의 위치 알 수 있음
# Calculation of eigenvectors & eigenvalues
w, v = np.linalg.eig(np.array([[ 0.91335, 0.75969],[0.75969, 0.69702]]))
print("\nEigen Values\n", w)
print("\nEigen Vectors\n", v)
# Eigen Values
#  [1.57253666 0.03783334]
#
# Eigen Vectors
#  [[ 0.75530088 -0.6553782 ]
#  [ 0.6553782   0.75530088]]

# 앞의 결과를 바탕으로 군집수 k=4로 하고 모델 훈련
k_means_fit = KMeans(n_clusters=4, max_iter=300)
k_means_fit.fit(X_scale)

# 4개 군집의 중심점
# 첫 번째 군집이 꽃 받침의 길이와 너비가 모두 가장 큰 꽃들이라는 특징
k_means_fit.cluster_centers_
# array([[ 1.14840215,  0.14758367,  0.99527619,  1.02432052],
#        [-1.34320731,  0.12656736, -1.31407576, -1.30726051],
#        [-0.01725724, -0.88648372,  0.37193617,  0.3029456 ],
#        [-0.73463631,  1.45201075, -1.29704352, -1.21071997]])

# 교차표 : 실제 3종류의 꽃(y_iris)와 군집으로 나눈 4개 비교
print("\nK-Means Clustering - Confusion Matrix\n\n",
      pd.crosstab(y_iris, k_means_fit.labels_,
                  rownames=['Actual'], colnames=['Predicted']))
# K-Means Clustering - Confusion Matrix
#
#  Predicted         0   1   2   3
# Actual                         
# Iris-setosa       0  23   0  27
# Iris-versicolor  11   0  39   0
# Iris-virginica   34   0  16   0

# Silhouette-score 확인
print("\nSilhouette-score: %0.3f"
      %silhouette_score(x_iris, k_means_fit.labels_, metric='euclidean'))
# Silhouette-score: 0.353

# 군집 수를 2개에서 10개까지 변경하면서 Silhouette-score 확인
for k in range(2,10):
    k_means_fitk = KMeans(n_clusters=k, max_iter=300)
    k_means_fitk.fit(x_iris)
    print("For K value", k, "Silhouette-score: %0.3f"
          %silhouette_score(x_iris, k_means_fitk.labels_, metric='euclidean'))
# For K value 2 Silhouette-score: 0.681
# For K value 3 Silhouette-score: 0.553
# For K value 4 Silhouette-score: 0.498
# For K value 5 Silhouette-score: 0.489
# For K value 6 Silhouette-score: 0.368
# For K value 7 Silhouette-score: 0.362
# For K value 8 Silhouette-score: 0.355
# For K value 9 Silhouette-score: 0.347    


# =================================================================
# Hierarchical Clustering
# 계층적 군집분석
# - 한 번 어떤 군집에 속한 개체는 분석과정에서 다른 군집과 더 가깝게
#   계산되어도 다른 군집화가 허용되지 않음
# - 속도가 다소 오래 걸림
# =================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('plasma')

data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:, [3,4]].values

# Dendrogram to choose number of clusters(k)
import scipy.cluster.hierarchy as sch

# ward : 개체 간 거리 계산
# sch의 dendrogram() 사용해서 dendrogram 그리기
# 케이스가 많으면 묶이는 개체 번호를 확인하기 어려움
plt.figure(1)
z = sch.linkage(X, method='ward')
dendrogram = sch.dendrogram(z)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('ward distances')
plt.show()

# 군집수를 5로 하여 병합군집 Agglomerative Clustering을 적용한 군집분석 수행
# - 시작할 때 각 포인트를 하나의 클러스터로 지정
# - 다음 종료 조건(지정한 군집 수)을 만족할 때까지 가장 비슷한 두 클러스터를 합침
k = 5
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = k, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# 군집분석 결과 시각화
labels = [('Cluster ' + str(i+1)) for i in range(k)]
plt.figure(2)
for i in range(k):
    plt.scatter(X[y_hc == i, 0],
                X[y_hc == i, 1],
                s=20,
                c=cmap(i/k),
                label=labels[i])

plt.xlabel('Age')
plt.ylabel('Spending score')
plt.title('HC cluster plot')
plt.legend()
plt.show()    

