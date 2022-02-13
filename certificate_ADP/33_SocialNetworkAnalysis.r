
#===============================================================================================================
# 사회연결망분석 SNA Social Network Analysis
# 개인과 집단들 간의 관계를 노드와 링크로 모델링
#
# 중심성 Centrality
# 연결정도 중심성 : 한 점에 직접적으로 연결된 점들의 집합
# 근접 중심성 : 한 노드로부터 다른 노드에 도달하기까지 필요한 최소 단계의 합
# 매개 중심성 : 네트워크 내에서 한 점이 담당하는 매개자 혹은 중재자 역할의 정도
# 위세 중심성 : 자신의 연결정도 중심성으로부터 발생하는 영향력 + 자신과 연결된 타인의 영향력
#               보나시치(Bonacich) 권력지수 : 연결된 노드의 중요성에 가중치를 둬 노드의 중심성 측정
#===============================================================================================================

# 커뮤니티 수 측정하기

### 1) WALKRAP 알고리즘
# 일련의 random walk 과정을 통해 커뮤니티 발견
# 각 vertex(그래프 꼭지점)를 하나의 커뮤니티로 취급
# -> 점차 더 큰 그룹을 병합하며 clustering

install.packages("igraph")
install.packages("animation")
install.packages("NetData")

library(igraph)
library(animation)

data(studentnets.M182, package = "NetData")
head(m182_full_data_frame, n=10)
#     ego alter friend_tie social_tie task_tie
# 1    1     1          0       0.00      0.0
# 2    1     2          0       0.00      0.0
# 3    1     3          0       0.00      0.0
# 4    1     4          0       0.00      0.0
# 5    1     5          0       1.20      0.3
# 6    1     6          0       0.00      0.0
# 7    1     7          0       0.00      0.0
# 8    1     8          0       0.15      0.0
# 9    1     9          0       2.85      0.3
# 10   1    10          0       6.45      0.3

# Reduce to non-zero edges and build a graph object
m182_full_nonzero_edges <- subset(m182_full_data_frame, (friend_tie > 0 | social_tie > 0 | task_tie > 0))
head(m182_full_nonzero_edges)
#     ego alter friend_tie social_tie task_tie
# 5    1     5          0       1.20     0.30
# 8    1     8          0       0.15     0.00
# 9    1     9          0       2.85     0.30
# 10   1    10          0       6.45     0.30
# 11   1    11          0       0.30     0.00
# 12   1    12          0       1.95     0.15

m182_full <- graph.data.frame(m182_full_nonzero_edges)
summary(m182_full)
# IGRAPH 72aa88d DN-- 16 144 -- 
#   + attr: name (v/c), friend_tie (e/n), social_tie (e/n), task_tie (e/n)

plot(m182_full)

# Create sub-graphs based on edge attributes
m182_friend <- delete.edges(m182_full, E(m182_full)[get.edge.attribute(m182_full,name = "friend_tie")==0])
m182_social <- delete.edges(m182_full, E(m182_full)[get.edge.attribute(m182_full,name = "social_tie")==0])
m182_task <- delete.edges(m182_full, E(m182_full)[get.edge.attribute(m182_full,name = "task_tie")==0])

# Look at the plots for each sub-graph
friend_layout <- layout.fruchterman.reingold(m182_friend)
plot(m182_friend, layout=friend_layout, edge.arrow.size=.5)

social_layout <- layout.fruchterman.reingold(m182_social)
plot(m182_social, layout=social_layout, edge.arrow.size=.5)

task_layout <- layout.fruchterman.reingold(m182_task)
plot(m182_task, layout=task_layout, edge.arrow.size=.5)

# COMMUNITY DETECTION
m182_friend_und <- as.undirected(m182_friend, mode='collapse')
m182_friend_no_iso <- delete.vertices(m182_friend_und, V(m182_friend_und)[degree(m182_friend_und)==0])
summary(m182_friend_no_iso)
# IGRAPH d9c98fa UN-- 14 42 -- 
#   + attr: name (v/c)

friend_comm_wt <- walktrap.community(m182_friend_no_iso, step=200, modularity=TRUE)
friend_comm_wt
# IGRAPH clustering walktrap, groups: 3, mod: 0.099
# + groups:
#   $`1`
# [1] "2"  "8"  "13"
# 
# $`2`
# [1] "1"  "3"  "5"  "9"  "10" "12" "15"
# 
# $`3`
# [1] "6"  "7"  "11" "14"

friend_comm_dend <- as.dendrogram(friend_comm_wt, use.modularity=TRUE)
plot(friend_comm_dend)

### 2) Edge Betweenness method
# 그래프에 존재하는 최단거리 중 몇 개가 그 edge(연결, link)를 거쳐가는 지를
# 이용해 edge-betweenness 점수 측정
# 높은 edge-betweenness 점수를 갖는 edge가 클러스터를 분리하는 속성을 가진다고 가정

friend_comm_eb <- edge.betweenness.community(m182_friend_no_iso)
friend_comm_eb
# IGRAPH clustering edge betweenness, groups: 3, mod: 0.28
# + groups:
#   $`1`
# [1] "1"  "9"  "10" "12" "15"
# 
# $`2`
# [1] "2"  "7"  "8"  "13" "14"
# 
# $`3`
# [1] "3"  "5"  "6"  "11"

plot(as.dendrogram(friend_comm_eb))

