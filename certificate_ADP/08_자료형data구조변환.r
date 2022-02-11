
#=====================================
# 자료형 data 구조 변환
#
# data 프레임 내용 접근
# with(df, expr)
# attach(df)
# detach(df)
#
# 자료형 변환
# as.character()
# as.complex()
# as.numeric()
# as.double()
# as.integer()
# as.logical()
#
# data 구조 변환
# 벡터 -> 리스트 : as.list(vec)
# 벡터 -> 행렬
#   1열 행렬 : cbind(vec), as.matrix(vec)
#   1행 행렬 : rbind(vec)
#   n*m 행렬 : matrix(vec,n,m)
# 벡터 -> data프레임
#   1열 df : as.data.frame(vec)
#   1행 df : as.data.frame(rbind(vec))
# 리스트 -> 벡터 : unlist(list)
# 리스트 -> 행렬
#   1열 행렬 : as.matrix(lst)
#   1행 행렬 : as.matrix(rbind(lst))
#   n*m 행렬 : matrix(lst,n,m)
# 리스트 -> data프레임
#   목록 원소들이 data의 열인 경우 : as.data.frame(lst)
#   리스트 원소들이 data 행인 경우 : rbind(obs[[1]],obs[[2]])
# 행렬 -> 벡터 : as.vector(mat)
# 행렬 -> 리스트 : as.list(mat)
# 행렬 -> data프레임 : as.dataframe(mat)
# data프레임 -> 벡터
#   1열 df : df[[1]], df[,1]
#   1행 df : df[1,]
# data프레임 -> 리스트 : as.list(df)
# data프레임 -> 행렬 : as.matrix(df)
#=====================================

