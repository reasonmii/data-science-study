#=====================================
# 함수 적용
#
# 리스트의 각 원소에 함수 적용
# lapply(결과를 리스트 형태로 반환) : list <- lapply(l, func)
# sapply(결과를 벡터 또는 행렬로 반환) : vec <- sapply(l, func)
#
# 행렬에 함수 적용
# m <- apply(m1, 1, func)
#
# dataframe에 함수 적용
# df <- lapply(df, func)
# df <- sapply(df, func)
# df <- apply(df, func)
# ※ appply : dataframe이 동질적인 경우만(모두 숫자, 문자) 가능
# dataframe을 행렬로 변환 후 함수 적용
#
# 대용량 data의 함수 적용 : 다중회귀분석
# sapply를 통한 간단한 R coding
# 1) 타겟변수와 상관계수 구하기
#    cors <- sapply(df, cor, y=target)
# 2) 상관계수가 높은 상위 10개의 변수를 입력변수로 선정
#    mask <- (rank(-abs(cors))<=10)
#    best.pred <- df[,mask]
# 3) 타겟변수와 입력변수로 다중회귀분석 실시
#    lm(target~bes.pred)
#
# 집단별 함수 적용
# tapply(vec, factor, func)
#
# 행집단 함수 적용
# by(df, factor, func)
#
# 병렬 벡터, 리스트 함수 적용
# mapply(factor, vec1, vec2, ...)
# mapply(vector, list1, list2, ...)
#=====================================
