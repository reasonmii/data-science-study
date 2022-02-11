
#=====================================
# dataframe
#
# 벡터들로 dataset 생성
# data.frame(벡터,벡터,벡터)
#
# 열 data로 data 프레임 생성
# df <- data.frame(v1,v2,v3,f1,f2)
# df <- as.data.frame(list.of.vectors)
#
# dataframe 할당
# N = 1,000,000
# dt <- data.frame(dosage=numeric(N),
#    lab=character(N),
#    respones=numeric(N))
#
# dataframe 조회
# df[df$gender="m"]
# df[df$변수1 > 4 & df$변수2 > 5, c(변수3, 변수4)]
# df[grep("문자", df$변수1, ignore.case=T), c("변수2","변수3")]
#   -> dataset 변수1 내 "문자"가 들어간 케이스들의 변수2, 변수3 값 조회
#
# dataset 조회
# subset(df, select=변수, subset=변수>조건)
#
# data 선택
# lst[[2]], lst[2], lst[2,], lst[,2]
# lst[["name"]], lst$name
# lst[c("name1","name2","name3")]
#
# data 병합
# merge(df1, df2, by="공통 열이름")
# merge(df1, df2, by="열이름", all=T)
#
# 열이름 조회
# colnames(변수)
#
# 행/열 선택
# subset(df, select=열이름)
# subset(df, select=c(열1,열2,열3))
# subset(df, select=열이름, subset=(조건))
#
# 이름으로 열 제거
# subset(df, select=-"열이름")
#
# 열 이름 변경
# colnames(df) <- newnames
#
# NA 포함 행 삭제
# a <- na.omit(df)
#=====================================

a <- data.frame(a=1,b=2,c=3,d='a')
#   a b c d
# 1 1 2 3 a

b <- data.frame(a=4,b=5,c=6,d='b')

# dataset 행결합
rbind(a,b)
#   a b c d
# 1 1 2 3 a
# 2 4 5 6 b

# dataset 열결합
cbind(a,b)
#   a b c d a b c d
# 1 1 2 3 a 4 5 6 b


