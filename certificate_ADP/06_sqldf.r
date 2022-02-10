

#=====================================
# sqldf
# R에서 sql 명령어 사용
# SASㅇ서 proc sql과 같은 역할
#=====================================

install.packages("sqldf")
library(sqldf)

sqldf("select * from [dataframe]")
sqldf("select * from [dataframe] limit 10")
sqldf("select * from [dataframe] where [col] like '%char%'")

# head([df])
sqldf("select * from [df] limit 6")

# subset([df], grep1("qn%",[col]))
sqldf("select * from [df] where [col] like 'qn%'")

# subset([df], [col] %in% c("BF","HF"))
sqldf("select * from [df] where [col] in ('BF','HF')")

# rbind([df1],[df2])
sqldf("select * from [df1] union all select * from [df2]")

# merge([df1],[df2])
sqldf("select * from [df1], [df2]")

# df[order([df]$[col], decreasing=T),]
sqldf("select * from [df] order by [col] desc")

# iris dataset example
sqldf("select * from iris")

