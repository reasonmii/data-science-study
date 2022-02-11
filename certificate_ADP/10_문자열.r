
# 길이
nchar("ABC")   # 3

# 문자 붙이기
A <- paste("a","b","c",sep="-")  # "a-b-c"
paste(A,c("e","f"))              # "a-b-c e" "a-b-c f"
paste(A,10,sep="")               # "a-b-c10"

paste("abc","def",sep="-")  # "abc-def"
paste("the pi is approximately", pi)
paste(v1, "loves me", collapse=", and ")
# "1 loves me, and 2 loves me, and 3 loves me, and 4 loves me"

# 하위 문자열 추출
substr("Bigdataanalysis",1,4)  # Bigd
substr("statistics",1,4)  # stat

# 구분자로 문자열 추출
strsplit("abc,def",",")   # "abc" "def"

# 하위 문자열 대체
# sub(old, new, string)
# gsub(old, new, string)

# 쌍별 조합
# mat <- outer(문자열1, 문자열2, paste, sep="")

