
#===============================================================================================================
# Mosaic Plot
# 복수의 categorical variable 분포 파악
# 두 변수의 구조적 특징 파악 가능
#===============================================================================================================

# vcd : visualize categorical data
install.packages("vcd")
library(vcd)

library(datasets)
data(Titanic)
str(Titanic)
# 'table' num [1:4, 1:2, 1:2, 1:2] 0 0 35 0 0 0 17 0 118 154 ...
# - attr(*, "dimnames")=List of 4
# ..$ Class   : chr [1:4] "1st" "2nd" "3rd" "Crew"
# ..$ Sex     : chr [1:2] "Male" "Female"
# ..$ Age     : chr [1:2] "Child" "Adult"
# ..$ Survived: chr [1:2] "No" "Yes"

# 기본형태
# 1등석 여성 승객 생존율이 상대적으로 높음을 알 수 있음
# but, 그 비율의 높고 낮음 여부는 파악이 어려움
mosaic(Titanic)

# 색상 추가 - 비교
# 유의한 집단을 확실히 파악
mosaic(Titanic, shade=T, legend=T)

# 색상 추가 - 특정 집단만
# ★ "" 안에 문구 띄어쓰기 하면 작동 안 함
# - Error in editDLfromGPath(gPath, specs, strict, grep, global, redraw)
strucplot(Titanic, pop=F)
grid.edit("rect:Class=1st,Sex=Male,Age=Adult,Survived=Yes", gp=gpar(fill="red"))
