
#===============================================================================================================
# 샤이니 shiny
# R 분석결과를 바로 publishing
# web programming 지식이 전혀 없더라도 interactive한 web graphic을 만들 수 있는 환경 제공
#===============================================================================================================

options(repos=c(RStudio="https://rstudio.org/_packages", getOption('repos')))
install.packages('shiny')
library(shiny)

runExample("01_hello")
runExample("02_text")
runExample("03_reactivity")


