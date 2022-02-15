
#===============================================================================================================
# 샤이니 shiny
# 1) R에서 바로 interactive하게 웹 앱(Shiny App) 생성 가능
#    web programming 지식이 전혀 없더라도 interactive한 web graphic을 만들 수 있는 환경 제공
# 2) 이미 구현된 동적 시각화 자료를 웹으로 쉽게 배포 가능
# 3) 사용자와 쉽게 상호작용할 수 있도록 웹 페이지에서 독립형 앱을 호스트하거나,
#    R Markdown 문서에 포함하거나, 대시보드 작성
# 4) CSS 테마, htmlwidgets, JavaScript 작업으로 Shiny App 확장 가능
#
# R Markdown : R에서 사람이 읽을 수 있고 편집 등의 용도로 사용할 수 있도록 문서를 만들어 주는 언어
# R Markdown에서 Document 대신 Shiny 지정하면, Shiny에서도 R Markdown 사용 가능
#
# 샤이니 기본 구조
# 향후 코딩의 유지, 보수가 쉽도록 header, body, footer 구조 (html과 유사)
# 1) headerPanel : 기본적인 제목과 주제
# 2) sidebarPanel : mainPanel에서 control 가능한 component들
#    html 페이지에서 field, button, combo, select box 등이 들어가 mainPanel을 유동적으로 control 하는 것과 유사
# 3) mainPanel : sidebarPanel에서 control component로 조정한 값을 받아 결과화면 출력
#
# ui.R vs server.R
# 샤이니를 실행하려면 ui.R과 server.R 파일이 동일 directory에 있어야 함
# ui.R : 화면 구성과 component class 설정
# server.R : 실제적으로 R에서 구동시킨 코드들이 들어가는 곳
#            각각의 id값을 설정하여 ui.R에 input과 output 값으로 작동
#===============================================================================================================

options(repos=c(RStudio="https://rstudio.org/_packages", getOption('repos')))
install.packages('shiny')
library(shiny)

runExample("01_hello")
runExample("02_text")
runExample("03_reactivity")

### 기본적인 hello_shiny의 ui.R 코드
library(shiny)
shinyUI(pageWithSidebar(
  headerPanel("Hello shiiny!"),                # title
  sidebarPanel(
    sliderInput("obs",                         # input ID : server.R에 input$obs로 전달됨
                "Number of observations: ",    # component로 쓰여 질 text
                min=1,                         # input$obs component의 값은 1~1000, 기본값=500
                max=1000,
                value=500)),
  mainPanel(plotOutput("distPlot"))            # distPlot이라는 이름을 갖는 plot을 output으로 보여줌
))

### 기본적인 hello_shiny의 server.R 코드
# 1) input, output에 대한 함수 생성
# 2) ui.R에서 그려준 sliderInput 객체 안의 input$obs 값 만큼의 표본수를 갖는
#    normal distribution의 random 값들을 dist 변수에 저장
#    이 변수는 R에서 제공하는 hist라는 함수로 보임
# 3) ui.R에서 지정한 plotOutput("distPlot")에서
#    distPlot이라는 이름을 갖는 Plot을 output 값으로 보내 renderPlot 출력
library(shiny)
shinyServer(function(input,output){
  output$distPlot <- renderPlot({
    dist <- rnorm(input$obs)
    hist(dist)
  })
})

### 시작과 종료 ------------------------------------------------------------------------------------------------
# 1) 하나의 R 파일에서 직접 실행 가능하지만, 일반적으로 ui.R과 server.R을 만들어 사용
# 2) 가장 일반적인 방법 : c:/test/shiny로 directory 생성
#    각각의 ui.R과 server.R 파일을 내용에 맞는 폴더에 넣어 관리
# 3) ★ 이때 ui.R과 server.R은 항상 동일 폴더에 있어야 함
# 4) ui.R과 server.R이 들어있는 폴더로 working directory 지정 후 runApp() 실행
# 5) 종료 : esc 키 or 빨간 정지 버튼
#    R server에서 브러우저와 별도로 운영되기 때문에 브라우저 종료 후 반드시 세션 종료할 것
library(shiny)
setwd("C:\\Users\\User\\datasets_for_practice\\shiny\\hello")
runApp()

### Input과 Output ---------------------------------------------------------------------------------------------
# 위 경로에 ui.R, server.R code 파일 각각 저장
# ★ ui.R, server.R 수정 시 세션 반드시 종료 후 재실행

# ui.R code
library(shiny)
shinyUI(pageWithSidebar(
  headerPanel("Miles Per Gallon"),
  sidebarPanel(
    selectInput("variable", "Variable: ",
                list("Cylinders"="cyl",
                     "Transmission"="am",
                     "Geers"="gear")),
    checkboxInput("outliers", "Show outliers", FALSE)
  ),
  #mainPanel()  -> mainPanel이 비면 sidebarPanel에 select combo box, check box 만들어도 출력 안 됨
  mainPanel(h3(textOutput("caption")),
            plotOutput("mpgPlot"))
))

# server.R code
library(shiny)
library(datasets)
mpgData <- mtcars
# mpgData$am : 0,1로 구성 -> Automatic, Manual
mpgData$am <- factor(mpgData$am, labels=c("Automatic","Manual"))

shinyServer(function(input,output){
  formulaText <- reactive({
    paste("mpg ~", input$variable)   # ex) mpg ~ cyl
  })
  output$caption <- renderText({
    formulaText()
  })
  output$mpgPlot <- renderPlot({
    boxplot(as.formula(formulaText()),
            data=mpgData,
            outline=input$outliers)
  })
})

setwd("C:\\Users\\User\\datasets_for_practice\\shiny\\hello")
runApp()

### Slider -----------------------------------------------------------------------------------------------------
# shiny는 html 기반 다양한 component 제공
# 그 중 slider는 내부 속성 값에 따라 다양한 형태
# slider bar의 출력
# - 슬라이드바의 inputID 지정
# - label, min, max, value(기본값), step(bar의 단위), format(데이터형식) 지정
# - ticks : T/F - 슬라이드 바 안의 눈금 표시 여부
# - animate : T/F - 슬라이드바의 움직임

# ui.R code
# ※ locale : 국가 지정 (우리나라 : kr)
library(shiny)
shinyUI(pageWithSidebar(
  headerPanel("Sliders"),
  sidebarPanel(
    sliderInput("integer","integer: ",
                min=0, max=1000, value=500),                     # 0~1000, 기본값 500
    sliderInput("decimal","Decimal: ",
                min=0, max=1, value=.5, step=.1),
    sliderInput("range","Range: ",                               # 특정 구간 default 선택 가능
                min=1, max=1000, value=c(200,500)),
    sliderInput("format","Custom Format: ",                      # 단위, #, 숫자 이용해서 특정 format 지정 가능
                min=0, max=10000, value=0, step=2500,
                animate=TRUE),
    sliderInput("animation","Looping animation: ",1,2000,1,      # 슬라이드 바가자동으로 움직이도록 함
                animate=animationOptions(interval=300, loop=T))  # 옵션으로 구간과 반복 여부 결정
  ),
  mainPanel(
    tableOutput("values")
  )
))

# server.R code
# reactive 함수 : 동적으로 움직이는 부분
library(shiny)
shinyServer(function(input,output){
  SliderValues <- reactive({
    
    # mainPanel에 넣을 테이블을 data.frame으로 만들어 슬라이드 바 값에 따라 변하게 설정
    # 테이블 변수 명, 값, 속성 지정 후 ui.R에서 지정했던
    # integer, decimal, range, format, animation 값들을 character 형태로 넣기
    data.frame(
      Name=c("Integer","Decimal","Range","Custom Format","Animation"),
      Value=as.character(c(input$integer,
                           input$decimal,
                           paste(input$range, collapse=' '),
                           input$format,
                           input$animation)),
      stringAsFactors=FALSE)
    
  })
  
  # 최종 테이블 형태
  # renderTable : slider value의 변하는 값들을 mainPanel 값에 적용
  output$values <- renderTable({
    SliderValues()
  })
})

setwd("C:\\Users\\User\\datasets_for_practice\\shiny\\hello")
runApp()

### Tabsets ----------------------------------------------------------------------------------------------------
# 한 화면에 Tab을 만들어 Tab 별로 다른 그래프나 테이블 보여주기

# ui.R code
library(shiny)
shinyUI(pageWithSidebar(
  
  headerPanel("Tabsets"),
  
  sidebarPanel(
    # radio button
    # 정규분포, 균등분포, 로그노말분포, 지수분포를 선택할 수 있는 버튼
    radioButtons("dist",
                 "Distribution type: ",
                 list("Normal"="norm",
                      "Uniform"="unif",
                      "Log-normal"="lnorm",
                      "Exponential"="exp")),
    br(),
    # 분포 안 랜덤 값 개수 설정
    sliderInput("n",
                "Number of observations: ",
                min=1, max=1000, value=500)
    ),
  
  mainPanel(
    # tabsetPanel : tabPanel이 각각 독립적이도록 만듦
    # tabsetPanel 안에 들어가는 탭마다 Plot, summary, Table이 나타날 수 있음
    tabsetPanel(
      tabPanel("Plot", plotOutput("plot")),
      tabPanel("Summary", verbatimTextOutput("summary")),
      tabPanel("Table", tableOutput("table"))
    )
  )
))

# server.R code
library(shiny)
shinyServer(function(input, output){
  data <- reactive({
    # dist 변수에 switch를 이용해 
    # 정규분포, 균등분포, 로그노말분포, 지수분포를 랜덤하게 n개 생성
    # ui.R에서 받아온 n값 적용
    dist <- switch(input$dist,
                   norm=rnorm,      # 기본 rnorm 문법 : n, mean=0, sd=1
                   unif=runif,
                   lnorm=rnorm,
                   exp=rexp,
                   rnorm)
    dist(input$n)
  })
  output$plot <- renderPlot({
    dist <- input$dist
    n <- input$n
    hist(data(), main=paste('r',dist,'(',n,')',sep=''))
  })
  # 최종값 : data 함수
  # 이를 이용해 histogram, summary, table 그림
  output$summary <- renderPrint({
    summary(data())
  })
})

setwd("C:\\Users\\User\\datasets_for_practice\\shiny\\hello")
runApp()

### dataTable --------------------------------------------------------------------------------------------------
# dataTable : http://datatables.net에서 내놓은 자바스크립트 사용
# ui.R, server.R 코드를 나누지 않고 한 번에 코딩 가능
# but, 코드 관리를 위해 나눠서 작성하는 것이 좋음

# 1. ui.R, server.R code 분리X
# dataTable.R code
# mtcars table을 단순히 dataTable로 웹 브러우저에서 그려주는 것
# 여기 R Script에서 아래 코드 실행하면 됨
# - 페이지 당 보여주는 개수 정하기 가능
# - 특정 레코드 검색해 찾기 가능
# - 변수마다 오름차순/내림차순 정렬 가능
library(shiny)

runApp(list(
  ui=basicPage(
    h2('The mtcars data'),
    dataTableOutput('mytable')
  ),
  server=function(input,output){
    output$mytable=renderDataTable({
      mtcars
    })
  }
))

# 2. ui.R, server.R code 분리

# ui.R code
# checkbox 생성
# 1) checkboxInput : 하나의 체크박스만 생성, 하나의 값만 보여줌
# 2) checkboxGroupInput : list처럼 데이터 속성 값들을 하나의 그룹으로 엮어서 보여줌
library(shiny)
library(ggplot2)
shinyUI(pageWithSidebar(
  headerPanel("Examples of DataTables"),
  sidebarPanel(
    # selected : 기본 값으로 diamonds의 '모든 속성 값' 보여줌
    checkboxGroupInput('show_vars','Columns in diamonds to show:', names(diamonds),
                       selected=names(diamonds)),
    helpText('For the diamonds data, we can select variables to show in the table;
              for the mtcars example, we use bSortClasses = TRUE
              so that sorted columns are colored since they have special CSS classes attached;
              for the iris data, we customize the length menu so we can display 5 rows per page.')
  ),
  mainPanel(
    tabsetPanel(
      tabsetPanel(
        tabPanel('diamonds', dataTableOutput("mytable1")),
        tabPanel('mtcars', dataTableOutput("mytable2")),
        tabPanel('iris', dataTableOutput("mytable3"))
      )
    )
  )
))

# server.R code
library(shiny)
shinyServer(function(input,output){
  output$mytable1 <- renderDataTable({
    library(ggplot2)
    diamonds[,input$show_vars,drop=FALSE]
  })
  # bSortClasses : mytable2에서 값 sorting 시 컬럼 값이 선택되어 나타나도록 함
  output$mytable2 <- renderDataTable({mtcars},
                                     options=list(bSortClasses=TRUE))
  # aLengthMenu : 한 페이지 당 보여줄 레코드 개수 설정
  # iris data 값은 한 화면에 보여줄 레코드 수로 5개, 30개, 50개 중 선택 가능
  # iDisplayLength : 처음 보여주는 레코드 개수
  output$mytable3 <- renderDataTable({iris},
                                     options=list(aLengthMenu=c(5,30,50),
                                                  iDislayLength=5))
})

setwd("C:\\Users\\User\\datasets_for_practice\\shiny\\hello")
runApp()

### MoreWidget -------------------------------------------------------------------------------------------------

# ui.R code
library(shiny)
shinyUI(pageWithSidebar(
  headerPanel("More Widgets"),
  sidebarPanel(
    selectInput("dataset","Choose a dataset: ",
                choices=c("rock","pressure","cars")),
    numericInput("obs", "Number of observations to view: ", 10),
    # helptext : 보조설명, 팁
    helpText("Note: While the data view will show only the specified",
             "number of observations, the summary will still be based",
             "on the full dataset."),
    submitButton("Update View")
  ),
  mainPanel(
    # ※ h1, br 등 html 기본 태그도 잘 사용됨
    # 태그 사용 시 리스트를 만들어 사용하기도 함 : taglist()
    # verbatimTextOutput : R console 창에서 잘 정리된 summary가 나오는 것을 mainPanel에 옮겨줌
    # textOutput : 나오는 값들이 한 줄에 그대로 옮겨짐 (화면이 흐트러질 수 있음)
    h4("Summary"),
    verbatimTextOutput("summary"),
    h4("Observations"),
    tableOutput("view")
  )
))

# server.R code
# Plot, Table, Print, Text은 renderPlot, renderTable, renderPrint, renderText로 보여줌
# 이 과정에서 dataset이나 input 값은 reactive에 정해놓은 function 이름 사용
library(shiny)
shinyServer(function(input,output){
  datasetInput <- reactive({
    switch(input$dataset,
           "rock"=rock,
           "pressure"=pressure,
           "cars"=cars)
  })
  output$summary <- renderPrint({
    dataset <- datasetInput()
    summary(dataset)
  })
  output$view <- renderTable({
    head(datasetInput(), n=input$obs)
  })
})

# dataset 선택 후 update view 버튼 클릭하면 summary와 table 출력됨
# number of observations to view : row 개수 조정
setwd("C:\\Users\\User\\datasets_for_practice\\shiny\\hello")
runApp()

### uploading_Files --------------------------------------------------------------------------------------------
# shiny로 application 생성 시 web에서 data를 불러오고 받을 수 있음

# ui.R code
# csv 파일을 웹으로 불러와 테이블로 표현한 것
library(shiny)
shinyUI(pageWithSidebar(
  headerPanel("CSV viewer"),
  sidebarPanel(
    # fileInput : csv 파일을 불러오는 버튼 생성
    fileInput('file1',
              'Choose CSV File',
              accept=c('text/csv', 'tect/comma-separated-values,text/plain', '.csv')),
    tags$hr(),
    # read.csv 속성에 들어가는 header, sep, quote를 radiobutton과 checkbox로 선택하게 설정
    checkboxInput('header','Header',TRUE),
    radioButtons('sep','Separator',
                 c(Comma=',', Semicolon=';', Tab='\t'),
                 'Comma'),
    radioButtons('quote','Quote',
                 c(None='', 'Double Quote'='"', 'Single Quote'="'"),
                 'Double Quote')
  ),
  # 설정된 파일을 테이블 형태로 출력
  mainPanel(
    tableOutput('contents')
  )
))

# server.R code
# ui.R에서 가져온 파일 경로와 header, separator, quote을 read.csv로 읽기
# renderTable로 그리기
library(shiny)
shinyServer(function(input,output){
  output$contents <- renderTable({
    inFile <- input$file1
    if(is.null(inFile))
      return(NULL)
    read.csv(inFile$datapath, header=input$header, sep=input$sep, quote=input$quote)
  })
})

# fileInput에서 파일 선택
setwd("C:\\Users\\User\\datasets_for_practice\\shiny\\hello")
runApp()

### downloading_Files ------------------------------------------------------------------------------------------
# submit, download button 사용
# downloadHandler를 통해 원하는 파일 저장하기

# ui.R code
library(shiny)
shinyUI(pageWithSidebar(
  headerPanel("Download Example"),
  sidebarPanel(
    # dataset을 선택할 수 있는 selectbox 생성
    # 이를 저장할 수 있는 downloadButton 생성
    selectInput("dataset", "Choose a dataset: ",
                choices=c("rock","pressure","cars")),
    downloadButton('downloadData','Download')),
  mainPanel(
    tableOutput('table')
  )
))

# server.R code
library(shiny)
shinyServer(function(input,output){
  datasetInput <- reactive({
    # rock, pressure, cars의 dataset을 datasetInput 함수로 설정 -> renderTable로 보여줌
    switch(input$dataset,
           "rock"=rock,
           "pressure"=pressure,
           "cars"=cars)
  })
  output$table <- renderTable({
    datasetInput()
  })
  # downloadHandler : datasetInput에서 선택된 dataset을 write.csv 함수로 저장
  # filename : 속성 값에서 csv, txt 형식 지정 가능
  output$downloadData <- downloadHandler(
    filename = function(){
      paste(input$dataset, '.csv', sep='')
    },
    content = function(file){
      write.csv(datasetInput(), file)
    }
  )
})

setwd("C:\\Users\\User\\datasets_for_practice\\shiny\\hello")
runApp()

### HTML_ui ----------------------------------------------------------------------------------------------------
# html 태그로 페이지 만들기 가능
# 1) 'www' 폴더에 두 개 파일 저장하기
# 2) 기본 파일은 ui.R이 아닌 'index.html'
# 3) plot, summary, table 등도 동적으로 적용하기 위해 server.R 생성해야 함

# 아래 코드 실행 안됨

# index.html code
# html 코드에서 sidebarPanel과 mainPanel을 나누지 않아야
# mainPanel에서 볼 수 있었던 summary, histogram, table을 동시에 볼 수 있음

# <!DOCTYPE html>
# <html>
# 
# <head>
#   <script src="shared/jquery.js", type="text/javascript"></script>
#   <script src="shared/shiny", type="text/javascript"></script>
#   <link rel="stylesheet" type="text/css" href="shared/shiny.css"/>
# </head>
# 
# <body>
#   <h1>HTML UI</h1>
#   
#   <p>
#     <label>Distribution type:</label><br />
#     <select name="dist">
#       <option value="norm">Normal</option>
#       <option value="unif">uniform</option>
#       <option value="lnorm">Log-normal</option>
#       <option value="exp">Exponential</option>
#     </select>
#   </p>
#   
#   <p>
#     <label>Number of observations:</label><br />
#     <input type="number" name="n" value="500" min="1" max="1000" />
#   </p>
#   
#   <h3>Summary of data:</h3>
#   <pre id="summary" class="shiny-text-output"></pre>
#   
#   <h3>Plot of data:</h3>
#   <div id="plot" class="shiny-plot-output"
#        style="width: 100%; height: 300px"></div>
#   
#   <h3>Head of data:</h3>
#   <div id="table" class="shiny-html-output"></div>
#          
# </body>
# <html>
         
# server.R code
library(shiny)
server <- function(input, output) {
  d <- reactive({
    dist <- switch(input$dist,
                   norm = rnorm,
                   unif = runif,
                   lnorm = rlnorm,
                   exp = rexp,
                   rnorm)
    dist(input$n)
  })
  
  output$plot <- renderPlot({
    dist <- input$dist
    n <- input$n
    
    hist(d(), main=paste("r",dist,"(",n,")",sep=""),
         col="#75AADB", border="white")
  })
  output$summary <- renderPrint({
    summary(d())
  })
  output$table <- renderTable({
    head(data.frame(x=d()))
  })
}
# ★ shiny App 구동을 위해 이 마지막 줄 코드 반드시 추가해야 함
shinyApp(ui=htmlTemplate("C:\\Users\\User\\datasets_for_practice\\shiny\\www\\index.html"), server)

# 실행
setwd("C:\\Users\\User\\datasets_for_practice\\shiny\\www")
runApp()

