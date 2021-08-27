

'''참고강의
Inflearn 파이썬 무료강의 (기본편) - 6시간 뒤면 나도 개발
'''

'''
(필수) pip3 install BeautifulSoup4 or pip3 install bs4
(필수) pip3 install requests
(필수) pip3 install pandas
(필수) pip3 install plotly
(선택) pip3 install lxml
'''

''' URL : Uniform Resource Locator
1) 자원이 어디 있는지 알려주기 위한 규약
2) 흔히 웹 사이트 주소로 알고 있지만, URL은 웹 사이트 주소뿐만 아니라
   컴퓨터 네트워크상의 자원을 모두 나타낼 수 있음
3) 그 주소에 접속하려면 해당 URL에 맞는 프로토콜을 알아야 하고,
   그와 동일한 프로토콜로 접속
   - FTP 프로토콜인 경우 FTP 클라이언트 이용
   - HTTP인 경우 웹 브라우저 이용
   - 텔넷의 경우 텔넷 프로그램을 이용해서 접속
   
ex) 시작메뉴 : Windows PowerShell
- 검색 : nslookup www.naver.com
- 결과 : Addresses:  125.209.222.141
- 인터넷 주소 창에 해당 주소 검색하면 URL로 바뀌면서 네이버 연결 됨
- 즉, 우리가 이런 주소가 아닌 좀 더 쉽게 웹 서비스에 접속하기 위해 URL을 사용하는 것
'''

''' HTTP : Hypertext Transger Protocol
1) HTML, XML, Javascript, 오디오, 비디오, 이미지, PDF 등을 서비스하기 위한 protocol
2) 구성 : 요청 또는 상태 라인 / 해더(생략가능) / 빈줄(헤더의 끝) / 바디 (생략가능)

#요청
GET /stock.html HTTP/1.1
Host www.paullab.co.kr

# 응답
HTTP/1.1 200 OK                                      ## 상태라인
Content-Type: application/xhtml+xml; charset=utf-8   ## 해더
                                                     ## 빈줄
<html>                                               ## 바디
...
</html>

네이버 페이지 상태에서 ctrl + U 클릭하면
이와 같은 URL 볼 수 있음


HTTP 처리방식
1) GET : 리소스 취득 (? 뒤에 이어붙이는 방식 - 작은 값들)
2) POST : 리소스 생성 (Body에 붙이는 방식 - 상대적으로 큰 용량, 주로 이미지나 동영상)
3) PUT, DELETE, HEAD, OPTIONS, TRACE, CONNECT


HTTP 상태코드
1XX (정보) : 요청을 받았으며 프로세스를 계속한다
2XX (성공) 요청을 성공적으로 받았으며 인식했고 수용했다
3XX (redirection) : 요청 완료를 위해 추가 작업 조치가 필요하다
4XX (client 오류) : 요청의 문법이 잘못되었거나 요청을 처리할 수 없다
5XX (server 오류) : 서버가 명백히 유효한 요청에 대해 충족을 실패했다

ex)
200 : 서버가 요청을 제대로 처리함
201 : 성공적으로 요청되었으며 서버가 새 리소스를 작성함
202 : 서버가 요청을 접수했지만 아직 처리하지 않음
301 : 요청한 페이지를 새 위치로 영구적으로 이동함
403 : Forbidden
404 : Not Found
500 : 내부 서버 오류
503 : 서비스를 사용할 수 없음
'''

import requests
import bs4

requests.__version__
bs4.__version__


### crawling 연습용 데이터 : http://paullab.co.kr/stock.html
# http://paullab.co.kr -> Services -> 크롤링 지원

html = requests.get('http://www.paullab.co.kr/stock.html')
html
# <Response [200]>
# 200은 성공했다는 의미

# 한글이 깨져서 나옴
html.text
html.headers

# 한글 깨지지 않게 나옴
html.encoding
html.encoding = 'utf-8'
html.text

html.status_code    # 200
html.ok             # 제대로 접속이 되었는가 : ok



### -------------------------------------- < html script를 editor로 불러오기 >

import requests
from bs4 import BeautifulSoup

response = requests.get('http://www.paullab.co.kr/stock.html')
response.encoding = 'utf-8'
html = response.text

# html 문자형식으로 보기 좋게 출력하기
soup = BeautifulSoup(html, 'html.parser')
print(soup.prettify())


'''
위와 같이 파이썬으로 불러오면 형식이 분석하기 쉽지 않기 때문에 editor 활용
정보보안에서 악성 script 분석 할 때도 직접 홈페이지로 들어갈 수 없으니
이렇게 다운로드 받아서 script 보는 경우 많음

editor : 주로 Atom or Visual Studio 사용

해당 프로그램에서 제공하는 기능들도 활용할 수 있어서 더 좋음
'''

# 1) 임시 test html 파일 만들기
# 2) 분석하고 싶은 url 코드 저장
# 2) 만들어진 파일 우클릭 - 연결프로그램 - Atom or Visual Studio
f = open('test.html', 'w', encoding = 'utf-8')
html = requests.get('http://www.paullab.co.kr/stock.html')
f.write(html)
f.close()
!dir



### -------------------------------------- < Beautiful Soup >

# 띄어쓰기 단위로 검색하기
s = html.split(' ')
word = input('페이지에서 검색할 단어를 입력하세요 : ')
s.count(word)
# 제주 입력 시 '제주'라는 단어가 많이 있지만 0개로 출력됨
# split(' ') 때문
# 앞뒤로 띄어쓰기가 안 되어 있으면 해당 단어를 인식하지 못함
# 이와 같은 문제점을 beautifulsoup에서는 간단하게 해결


''' BeautifulSoup
1) str타입의 html 데이터를 html 구조를 가진 데이터로 가공해주는 library
2) BeautifulSoup(markup, "html.parser")
3) BeautifulSoup(markup, "lxml")
4) BeautifulSoup(markup, "lxml-xml") BeautifulSoup(markup, "xml")
5) BeautifulSoup(markup, "html5lib")
'''

import requests
from bs4 import BeautifulSoup

response = requests.get('http://www.paullab.co.kr/stock.html')
response.encoding = 'utf-8'
html = response.text

soup = BeautifulSoup(html, 'html.parser')

# title 바로 접근
soup.title          # <title>Document</title>
soup.title.string   # 'Document'
soup.title.text     # 'Document'

soup.title.parent.name   # 'head'


### first table row
soup.tr
#<tr>
#<th class="strong" scope="row">시가총액</th>
#<!-- 공백은 의도적으로 넣은 것입니다. -->
#<td class="strong"><em id="_market_sum">349조 2,323</em>억원</td>
#</tr>

# table head
soup.th
#<th class="strong" scope="row">시가총액</th>


### first table data
soup.td
#<td class="strong"><em id="_market_sum">349조 2,323</em>억원</td>



''' 우리가 분석하고 싶은 부분에 대해 웹 상에서 코드 찾아내기
1) ctrl + shift + I
2) ctrl + shift + C
3) 이후 웹사이트 부분에서 내가 원하는 데이터 위에 마우스를 놓으면 그 데이터 부분의 html 코드가 보여짐
   각각의 데이터가 어디서 어떤 tag로 출력되는 지 코드 부분 알 수 있음
4) 내가 crawling 하고 싶은 부분 찾기
'''

# 이렇게 출력을 하면 분석하기가 어려움
# table 전체가 나오기 때문
soup.table

soup.find('title')   # <title>Document</title>

soup.find('tr')
#<tr>
#<th class="strong" scope="row">시가총액</th>
#<!-- 공백은 의도적으로 넣은 것입니다. -->
#<td class="strong"><em id="_market_sum">349조 2,323</em>억원</td>
#</tr>

soup.find('th')
#<th class="strong" scope="row">시가총액</th>

soup.find(id = ('update'))
#<span id="update">update : 20.12.30 / 해외 크롤링이 Block되어 있으므로
# 크롤링이 안되시는 분은 이 URL(http://paullab.synology.me/stock.html)을 사용하세요.</span>

soup.find(id=('update')).text
#'update : 20.12.30 / 해외 크롤링이 Block되어 있으므로
# 크롤링이 안되시는 분은 이 URL(http://paullab.synology.me/stock.html)을 사용하세요.'

soup.find('head').find('title')
# <title>Document</title>

soup.find('h2', id = '제주코딩베이스캠프연구원')
# <h2 id="제주코딩베이스캠프연구원">제주코딩베이스캠프 연구원</h2>

# 문서에서 모든 heading2 출력하기
soup.find_all('h2')
#[<h2>(주)캣네생선</h2>,
# <h2 id="제주코딩베이스캠프연구원">제주코딩베이스캠프 연구원</h2>,
# <h2 id="제주코딩베이스캠프공업">제주코딩베이스캠프 공업</h2>,
# <h2 id="제주코딩베이스캠프출판사">제주코딩베이스캠프 출판사</h2>,
# <h2 id="제주코딩베이스캠프학원">제주코딩베이스캠프 학원</h2>]
soup.find_all('h2')[0]
# <h2>(주)캣네생선</h2>

# 각각의 table 모두 찾기
# class는 예약어이기 때문에 under bar 넣기
soup.find_all('table', class_='table')



### -------------------------------------- < tag >

soup = BeautifulSoup('''
                     <yuna id = 'stndford' class = 'graduateSchool codingLevelUp'>
                         hello world
                     </yuna>
                     ''')

tag = soup.yuna
tag
#<yuna class="graduateSchool codingLevelUp" id="stndford">
#                         hello world
#                     </yuna>
type(tag)    # bs4.element.Tag

# 사용할 수 있는 method 모두 출력
dir(tag)

tag.name        # 'yuna'
tag['class']    # ['graduateSchool', 'codingLevelUp']
tag['id']       # 'stndford'

# tag 모든 정보 보기
tag.attrs
# {'id': 'stndford', 'class': ['graduateSchool', 'codingLevelUp']}

tag.string
tag.text
# '\n                         hello world\n                     '

tag.contents
# ['\n                         hello world\n                     ']


tag.children      # <list_iterator at 0x259eff360b8>

for i in tag.children:
    print(i)
# hello world


###    
### replace, remove를 활용해서 거슬리는 부분 제거 가능
soup = BeautifulSoup('''
                     <ul>
                         <li id = 'standford' class = 'graduateSchool codingLevelUp'>hello world</li>
                         <li id = 'standford' class = 'graduateSchool codingLevelUp'>hello world</li>
                         <li id = 'standford' class = 'graduateSchool codingLevelUp'>hello world</li>
                     </ul>
                     ''')

tag = soup.ul
tag
#<ul>
#<li class="graduateSchool codingLevelUp" id="standford">hello world</li>
#<li class="graduateSchool codingLevelUp" id="standford">hello world</li>
#<li class="graduateSchool codingLevelUp" id="standford">hello world</li>
#</ul>

tag.contents
#['\n',
# <li class="graduateSchool codingLevelUp" id="standford">hello world</li>,
# '\n',
# <li class="graduateSchool codingLevelUp" id="standford">hello world</li>,
# '\n',
# <li class="graduateSchool codingLevelUp" id="standford">hello world</li>,
# '\n']

tag.contents[1]
# <li class="graduateSchool codingLevelUp" id="standford">hello world</li>

tag.li
# <li class="graduateSchool codingLevelUp" id="standford">hello world</li>

tag.li.parent
#<ul>
#<li class="graduateSchool codingLevelUp" id="standford">hello world</li>
#<li class="graduateSchool codingLevelUp" id="standford">hello world</li>
#<li class="graduateSchool codingLevelUp" id="standford">hello world</li>
#</ul>



### -------------------------------------- < Selector >
'''
1) 태그에 좀 더 세밀한 접근 가능
2) class 지칭할 때 : '.' 사용
3) id 지칭할 때 : '#' 사용
4) 탐색하고자 하는 태그가 특정태그 하위에 있을 때 '>' 사용
'''

import requests
from bs4 import BeautifulSoup

response = requests.get('http://www.paullab.co.kr/stock.html')
response.encoding = 'utf-8'
html = response.text

soup = BeautifulSoup(html, 'html.parser')

soup.select('#update')
# [<span id="update">update : 20.12.30 /
# 해외 크롤링이 Block되어 있으므로 크롤링이 안되시는 분은
# 이 URL(http://paullab.synology.me/stock.html)을 사용하세요.</span>]

            
### 'table' class 안에 tbody 안에 모든 tr 태그 출력
soup.select('.table > tr')
# []
# 다시 웹사이트로 들어가서 코드를 보면
# table 및에 tbody가 있어서 바로 trow 접근이 불가능하다는 것 알 수 있음
# 따라서 아래와 같이 수정해줘야 함
soup.select('.table > tbody > tr')

soup.select('.table > tbody > tr')[0]
#<tr>
#<th scope="col">날짜</th>
#<th scope="col">종가</th>
#<th scope="col">전일비</th>
#<th scope="col">시가</th>
#<th scope="col">고가</th>
#<th scope="col">저가</th>
#<th scope="col">거래량</th>
#</tr>


### select가 지원하는 양식
### CSS에서 지원하는 요소 선택 방법을 대부분 허용하고 있음
soup.select("p > a:nth-of-type(2)")    # p tag 안 a tag 안 2번째 요소
soup.select("p > a:nth-child(even)")   # p tag 안 짝수 요소
soup.select("a[href]")                 # 특정 attribute 가진 요소
soup.select("#link1 + .sister")        # ID, Class 동시에 가진 요소

            
### 특정 부분만 가져오기
soup.select('.table > tbody > tr')[0]
oneStep = soup.select('.main')[2]
twoStep = oneStep.select('tbody > tr')[1:]

# table data
twoStep[0].select('td')[0]
# <td align="center "><span class="date">2019.10.23</span></td>
twoStep[0].select('td')[0].text    # '2019.10.23'


### 문자데이터를 숫자로 변환
twoStep[0].select('td')[1]
# <td class="num"><span>6,650</span></td>
twoStep[0].select('td')[1].text                       # '6,650'
int(twoStep[0].select('td')[1].text.replace(',',''))  # '6650'


date = []
endPrice = []

for i in twoStep:
    date.append(i.select('td')[0].text)
    endPrice.append(int(i.select('td')[1].text.replace(',', '')))

date
#['2019.10.23',
# '2019.10.22',
# '2019.10.21',
# '2019.10.18',
# '2019.10.17',
# '2019.10.16',
# '2019.10.15',
# '2019.10.14',
# '2019.10.11',
# '2019.10.10',
# '2019.10.08',
# '2019.10.07',
# '2019.10.04',
# '2019.10.02',
# '2019.10.01',
# '2019.09.30',
# '2019.09.27',
# '2019.09.26',
# '2019.09.25',
# '2019.09.24']

endPrice
#[6650,
# 6630,
# 6820,
# 6430,
# 5950,
# 5930,
# 5640,
# 5380,
# 5040,
# 5100,
# 5050,
# 4940,
# 5010,
# 4920,
# 5010,
# 5000,
# 5010,
# 5060,
# 5060,
# 5330]



### -------------------------------------- < Visualization >
# plotly graphs and htmlwidgets are not rendered in R notebook
import plotly.express as px

fig = px.line(x = date, y = endPrice, title = '날짜 별 종가값')
fig.show()



### -------------------------------------- < Quiz >
### 문제1 : 각 회사별 1만주씩 있다고 가정했을 때, 전그룹사 시가총액은?
import requests
from bs4 import BeautifulSoup

response = requests.get("http://www.paullab.co.kr/stock.html")

response.encoding = 'utf-8'
html = response.text

soup = BeautifulSoup(html, 'html.parser')


### 데이터탐색
soup.select('.main')[0]
soup.select('.main')[1]
soup.select('.main')[2]
soup.select('.main')[3]
soup.select('.main')[4]


### 데이터탐색
int(soup.select('.main')[1].select('.table > tbody > tr')[1].select('td > span')[1].text.replace(',',''))
soup.select('.main')[2].select('.table > tbody > tr')[1].select('td > span')[1].text
soup.select('.main')[3].select('.table > tbody > tr')[1].select('td > span')[1].text
soup.select('.main')[4].select('.table > tbody > tr')[1].select('td > span')[1].text
soup.select('.main')[5].select('.table > tbody > tr')[1].select('td > span')[1].text


### 전그룹사 시가총액 계산하기
오늘종가 = []
그룹사별일일시가 = soup.select('.main')[2:6]

for i in 그룹사별일일시가:
    오늘종가.append((int(i.select('.table > tbody > tr')[1].select('td > span')[1].text.replace(',',''))))

오늘시가총액 = [i * 10000 for i in 오늘종가]
format(sum(오늘시가총액), ',')
# 전그룹사시가총액 : '538,000,000'


### 문제2 : 전그룹사 시가총액 추이를 그래프로 그려주세요. x축은 날짜, y축은 가격입니다.
오늘종가 = []
그룹사별일일시가 = soup.select('.main')[2:6]
오늘시가총액 = []

for j in range(1, len(soup.select('.main')[2].select('.table > tbody > tr'))):
    오늘종가 = []
    for i in 그룹사별일일시가:
        오늘종가.append((int(i.select('.table > tbody > tr')[j].select('td > span')[1].text.replace(',',''))))
    오늘시가총액.append(sum(오늘종가))

오늘시가총액 = [i * 10000 for i in 오늘시가총액]

# 날짜가 적혀있는 코딩 부분 class 값 : date 입력하면 됨
날짜 = soup.select('.main')[2].select('.table > tbody > tr > td > .date')

date = []
for i in 날짜:
    date.append(i.text)

date

# visualization
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(date, 오늘시가총액) 
plt.xticks(rotation = -45)                # x축 label 기울이기
plt.show()

# 그래프 뒤집어야 할 때
# ex) 가장 최신 데이터가 가장 마지막에 있는 경우
plt.plot(date[::-1], 오늘시가총액[::-1])  
plt.xticks(rotation = -45) 
plt.show()


### 문제3 : 각 그룹사별 거래량 추이를 그래프로 그려주세요. 전체 그룹사 총량도 함께 표시해주세요.

# 데이터탐색
soup.select('.main')[2].select('.table > tbody > tr')[1].select('td > span')[-1].text.replace(',','')
soup.select('.main')[2].select('.table > tbody > tr')[2].select('td > span')[-1].text.replace(',','')
soup.select('.main')[2].select('.table > tbody > tr')[3].select('td > span')[-1].text.replace(',','')
soup.select('.main')[2].select('.table > tbody > tr')[20].select('td > span')[-1].text.replace(',','')


그룹사별거래량 = []
그룹사별전체거래량 = []

for i in soup.select('.main')[2:6]:
    거래량 = []
    for j in range(1, len(soup.select('.main')[2].select('.table > tbody > tr'))):
        거래량.append(int(i.select('.table > tbody > tr')[j].select('td > span')[-1].text.replace(',','')))
    그룹사별거래량.append(거래량)
    
for i in range(len(그룹사별일일거래량[0])):
    s = 0
    for j in range(4):
        s += 그룹사별일일거래량[j][i]
    그룹사별전체거래량.append(s)

#
%matplotlib inline
import matplotlib.pyplot as plt

f = plt.figure(figsize = (10,3))
ax = f.add_subplot(1, 2, 1)        # 1행 2열 subplot을 만들고 그 중 1번째
ax.plot(date[::-1], 그룹사별거래량[0][::-1], label = 'A')
ax.plot(date[::-1], 그룹사별거래량[1][::-1], label = 'B')
ax.plot(date[::-1], 그룹사별거래량[2][::-1], label = 'C')
ax.plot(date[::-1], 그룹사별거래량[3][::-1], label = 'D')
plt.xticks(rotation = -45)
ax.legend(loc = 2)
ax2 = f.add_subplot(1, 2, 2)
ax2.plot(date[::-1], 그룹사별전체거래량[::-1], label = 'ALL')
plt.xticks(rotation = -45)
ax2.legend(loc = 2)
plt.show()
