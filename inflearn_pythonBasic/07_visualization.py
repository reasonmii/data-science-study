

'''참고강의
Inflearn 파이썬 무료강의 (기본편) - 6시간 뒤면 나도 개발

데이터 import : 공공데이터포털 활용 -> 다양한 데이터 있음
'''

''' matplotlib
선 색상과 스타일
공식문서 : https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html
공식문서 : https://matplotlib.org/3.2.1/api/pyplot_summary.html

'.' : point marker
'o' : circle marker(plt.plot(x, y_, '-o'))
'^' : triangle_up marker
's' : square marker
'p' : pentagon marker
'*' : star marker
'+' : plus marker
'x' : x marker
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 도표, 그림, 소리, 애니메이션과 같은 결과물들 = Rich output
# Jupyter notebook을 실행한 브라우저에서 바로 Rich output들을
# 확인하고 싶을 때, 아래 문장 꼭 써 주기
%matplotlib inline


### -------------------------------------- < Draw A Line >

### 방법1
x = [100, 200, 300]
y = [1,2,3]
plt.plot(x, y)


### 방법2
value = pd.Series([1,2,3], [100,200,300])
plt.plot(value)


### line stye
plt.plot(x, y, linestyle = 'solid')    # 실선('-')
plt.plot(x, y, linestyle = 'dashed')   # 파선('--')
plt.plot(x, y, linestyle = 'dashdot')  # 1점 쇄선('-.')
plt.plot(x, y, linestyle = 'dotted')   # 점선(':')


### change the color
plt.plot(x, y, color = '0.2')         # 회색조(0-1사이 값)
plt.plot(x, y, color = 'red')         # 색상이름 : red, green, blue 등
plt.plot(x, y, color = 'r')           # r, g, b, y, m(자홍), k(검정) 등 색상 이름

# 16진수 색상값
plt.plot(x, y, color = '#ff0000')     # red
plt.plot(x, y, color = '#ffff00')     # yellow
         

### change the line's width
plt.plot(x, y, color = '#ff0000', linewidth = 10)     # red


### dash lines with color
plt.plot(value, '--r')    # red
plt.plot(value, '--g')    # green
plt.plot(value, '--b')    # blue


### 선 중간중간 모양 넣기
plt.plot(value, '--<')    # 화살표
plt.plot(value, '-->')
plt.plot(value, '--*')    # 별
plt.plot(value, '--o')    # 동그라미


### 점선
plt.plot(value, ':')
plt.plot(value, ':r')    # red



### -------------------------------------- < Write Words >

plt.plot(x, y, color = '#ff0000', linewidth = 10)     # red
plt.title('hello world', fontsize = 20)
plt.xlabel('hello', fontsize = 10)
plt.ylabel('world', fontsize = 10)



### -------------------------------------- < Save the Graph >

plt.savefig('sample.png')



### -------------------------------------- < Draw a graph >

x = np.linspace(0, 10, 100)
y = np.sin(x)
y_ = np.cos(x)
plt.plot(x, y, label = 'sin')
plt.plot(x, y_, '-o', label = 'cos')
# Set the legend's location in the graph
plt.legend(loc = 1)     # right top
plt.legend(loc = 2)     # left top
plt.legend(loc = 3)     # left bottom
plt.legend(loc = 4)     # right bottom



### -------------------------------------- < Scatter >

x = np.linspace(0, 10, 20)
y = x ** 2
plt.scatter(x, y, c = 'r', alpha = 0.5)
# plt.show()



### -------------------------------------- < Pie Chart >

labels = ['one', 'two', 'three']
size = [100, 20, 10]

plt.pie(size,
        labels = labels,
        shadow = True,         # 그림에 shadow
        autopct = '%1.2f%%'    # % 값
        )
plt.show()



### -------------------------------------- < Bar Chart >

### Histogram
x = [np.random.randint(1, 7) for i in range(100)]
x

# bins : box 간 간격
plt.hist(x, bins = 11)
plt.show()


### Normal Bar chart
plt.bar(['one','two','three'], [10, 20, 30])
plt.show()

# 왼쪽에서 오른쪽으로 가는 bar graph
plt.barh(['one','two','three'], [10, 20, 30])
plt.show()



### -------------------------------------- < Plotly >
''' 공식홈페이지 : https://plotly.com/python
Anaconda Python : pip install plotly

단점
plotly graphs and htmlwidgets are not rendered in R notebook
'''

import plotly.offline as pyo
import plotly.express as px

x_ = np.array([1,2,3,4,5])
y_ = x_ ** 2
fig = px.line(x = x_, y = y_)
fig.show()


import plotly.express as px
import plotly.graph_objects as go


# gapminder : 국가별 경제 수준과 의료 수준 동향을 정리한 DataSet
korea_life = px.data.gapminder().query("country == 'Korea, Rep.'")
fig = px.line(korea_life, x = 'year', y = 'lifeExp', title = 'Life expectancy in Korea')
fig.show()        # Jupyter 환경에서는 잘 됨
go.Figure(fig)
pyo.plot(fig)     # fig.show()가 안 되는 경우

korea_life              # dataframe
korea_life["year"]
korea_life["lifeExp"]


### bar graph
korea_gdp = px.data.gapminder().query("country == 'Korea, Rep.'")
fig = px.bar(korea_gdp, x = 'year', y = 'gdpPercap', title = '한국인 GDP')
fig.show()


### scatter chart
korea_data = px.data.gapminder().query("country == 'Korea, Rep.'")
fig = px.scatter(korea_data, x = 'gdpPercap', y = 'lifeExp', title = '한국인 GDP')
fig.show()


### pie chart
fig = px.pie(values = [20, 30, 50])
fig.show()



### -------------------------------------- < 이미지 분석 >

''' 자주 쓰는 3개 library
skimage
PIL : plot과 비슷
cv2 : 많은 영상처리 library가 많이 있음
'''


import numpy as np
from skimage import io
# from PIL import image
# from cv2

import matplotlib.pyplot as plt

pic = io.imread('wallpaper.png')
type(pic)             # imageio.core.util.Array
pic.shape             # (562, 1000, 4)
pic

np.min(pic), np.max(pic)
plt.imshow(pic)


### 방향 바꾸기

# 상하좌우 바꾸기
plt.imshow(pic[::-1])

# 좌우 바꾸기
plt.imshow(pic[:, ::-1])

# 상하 바꾸기
plt.imshow(pic[::-1, :])


### 이미지 자르기
plt.imshow(pic[300:800, :])
plt.imshow(pic[300:800, 500:600])


### 일부분 건너 뛰어서 출력
plt.imshow(pic[::3, ::3])   # 3칸씩 건너뛰기
plt.imshow(pic[::6, ::6])   # 3칸씩 건너뛰기
plt.imshow(pic[::10, ::10])   # 3칸씩 건너뛰기


### 색 변경
# 색의 분포 보기
pic[:, :]
plt.hist(pic.ravel(), 115, [0, 256])
plt.show()

# 50보다 같거나 큰 값에는 0을 넣어라 -> 0 = 검정
pic_ = np.where(pic < 50, pic, 0)
plt.imshow(pic_)

plt.imshow(pic[:, :, 0])
plt.imshow(pic[:, :, 1])
plt.imshow(pic[:, :, 2])
plt.imshow(pic[:, :, 3])


### 회색 계열로 변경
from skimage import color
plt.imshow(color.rgb2gray(pic), cmap = plt.cm.gray)

