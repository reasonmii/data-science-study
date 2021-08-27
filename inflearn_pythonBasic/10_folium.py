

'''참고강의
Inflearn 파이썬 무료강의 (기본편) - 6시간 뒤면 나도 개발
'''

''' Jupyter에서 folium 사용하면 바로 map 보임

활용데이터
공공데이터포털 : 오름 검색
'제주특별자치도_오름현황' 다운로드
'''

# 

import pandas as pd
import numpy as np
import folium

# 한글깨짐 해결
# https://desktop.github.com/ 먼저 설치 필요
# exe 파일 우클릭 > 속성 > 위치 : 경로복사
# 내컴퓨터 우클릭 > 속성 > 고급시스템설정 > 환경변수
# 시스템변수의 path 항목에 복사한 경로 추가

# anaconda prompt 우클릭 > 관리자 권한으로 열기
# conda update conda
# conda install git
# !pip install git+https://github.com/python-visualization/branca.git@master

import warnings
warnings.filterwarnings(action = 'ignore')

# location : 좌표 넣기
# Google map 활용 > 원하는 부분 마우스 우클릭 > 좌표 클릭 = 복사
# 제주도 오름 : 33.37329, 126.53191
m = folium.Map(location = [33.37329, 126.53191])
m


### 마우스를 지도 위에 올리면 Marker 뜨게 하기
tooltip = "안녕하세요!"
folium.Marker(
        [33.37329, 126.53191],
        tooltip = tooltip
        ).add_to(m)
m


### html 파일로 지도 표시하기
m.save('map.html')


### 팝업창 띄우기
m = folium.Map(location = [33.37329, 126.53191])
tooltip = "클릭해보세요"

folium.Marker(
        [33.37329, 126.53191],
        popup = '<strong>한라산</strong>',  # 마커 클릭 시 표시될 문구
        tooltip = tooltip
        ).add_to(m)

folium.Marker(
        [33.507796, 126.492815],
        popup = '<strong>제주국제공항</strong>',
        tooltip = tooltip
        ).add_to(m)

m.save('map.html')


### 아이콘 변경하기
folium.Marker(
        [33.37329, 126.53191],
        popup = '<strong>한라산</strong>',
        icon = folium.Icon(color = "##ff0000", Icon = 'info-sign'),  # 빨간색 아이콘
        tooltip = tooltip
        ).add_to(m)

m.save('map.html')


### 위도, 경도가 있는 DATA FRAME 활용해서 지도 만들기
import pandas as pd

# cp949, euc-kr : 한글 인코딩 방식
# cp949는 euc-kr의 확장버전
# 코드에러 발생 시 파일명 간단하게 변경해보기
오름 = pd.read_csv('제주오름.csv', encoding = 'cp949')
오름.head()
오름.tail()

m = folium.Map(location = [33.37329, 126.53191])
tooltip = "클릭해보세요"

for i in range(len(오름)):
    folium.Marker(
            [오름.iloc[i]['위도'], 오름.iloc[i]['경도']],
            popup = folium.Popup(f'<strong>{오름.iloc[i]["오름명"]}</strong>', max_width = 500),
            tooltip = tooltip
            ).add_to(m)

m.save('map.html')


### 특정 지역 원으로 표시하기
folium.Circle(
        [33.37329, 126.53191],
        radius = 10000,
        color = 'red',
        fill = True
        ).add_to(m)

m.save('map.html')


### A 지역에서 B 지역까지 선 그리기

# ex. 대정읍 위도 경도 : [33.2043, 126.282]
오름.iloc[1]

folium.PolyLine(
        locations = [[33.37329, 126.53191], [33.2043, 126.282]],
        tooltip = '한라산에서 대정읍까지'
        ).add_to(m)

m.save('map.html')

