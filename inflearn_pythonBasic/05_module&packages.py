
'''참고강의
Inflearn 파이썬 무료강의 (기본편) - 6시간 뒤면 나도 개발
'''

### -------------------------------------- < Module >
# 필요한 부품인 함수정의, class 등을 담고 있는 file
# 파일 확장자 : .py

# 모듈을 사용하려면
# 모듈과 모듈을 쓰려는 파일이 같은 경로에 있거나
# Python library가 모여있는 폴더에 모듈이 있어야 함

# file path 잘 설정되어 있는지 확인
# 현재 파일과 모듈파일이 모두 있는 곳으로 설정할 것

import theater_module
theater_module.price(3)          # 3명 가격은 30000원 입니다.
theater_module.price_morning(4)  # 4명 조조 할인 가격은 24000원 입니다.
theater_module.price_soldier(5)  # 5명 군인 할인 가격은 20000원 입니다.

# Simplify module name
import theater_module as mv
mv.price(3)
mv.price_morning(4)
mv.price_soldier(5)

# To use all the things in the module
from theater_module import *
price(3)
price_morning(4)
price_soldier(5)

# Import specific functions
from theater_module import price, price_morning
price(3)
price_morning(4)
price_soldier(5)        # Error

from theater_module import price_soldier as ps
ps(5)                   # 5명 군인 할인 가격은 20000원 입니다.


### -------------------------------------- < Package >
# Module의 집합

# Create a New Folder - Travel
# Create files under the folder
# File1 name : __init__.py
# Other files : Create whatever you want

import travel.thailand
trip_to = travel.thailand.ThailandPackage()
trip_to.detail()
# [태국 패키지 3박 5일] 방콕, 파타야 여행 (야시장 투어) 50만원

from travel.thailand import ThailandPackage
# travel.thailand.ThailandPackage     -> 이런 식으로 바로 가져오기는 불가능
trip_to = ThailandPackage()
trip_to.detail()
# [태국 패키지 3박 5일] 방콕, 파타야 여행 (야시장 투어) 50만원

from travel import vietnam
trip_to = vietnam.VietnamPackage()
trip_to.detail()
# [베트남 패키지 3박 5일] 다낭 효도 여행 60만원


### __all__
# __init__ 파일에 아래와 같이 작성하면 from travel만으로도 vietnam 불러오기 가능
# __all__ = ["vietnam"]

from travel import *
trip_to = vietnam.VietnamPackage()
trip_to.detail()

trip_to = thailand.ThailandPackage()
trip_to.detail()
# NameError: name 'thailand' is not defined
# __init__ 파일 내에 thailand는 정의하지 않았기 때문

# tahiland도 호출하기 위해 __init__ 파일에 아래와 같이 작성
from travel import *
trip_to = thailand.ThailandPackage()
trip_to.detail()


### -------------------------------------- < Find the location of package & module >

# Find where the 'random module' is
import inspect
import random
print(inspect.getfile(random))
# C:\ProgramData\Anaconda3\lib\random.py

print(inspect.getfile(thailand))
# D:\000tmp\Data_Science\python_OnlineClass_kr\travel\thailand.py

# 모듈 직접 만들고 같은 폴더 경로에 있는 파일이 아니라도 어디서든 쓰고 싶으면
# 기본 모듈들이 있는 이 경로에 모듈 넣기
# C:\ProgramData\Anaconda3\lib\random.py


### -------------------------------------- < pip install >
# Google : pypi 검색
# Pypi.org
# 파이썬 내 패키지 모두 있음 + 설명, 검색도 가능

from bs4 import BeautifulSoup
soup = BeautifulSoup("<p>Some<b>bad<i>HTML")
print(soup.prettify())


### Anaconda Prompt
# To see all the packages installed : pip list
# The information of the package : pip show beautifulsoup4
# To update the package : pip install --upgrade beautifulsoup4
# To delete the package : pip uninstall beautifulsoup4


### -------------------------------------- < 내장함수 >
# Google : list of python builtins 검색

### input : 사용자 입력을 받는 함수
language = input("무슨 언어를 좋아하세요?")
print("{}은 아주 좋은 언어입니다!".format(language))


### dir : 어떤 객체를 넘겨줬을 때 그 객체가 어떤 변수와 함수를 가지고 있는지 표시
print(dir())    # 현재 파일에서 쓸 수 있는 모든 함수 호출

# random 함수에서 사용할 수 있는 변수들
import random         # random : 외장 함수
print(dir(random))

# list에서 쓸 수 있는 변수들
lst = [1,2,3]
print(dir(lst))

# string에서 쓸 수 있는 변수들
name = "Jim"
print(dir(name))


### -------------------------------------- < 외장함수 >
# Google : list of python modules 검색

### glob : 경로 내의 폴더, 파일 목록 조회 (윈도우 dir)
# 현재 경로에서 확장자가 py인 모든 파일 검색
import glob
print(glob.glob("*.py"))    


### os : 운영체제에서 제공하는 기본 기능
import os

# 현재 디렉토리
print(os.getcwd())

folder = "sample_dir"

if os.path.exists(folder):
    print("이미 존재하는 폴더입니다.")
    os.rmdir(folder)
    print(folder, "폴더를 삭제하였습니다.")
else:
    os.makedirs(folder)    # 폴더 생성
    print(folder, "폴더를 생성하였습니다.")

# 현재 경로 내에 존재하는 모든 폴더, 파일
print(os.listdir())


### time : 시간 관련 함수
import time
print(time.localtime())
print(time.strftime("%Y-%m-%d %H:%M:%S"))

import datetime
print("오늘 날짜는 ", datetime.date.today())


### timedelta : 두 날짜 사이의 간격
today = datetime.date.today()
td = datetime.timedelta(days = 100)         # 100일 저장
print("우리가 만난지 100일은", today + td)   # 100일 후 날짜 



### -------------------------------------- < Quiz >
# 프로젝트 내에 나만의 시그니처 남기는 모듈 만들기
# 조건 : 모듈 파일명은 byme.py로 작성

import byme
byme.sign()
# 이 프로그램은 나도코딩에 의해 만들어졌습니다.
# 유튜브 : http://youtube.com
# 이메일 : nadocoding@gmail.com

