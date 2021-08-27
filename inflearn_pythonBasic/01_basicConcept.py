'''참고강의
Inflearn 30분 요약 강좌 시즌2 : Python 활용편
'''


''' Anaconda Prompt에서 실행하기
Jupyter 실행할 위치 생각해서 아래와 같이 코드 작성
code : Jupyter notebook --notebook-dir="D:\000tmp\Data_Science\python_OnlineClass_kr" 
'''

''' Google colab
Colaboratory : 클라우드에서 실행되는 무료 Jupyter 노트 환경

장점 : 속도 매우 빠름
      GPU 연결 가능 (여러 곳에서 작업 가능)

사용방법1
Google에 검색해서 사용
1) 왼쪽 코드 스니펫 검색 가능
    ex) visualization
    여러 가지 관련 코드 볼 수 있음
2) 셀 삭제 : ctrl + M + D
    ctrl + Z로 복원 불가능
3) 환경설정 : 들여쓰기 4 설정
4) 링크모양 복사 > url창 : ctrl + V
    - 로그아웃 상태에서도 해당 코드 조회 가능

사용방법2
Google Drive > 더보기 > Google Colaboratory
1) Markdown : ctrl + M, M
2) import 방법
    from google.colab import drive
    pwd
    !ls
    drive.mount('/content/mountdrive/')
    연결할 드라이브 선택 - 허용 - 코드 ctrl + c
    Enter your authorization code : ctrl + v
3) 파일 확인
    cd mountdrive/
    ls
4) 폴더 만들기
    cd My\ Drive
    mkdir test
    cd test     # 이 폴더가 구글 드라이브에도 생성되어 있음
'''



### -------------------------------------- < Jupyter 활용법 >

'''
1) View - Toggle header : 파일 제목설정
2) Mark Down : ESC + M (Editor → Command) 
    # Hello World
    ## Hello World
    ### Hello World
    
    1. Hello world
    2. Hello world
    
    아래와 같이 작성하면 모두 ● 기호로 바뀜
    - hello world
    - hello world
    
    * hello world
    * hello world
    
    인용구
    > hello world
    
    Italic
    *hello world*
    
    Bold
    **Hello world**
    
    표 만들기
    Google : markdown table generator 검색
    보고 쓰거나 복+붙하기
    
3) 실행 : Ctrl + Enter
4) 실행 + 다음행 생성 : Alt + Enter
5) 여러 줄 출력하려면 모두 print 써야 함

6) 셀 삭제 : D + D
7) 셀 생성
    In[1] 부분 클릭하고 A = 상단 Insert > Insert Cell Above
    In[1] 부분 클릭하고 B = 상단 Insert > Insert Cell Below

8) 주석 : Ctrl + /
9) 함수 자동완성 : 함수 앞부분 쓰다가 tab
    
10) 실행했는데 너무 오래 걸리는 경우 : 상단 Kernel > Restart    

11) Shift + L : 셀 별 라인 숫자 생성

12) 상단 View > Cell Toolbar > Tags
    셀 별 tag 추가 가능
    코드가 길어졌을 때 검색해서 바로가기 가능
'''


''' ls, clear 명령어
리눅스, MAC OS 환경 terminal에서만 사용 가능

Window에서 사용하고 싶은 경우
Anaconda Prompt 작성
doskey 사용을 희망하는 명령어 = 기존 명령어

doskey ls = dir
doskey clear = cls
'''

### 현재 위치 폴더와 파일
# Windows
!dir
!dir -al

# linux, Mac
!ls
!ls -al

### install
!pip3 install bs4

### Create the file
from pathlib import Path
Path('001.py').touch()
!dir
# 001.py 파일 메모장으로 열어서 아래 문구 적어보기
# print('hello world!!')
%run 001.py


### 사용자 정의 함수 확인하기
# f + shift + tab : 간단한 정보
