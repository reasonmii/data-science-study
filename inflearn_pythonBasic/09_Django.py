

'''참고강의
Inflearn 파이썬 무료강의 (기본편) - 6시간 뒤면 나도 개발
'''

랜딩페이지 만들기

클라우드 환경에서 Django 실행하기
장점 : URL만 있으면 누구나 접속을 할 수 있음
      포트폴리오로 활용 가능

구글 : 구름IDE 검색
회원가입

상단 IDE > dashboard > 새 컨테이너 생성
컨테이너 생성 사용 이유를 제출하면 1~2일 내 구름IDE에서 허용해줌

이름설정, 공개범위 : private, python 선택

터미널창 명령어 안 먹을 때 : ctrl + C 눌러보기

1) mysite 만들기
    # mkdir mysite
    # cd mysite

2) 가상환경 설정하기
    가상환경 안에서 파이썬 작업하면, 파이썬 버전이 이후에 업데이트 되더라도
    코드 일일이 수정할 필요 없이 그 프로그램을 압축해서 서버 이전하면 제대로 작동 함
    # pip3 install virtualenv
    # pip3 list

3) 가상환경 my vertual environment 잡기
  (이름은 마음대로 설정)
  # virtualenv myvenv

4) 가상환경 속으로 들어가기
   pycharm 사용하면 처음부터 이 상태이기 때문에 이 앞에 과정들이 필요 없음
    # source myvenv/bin/activate
    # pip3 list

5) django 설치하기
    # pip3 install django
    # pip3 list

6) 'landingpage' 프로젝트 생성하기
    . -> 현재 페이지에 생성
    # django-admin startproject landingpage .
    # python mange.py migrate
    make migrations와 migrate은 python이 DB를 만질 수 있게 하는 명령어

7) 80 port를 이용해서 서비스하겠다는 의미
   pycharm으로 할 때는 0:80 생략 
    # python manage.py runserver 0:80
   

8) 접속할 수 있는 사람 설정하기
    왼쪽 프로젝트 폴더 부분
    djangolandingpage > mysite > landingpage > settings.py 클릭
    28번째 line : *로 바꾸기 = 모든 사람을 허용하겠다
    ALLOWED_HOST = ['*']
    CTRL + S (저장) -> 서버가 자동적으로 refresh 됨

9) 상단 프로젝트 > 실행 URL과 포트
   URL부분 네모박스 안 대각선 화살표 클릭하면 새로운 창 열림
   이 때 나타나는 URL이 다른 사람들과 내 프로젝트를 공유할 수 있는 URL
   포트폴리오로 사용 가능
   
10) 다시 원래 창으로 오기
    # clear

11) 'main'이라는 app 만들기
    # python manage.py startapp main
    왼쪽 프로젝트 창에서 main app 생성되었는지 확인 가능
    settings.py 열기 > 33번째 line : INSTALLED_APPS 부분
    'django.contrib.staticfiles', 밑에 line에 'main', 적기
    ctrl + S

-----------------------------------------------------------------------------------------
항상 구름IDE에 들어왔을 때 바로 써야 하는 자주 쓰는 코드

> 가상환경으로 들어가기
cd mysite
source myvenv/bin/activate

> 서버 구동시 항상 작성
python manage.py runserver 0:80

1) urls.py 수정하기 : 해당 URL이 어떤 경로로 가야하는지 접속 경로
2) views.py 만들기 : 사용자가 들어왔을 때 어떤 행동을 취할지 + 아래 .html 파일로 연결
3) main > templates > main > index.html 생성하기
4) models.py : 사용자 또는 내가 게시물을 올릴 수 있도록 DB 공간 마련
-----------------------------------------------------------------------------------------

12) 프로젝트 > urls.py : 파일 내 코드 수정
    from django.contrib import admin
    from django.urls import path
    
    아래에 적기
    main이라는 app에 views.py 파일에 index 함수 matching 시키기
    from main.views import index

13) 프로젝트 창 views.py 수정
    from django.shortcuts import render

    아래에 적기
    request(사용자질의)와 함께 main 폴더의 index.html 파일을 render해서 출력하기
    render : main/index.html에는 python 문법들이 들어가는데
             user browser에서는 HTML, CSS, JAVA Script만 인식 가능
             그래서 python 문법 부분은 render 안에서 다 돌린 후에
             사용자에게 보여줄 수 있게 하는 기능
    def index(request):
        return render(request, 'main/index.html')

14) 프로젝트 창 : main 폴더 우클릭 > 새로만들기 > 폴더
    폴더이름 : templates (template이 아니라 templates인 것 주의!)
    templates 폴더 우클릭 > 새로만들기 > 폴더
    폴더이름 : main
    main 폴더 우클릭 > 새로만들기 > 파일
    파일이름 : index.html
    
15) index.html 파일 열기
    hello world

16) 아래 터미널 부분에서 서버 구동해보기
    python manage.py runserver 0:80
    페이지가 뜨면 새로고침 F5
    hello world가 써 있는 것을 볼 수 있음

17) 페이지 디자인하기
    구글 : bootstrap 무료 템플릿 검색
          (유료 템플릿 : 아주 예쁜 홈페이지도 2~3만원이면 구매 가능)
    http://startbootstrap.com > creative > preview로 디자인 확인 > Free Download
    바탕화면에 폴더 만들고 저장

    폴더명 변경 : static
    이 안에 있는 것들이 이미 정적 파일들이기 때문에
    구름IDE에서 사용하기 위한 추가 작업 필요 없음

    폴더 내 index.html 클릭
    shift + 마우스 오른쪽 버튼 : Power Shell 창 열기
    atom . 입력 > 현재 폴더 기준으로 atom 열기 가능
    index.html 부분 클릭 > 내용 모두 복사

    구름 IDE : index.html 안에 모두 붙여넣기 > Ctrl + S
    
    프로젝트 창 : mysite 폴더 우클릭 > 폴더 업로드 > 파일선택
    > 아까 만든 static 업로드
    
    아까 열어놓은 페이지에서 F5
    글자만 나온 것 볼 수 있음
    이미지가 안 나오는 이유 : CSS 반영이 안 되어서
    ex) index.html 파일 코드 보면 link href = "vendor/fontawesome-free" 등의 부분을 찾지 못한 것
    이런 건 모두 static 폴더에 있음
    폴더를 찾을 수 있는 명령어 필요
    
    index.html 맨 윗 부분에 작성 (<!DOCTYPE html> 보다도 윗부분)
    {% load static %}
    link href 부분도 변경
    link href = "{% static 'vendor/fontawesome-free ~ .css' %}"
    
    {% %} -> 이런 걸 template tag라고 함
    용법은 공식홈페이지에 자세히 나와 있음
    
    settings.py에서 static 폴더를 가리키도록 수정 필요
    마지막 줄
    STATIC_URL = '/static/'
    STATICFILES_DIRS = (
            os.path.join(BASE_DIR, 'static'),
            )
    -> comma 꼭 써야 함 주의!
    
    다시 랜딩 페이지로 돌아가서 F5
    제대로 잘 적용된 걸 볼 수 있음
    
18) 페이지 내가 원하는 대로 수정하기

    구글에 django models field 검색하면 공식문서 있음
    Field types 부분에 여러 가지 기능 모두 있고 설명 있음
    게시판을 만들 때 어떤 field가 필요할 지 검색해 보고 만들기
    ex) CharField : 문자열 저장 필드
        DateField : 날짜 저장 필드
    
    터미널창 : 이미지 필드 사용을 위해 pillow 설치 필요
    가끔 버전 충돌이 일어나기 때문에 일단은 2.9.0 버전 설치하기
    # pip install pillow==2.9.0
    # python manage.py makemigrations
    
    프로젝트창 : djangolandingpage > mysite > main > models.py
    Post : 게시물을 찍어내는 틀
    
    from django.db import models
    class Post(models.Model):
        title = models.CharField(max_length = 50)
        contents = models.TextField()
        img = models.ImageField()
        dataCreat = models.DateTimeField()
        category = models.CharField(max_length = 20)    
        
        def __str__(self):
            return self.title
    
    프로젝트창 : djangolandingpage > mysite > main > migrations
    0001_initial.py 파일 생성된 것 확인 가능
    
    터미널창
    # python manage.py migrate
    # python manage.py createsuperuser
    아래와 같이 입력하기 (password는 눈에 보이지 않게 쳐짐)
    username (leave blank to use 'root'): yuna
    Email address: cdyn17@gmail.com
    Password: yuna!
    Password (again): yuna!
    
    프로젝트창 : djangolandingpage > mysite > templates > admin.py
    from django.contrib import admin
    from .models import Post
    
    admin.site.register.(Post)
    
    터미널창
    python manage.py runserver 0:80
    page가 나타남
    url부분 뒤에 /admin/ 추가
    Username, Password 입력
    
    MAIN 부분 Posts 생성된 것 보임
    Add Post 클릭
    title, contents, img, dataCreate, Caategory 있음
    아까 위에서 우리가 다 작성했던 것
    다 작성해보기
    SAVE
    
    title(1) 생성됨
    클릭 > 1.jpg 이미지 클릭하면 이미지가 안 뜸
    이거에 대한 URL이 없기 때문 > 작업 필요
    
    프로젝트창 : djangolandingpage > mysite > static
    1.jpg 이미지 올라간 것 확인 가능
    ① 여기가 아니라 다른 특정 폴더에 이미지 모아줄 것
    ② 이미지를 URL에 연결할 것

    앞으로 이미지가 업로드 되면 postImg라는 폴더가 생성되고
    그 안에 모든 이미지 저장되게 하기
    settings.py 마지막 줄에 추가
    MEDIA_ROOT = os.path.join(BASE_DIR, 'postImg')
    MEDIA_URL = '/postImg/'
    
    urls.py
    from django.conf.urls.static import static
    from django.conf import settings
    
    path : 사용자가 어떤 url을 검색했을 때 어디로 연결해야 하는가
    # path('기본경로', index)
    # path('기본경로 내 admin', )
    urlpatterns = [
        path('admin/', index),
        path('admin/', admin.site.urls),
        ]

    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
    
    터미널창 서버 다시 구동시키기
    # python manage.py runserver 0:80
    
    다시 landingpage 에서 tilte(1) 클릭
    image 다시 넣고 save
    title(1)클릭 > image1 클릭 > 이미지 나타남
    
    프로젝트창 : djangolandingpage > mysite > postimg 폴더 생성되었고
    그 안에 이미지 들어가 있는 것도 확인 가능
    
19) 다른 게시물들이 더 생겼을 때 어떤 화면이 생기는지 보기
    7개의 게시물 올려보기
    
    views.py
    from django.shortcuts import render
    from .models import Post
    
    def index(request):
        postAll = Post.objects.all()   # 전체 게시물이 이 변수로 들어가게 됨
        # index.html에서 postAll 이라는 Key 값으로 postAll 내 값을 받아라
        return render(request, 'main/index.html', {'postAll':postAll})
    
    index.html
    ctrl + A > ctrl + C > 보기 좋게 atom 파일에 붙여넣기
    'src =' 이라고 되어 있는 이미지 관리 부분 코드 모두 수정 필요
    <div class = "col-lg-4 col-sm-6">
    부분 다 접어보면 총 6개의 게시물 올라간 것 볼 수 있음
    첫 번째만 제외하고 모두 삭제하기 > loop 형태로 작성 예정
    첫 번째 시작부분 바로 위에 코드 작성하기
    syntax를 사용할 때 {%%} 사용
    {% for post in postAll %}
    <div class="col-lg-4 col-sm-6">
    마지막 끝나는 부분에 아래 코드 작성하기
    {% endfor %}
    </div>
    
    <div class="col-lg-4 col-sm-6"> 부분 다시 열어보기
    순회하는 부분 코드 변경하기
    변수를 사용할 때 중괄호 두 개 사용 {{}}
    <a class="portfolio-box" href="{{post.img.url}}">
        <img class="img-fluid" src="{{post.img.url}}" alt="">
        <div class="portfolio-box-caption">
            <div class="project-category text-white-50">
                {{post.category}}   # category
            </div>
            <div class="project-name">
                {{post.title}}      # Project Name
            </div>
            # Project contents도 이런 식으로 원하면 추가 가능
            # 아래와 같이 쓰면 제목과 같은 크기로 보이기 때문에
            # 필요하면 font 크기 조절도 필요
            <div class="project-name">
                {{post.contents}}
            </div>
                
    밑에 코드들 중에도 수정 필요한 부분 찾기
    <script src = "{% static 'vendor/jquery-easing/~.js' %}"></script>
    <script src = "{% static 'vendor/magnific-popup/~.js' %}"></script>
    <script src = "{% static 'js/creative.min.js' %}"></script>
    
    변경 완료 했으면 ctrl + A -> ctrl + C
    index.html 파일 ctrl + V
    ctrl + S
    
    page로 넘어가기

