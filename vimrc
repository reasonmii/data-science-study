"Syntax Highlighting : 구문 강조
if has("syntax")
    syntax on
endif

"color 테마 : onehalfdark 적용
colo onehalfdark

"Syntax Highlighting은 기본제공이라 안 예쁠 수 있음
" → 아래와 같이 추가 편집
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

set hlsearch            "검색어 하이라이팅
set nu                  "number : 줄번호 표시
"set numberwidth=4      "줄의 번호를 표시하는 곳의 가로 길이
set autoindent          "자동 들여쓰기

"the number of context lines you would like to see above and below the cursor
"→ 2 lines visible while scrolling
set scrolloff=2

"on first <Tab> it will complete to the longest common string
"and will invoke wildmenu (a horizontal and unobtrusive little menu).
"On next <Tab> it will complete the first alternative
"and it will start to cycle through the rest.
"You can go back and forth with <Tab> and <S-Tab> respectively.
set wildmenu
set wildmode=longest,list

set ts=4                  "tab stop : 문서에 있는 '\t' 문자를 인식하는 간격
set sts=4                 "soft tab stop : tab키를 눌렀을 때 간격
"set expandtab            "탭을 누르면 탭 대신 스페이스로 입력
set sw=4                  "shift width : <<, >>, auto indenting 입력 시 간격

set cindent               "C언어 자동 들여쓰기

set autowrite             "다른 파일로 넘어갈 때 자동 저장
set autoread              "작업 중인 파일이 외부에서 변경됐을 경우 자동으로 불러옴

"set ignorecase           "검색시 대소문자 무시
set smartcase             "검색시 대소문자 구별
set smarttab              "백스페이스로 지울때 탭 단위로 삭제
set smartindent           "언어별 자동으로 들여쓰기

set showmatch             "일치하는 괄호 하이라이팅
set incsearch             "단어 검색시 글자 입력할 때마다 검색

"set paste                "붙여넣기 계단 현상 없애기
set bs=eol,start,indent   "line의 끝, 시작, 들여쓰기에서 백스페이스 사용 시 이전 line과 연결됨
set history=256           "vi 편집기록 기억 개수
set ruler                 "현재 커서 위치 표시

"하단에 상태바 표
"0: 표시X, 1: 창이 두 개 이상일 때 표시, 2: 항상 표시
set laststatus=2
set statusline=\ %<%l:%v\ [%P]%=%a\ %h%m%r\ %F\

"커서를 파일의 마지막으로 수정된 위치로 이동
au BufReadPost *
\ if line("'\"") > 0 && line("'\"") <= line("$") |
\ exe "norm g`\"" |
\ endif

"파일 인코딩을 한국어로
if $LANG[0]=='k' && $LANG[1]=='o'
set fileencoding=korea
endif
