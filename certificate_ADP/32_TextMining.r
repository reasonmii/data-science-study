
#===============================================================================================================
# 텍스트 마이닝 Text Mining
# 입력된 텍스트를 구조화 해 그 데이터에서 패턴 도출, 결과를 평가/해석
# 자연어로 구성된 비정형 텍스트 데이터 속에서 정보, 관계 발견
# 분석결과 평가 : 정확도, 재현율 사용
#
# Corpus
# 데이터의 정제, 통합, 선택, 변환의 과정을 거친 구조화된 단계
#
# tm package
# - VCorpus() : 문서를 Corpus class로 만들어주는 함수 - 현재 구동 중인 R 메모리에서만 유지
# - PCorpus() : 문서를 Corpus class로 만들어 R 외부의 DB, 파일로 관리
# - DirSource(), VectorSource(), DataframeSource()
#   텍스트를 저장한 directory, vector, dataframe으로부터 corpus 생성을 위한 source를 만들어주는 함수들
#
# tm package 문서 전처리
# tm_map(x, FUN) : x 데이터에 대해 FUN에 지정한 함수 적용
# tm_map(data, as.PlainTextDocument) : XML 문서를 text로 전환
# tm_map(data, stripWhitespace) : Space제거
# tm_map(data, tolower) : 대문자를 소문자로 변환
# tm_map(data, removewords, stopwords("english")) : 띄어쓰기, 시제 표준화
#
# DocumentTermMatrix ; Corpus로부터 문서별 특정 문자의 빈도표 생성
# TermDocumentMatrix : Corpus로부터 단어별 문서의 빈도표 생성
#
# Dictionary
# 텍스트마이닝 분석 시 사용하고자 하는 단어들의 집합
# 분석하고자 하는 단어들을 별도의 사전으로 정의
#
# 감성분석
# 문장에서 사용된 단어의 긍정과 부정 여부에 따라 전체 문장의 긍정/부정 여부 평가
#
# 한글처리
# KoNLP package
# rJava package, JRE program 반드시 추가 설치해야 KoNLP package 사용 가능
# 명사 추출 : extractNoun("문장")
#
# word cloud
# 문서에 포함된 단어의 사용 빈도를 효과적으로 보여주기 위해
# 단어들을 크기, 색 등으로 나타내어 구름과 같은 형태로 시각화하는 기법
# wordcloud package
#===============================================================================================================

install.packages("tm")
library(tm)

data(crude, package="tm")
m <- TermDocumentMatrix(crude, control=list(removePunctuation=T, stopwords=T))

# 단어별 문서에 나온 갯수
inspect(m)
# <<TermDocumentMatrix (terms: 1000, documents: 20)>>
#   Non-/sparse entries: 1738/18262
# Sparsity           : 91%
# Maximal term length: 16
# Weighting          : term frequency (tf)
# Sample             :
#          Docs
# Terms    144 236 237 242 246 248 273 489 502 704
# bpd      4   7   0   0   0   2   8   0   0   0
# crude    0   2   0   0   0   0   5   0   0   0
# dlrs     0   2   1   0   0   4   2   1   1   0
# last     1   4   3   0   2   1   7   0   0   0
# market   3   0   0   2   0   8   1   0   0   2
# mln      4   4   1   0   0   3   9   3   3   0
# oil     12   7   3   3   5   9   5   4   5   3
# opec    13   6   1   2   1   6   5   0   0   0
# prices   5   5   1   2   1   9   5   2   2   3
# said    11  10   1   3   5   7   8   2   2   4

# 10개 이상 사용된 단어
findFreqTerms(m, 10)
# [1] "barrel"     "barrels"    "bpd"        "crude"      "dlrs"       "government" "industry"   "kuwait"     "last"       "market"    
# [11] "meeting"    "minister"   "mln"        "new"        "official"   "oil"        "one"        "opec"       "pct"        "price"     
# [21] "prices"     "production" "reuter"     "said"       "saudi"      "sheikh"     "will"       "world"  

# "oil" 단어와 65% 이상 연관성이 있는 단어
findAssocs(m, "oil", 0.7)
# $oil
# 158      opec     named   clearly     late    prices    trying    winter    
# 0.87     0.87      0.81     0.79      0.79      0.79      0.79      0.79      
# markets   said     analysts   agreement   emergency    buyers     fixed 
# 0.78      0.78      0.77      0.76        0.74         0.71       0.71
