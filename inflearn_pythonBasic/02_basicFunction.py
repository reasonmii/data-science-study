
'''참고강의
Udemy 김왼손의유기농냠냠파이썬
Inflearn 파이썬 무료강의 (기본편) - 6시간 뒤면 나도 개발
'''

''' Visualize your code
pythontutor.com
> Start visualizing your code now
'''

''' Comment Shortcut
1. Single line comment. Ctrl + 1.
2. Multi-line comment select the lines to be commented. Ctrl + 4.
3. Unblock Multi-line comment. Ctrl + 5.
'''

###
count = 0
count += 1
count -= 1
count *= 2

print(count)


### type
my_str2 = 'coding'
type(my_str2)


### split
fruit_str = 'apple banna lemon'
fruits = fruit_str.split()
print(fruits)

fruits = fruit_str.split('a')
print(fruits)


### format
print('Life is {}'.format('Short!'))
print('{}*{} = {}'.format(2,3,6))

print('Life is %s' % 'Short')
print('%d * %d = %d' % (2, 3, 2 * 3))



### -------------------------------------- < String >

### multiple lines
print("""첫번째
두번째
세번째""")


### 출력의 끝 지정하기
print('coding', end='')
print('coding', end='-')
print('coding', end='\n')   # 줄바꿈
print('coding', end='\t')   # tab


### escape code
print('줄바꿈\n줄바꿈')
print('탭\t탭')

print('\\n')
print('\\t')

print('I\'m Yours')


### to print ""
print("저는\"나도코딩\" 입니다.")
print('저는 "나도코딩" 입니다.')


### \r : go to the front and rewrite
print("Red Apple\rPine")     # PineApple
print("RedApple\rPine")      # Pinepple


### \b : backspace - delete one letter
print("Red\bApple")         # ReApple


### \t : tab
print("Red\tApple")         # Red     Apple


### Quiz : Create the password
# delete http://
# delete the ltters after the first dot (.)
# password : first 3 letters + the number of letters + the number of 'e' in the letter + "i"

url = "http://naver.com"
url = "http://youtube.com"
my_str = url.replace("http://","")
my_str = my_str[:my_str.index(".")]
password = my_str[:3] + str(len(my_str)) + str(my_str.count("e")) + "!"
print("{0} 의 비밀번호는 {1} 입니다.".format(url, password))



### -------------------------------------- < list >
### mutable : can change the value

### delete list elements
my_list = [123, 'abc', True]
del my_list[0]
print(my_list)


### sort
my_list = [1,5,3,4,2]
my_list.sort()
print(my_list)


### count list elements
my_list = ['a','c','a','b']
print(my_list.count('a'))
print(my_list.count('c'))


### in & not in
print('a' in my_list)
print('f' not in my_list)


### Quiz : Create the lottery program
# There's a pyton coding contest
# one participant : chicken
# three participants : coffee coupons
from random import *
users = list(range(1,21))
shuffle(users)
winners = sample(users, 4)

print("-- 당첨자 발표 --")
print("치킨 당첨자 : {0}".format(winners[0]))
print("커피 당첨자 : {0}".format(winners[1:]))
print("-- 축하합니다 --")



### -------------------------------------- < Tuple >
### immutable : cannot change the value

my_tuple1 = ()
my_tuple2 = (1,)                  # 요소가 1개 튜블 만들 때 comma 필수
my_tuple3 = ('a','b','c')
my_tuple4 = 'abc', 3.14, True     # 소괄호 생략 가능

print(my_tuple2)
print(type(my_tuple2))

print(my_tuple2)
print(type(my_tuple2))

# 요소가 1개인데 comma 없으면?
tmp = (1)
print(type(tmp))


### -------------------------------------- < Packing >

### Packing
my_tuple = 3.14, 'Python', False


### Unpacking
i, s, b = (123,'abc', True)



### -------------------------------------- < for : 횟수로 반복 >

print(range(3))
print(list(range(3)))

for count in range(0,3):
    print('횟수:', count)

for j in range(2):
    for i in range(2):
        print('i:{}, j:{}'.format(i,j))


### list Comprehension

numbers = [1,2,3,4,5,6,7,8,9,10]
odd_numbers = []

for number in numbers:
    if number % 2 == 1:
        odd_numbers.append(number)

# [number for 할 때 number : for에서 append 안에 있는 것
odd_numbers = [number for number in numbers if number % 2 == 1]


### Quiz : 50명의 승객과 매칭 기회가 있을 때, 총 탑승 승객 수 구하기
# 조건1 : 승객별 운행 소요 시간은 5분 ~ 50분 사이 난수
# 조건2 : 소요시간 5분 ~ 15분 사이의 승객만 매칭할 것
from random import *
cnt = 0
for guest in range(1,51):
    time = randrange(5,51)
    if 5 <= time <= 15:
        cnt += 1
        print("[O] {}번째 손님 (소요시간 : {}분)".format(guest, time))
    else:
        print("[ ] {}번째 손님 (소요시간 : {}분)".format(guest, time))
print("총 탑승승객 : {}".format(cnt))



### -------------------------------------- < 조건 >

### Compare
my_cmp = 1 <= 2 < 3
print(my_cmp)

my_con = not 1 == 1
print(my_con)

name = 'Alice'
if name == 'Alice':
    print('당신이 Alice군요.')
elif name == 'Bob':
    print('당신이 Bob이군요.')
else:
    print('당신은 누구십니까?')
    
    
    
### -------------------------------------- < While : 조건으로 반복 >

count = 0
while count < 3:
    print('횟수:', count)
    count = count+1


### continue
count = 0
while = count < 5:
    count = count + 1
    if count % 2 == 1:
        continue        # 다시 count < 5 돌아가기
    print(count)


### input
while True:
    name = input('당신의 이름은?')
    if name == '종료':
        print('종료합니다.')
        break
    print('{}님 안녕하세요'.format(name))


###
name = input('이름을 입력하세요')
print(name)
print(type(name))


### change type
print(type(40))
print(type(str(40)))
print(float(1))
print(int(1.0))

print(list('Lefty'))



### -------------------------------------- < Dictionary >
### key : immutable : cannot change the value

my_dict = {}
my_dict[1] = 'a'
my_dict['b'] = 2
my_dict['c'] = 'd'

print(my_dict)
print(my_dict[1])
print(my_dict['b'])


### remove
del my_dict[1]
del my_dict['b']
print(my_dict)


###
my_dict = {'k1': 'v1', 'k2': 'v2'}

for val in my_dict.values():
    print(val)
    
for key in my_dict.keys():
    print(key)

for key, val in my_dict.items():
    print(key, val)



### -------------------------------------- < Functions >
### 1) 내장함수
### 2) 모듈의 함수
### 3) 사용자 정의 함수

def my_sum(n1, n2, n3):
    return n1 + n2 + n3

my_sum(1,2,3)


###
def my_sum_mul(n1, n2):
    return n1 + n2, n1 * n2

s, m = my_sum_mul(2,3)
print(s)
print(m)

result = my_sum_mul(2,3)
result

print(type(result))


### Docstring = Documentation String
### 주로 함수 안에 설명을 위해 쓴 주석
def sum_mul(num1, num2):
    '''입력 값을 더하고 곱합니다.'''
    return num1 + num2, num1 * num2

sum_mul(2,3)


### 사용자 정의 함수 확인하기
def f(x, y):
    return x + y
f(3, 6)

f?             # f, 파일위치 등 설명
f??            # source code도 볼 수 있음


### random
import random
fruits = ['apple','banana','lemon']
my_fruit = random.choice(fruits)
print(my_fruit)


### sample : 중복 없이 선택
import random
fruits = ['apple','banana','lemon']
my_fruit = random.sample(fruits,2)
print(my_fruit)


### randint : 정해진 범위에서 랜덤하게 선택
import random
my_int = random.randint(0,10)
print(my_int)


### Quiz : 표준 체중 program
# 조건1 : 함수명 - std_weight
#        전달값 - 키(height), 성별(gender)
# 조건2 : 표준 체중은 소수점 둘째자리까지 표시
def std_weight(height, gender):
    if gender == "남자":
        return height * height * 22
    else:
        return height * height * 21

height = 175
gender = "남자"
weight = round(std_weight(height/100, gender),2)
print("키 {}cm {}의 표준 체중은 {}kg입니다.".format(height, gender, weight))



### -------------------------------------- < Input and Output >

# end : 뒷 문장이 줄바꿈이 안 됨
print("Python", "Java", sep = ",", end = "?")
print("무엇이 더 재밌을까요?")


###
import sys
print("Python", "Java", file = sys.stdout)  # 표준출력
print("Python", "Java", file = sys.stderr)  # 표준에러


### 정렬
# ljust(n) : n개의 공간을 확보하고 왼쪽정렬
# rjust(n) : n개의 공간을 확보하고 오른쪽 정렬
scores = {"수학" : 0, "영어" : 50, "코딩" : 100}
for subject, score in scores.items():
    print(subject.ljust(8), str(score).rjust(4), sep = ":")
    

### zfill(n) : n개의 공간을 확보하고 빈공간은 0으로 채우기 
# 은행 대기 순번표
for num in range(1, 21):
    print("대기번호 : " + str(num).zfill(3))
    

### input
# input으로 받은 글자는 항상 string으로 인식됨
answer = input("아무 값이나 입력하세요 : ")
print("입력하신 값은 " + answer + "입니다.")


### 빈 자리는 빈공간으로 두고, 오른쪽 정렬 하고, 총 10자리 공간 확보
print("{0: >10}".format(500))        #        500

# 양수는 +, 음수는 -
print("{0: >+10}".format(500))      #       +500
print("{0: >+10}".format(-500))     #       -500

# 왼쪽 정렬하고 빈칸으로 _ 채움
print("{0:_<+10}".format(500))     # +500______
print("{0:_<+10}".format(-500))    # -500______


### 3자리마다 comma
print("{0:,}".format(100000000000))        # 100,000,000,000

# 3자리마다 comma, 부호
print("{0:+,}".format(100000000000))       # +100,000,000,000
print("{0:+,}".format(-100000000000))      # -100,000,000,000

# 3자리마다 comma, 부호, 자릿수 확보, 빈자리는 ^ 표시
print("{0:^<+30,}".format(100000000000))  # +100,000,000,000^^^^^^^^^^^^^^


### 소수점 출력
print("{0:f}".format(5/3))     # 1.666667
print("{0:.2f}".format(5/3))   # 1.67


### file input and output
# encoding = utf8 : 한글 정보 인식 잘 함
score_file = open("score.txt", "w", encoding = "utf8")
print("수학 : 0", file = score_file)
print("영어 : 50", file = score_file)
score_file.close()

# "a" file 이어쓰기
score_file = open("score.txt", "a", encoding = "utf8")
score_file.write("과학 : 80")
score_file.write("코딩 : 100")
score_file.close()

# "r" : read files
# end = "" : 줄바꿈 안 하고 읽고 싶을 때
score_file = open("score.txt", "r", encoding = "utf8")
print(score_file.read())                     # read the whole file
print(score_file.readline())                 # read the 1st line
print(score_file.readline())                 # read the 2nd line
print(score_file.readline(), end = "")       # read the 3rd line
print(score_file.readline())                 # read the 4th line
score_file.close()

# 총 몇 줄인지 모르는 경우
score_file = open("score.txt", "r", encoding = "utf8")
while True:
    line = score_file.readline()
    # line이 없으면 = 읽어오는 내용이 없으면
    if not line:
        break
    print(line, end = "")
score_file.close()


### file 내용 list로 저장하기
score_file = open("score.txt", "r", encoding = "utf8")
lines = score_file.readlines()         # list 형태로 저장
for line in lines:
    print(line, end = "")
score_file.close()


### Create the file
import pickle
profile_file = open("profile.pickle", "wb")
profile = {"이름":"박명수", "나이":30, "취미":["축구","골프","코딩"]}
print(profile)
# Save the information in profile to the profile_file
pickle.dump(profile, profile_file)
profile_file.close()

profile_file = open("profile.pickle", "rb")
# Get the file's information to profile
profile = pickle.load(profile_file)
print(profile)
profile_file.close()


### Using the function "with"
# Simple
# Don't have to write close()
import pickle

with open("profile.pickle", "rb") as profile_file:
    print(pickle.load(profile_file))

with open("study.txt", "w", encoding = "utf8") as study_file:
    study_file.write("파이썬을 열심히 공부하고 있어요")

with open("study.txt", "r", encoding = "utf8") as study_file:
    print(study_file.read())


### Quiz : 1 ~ 50주차 보고서 파일 만드는 프로그램 만들기
# 매주 1회 작성 보고서
for i in range(1, 51):
    with open(str(i) + "주차.txt", "w", encoding = "utf8") as report_file:
        report_file.write("- {} 주차 주간보고 -".format(i))
        report_file.write("\n부서 : ")
        report_file.write("\n이름 : ")
        report_file.write("\n업무 요약 : ")



### -------------------------------------- < % : Magic Command >

# magic에 대한 설명
%magic

### 해당 셀의 실행 시간 출력
%time print(sum(range(1, 1000000)))
