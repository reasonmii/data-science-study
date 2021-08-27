

import timeit

### 재귀함수1 : 반복문을 이용한 1부터 100까지의 합과 곱 ==============================
# 내가 나를 호출하는 방법
# 반복문 <-> 재귀함수 : 서로 혼용 가능

### 1 ~ 100까지의 합
n = 1000000

start = timeit.default_timer()
x = 0
for i in range(1, n + 1):
    x += i
print(x)
end = timeit.default_timer()
print(end - start)             # 0.09969619999992574
# Big O : O(n)

# 시그마공식을 사용하기
start = timeit.default_timer()
x = n*(n+1) // 2
print(x)
end = timeit.default_timer()
print(end - start)             # 0.00034750000031635864
# Big O : O(1)

# -> 공식을 사용하면 훨씬 빠르게 계산
#    따라서, 항상 수학 공식은 없는지 살펴볼 것


### 1 ~ 5까지의 곱
x = 1
for i in range(1, 6):
    x *= i

print(x)


### 재귀함수2 : 재귀함수를 이용한 1부터 100까지의 합과 곱 ==============================

### 1 ~ 100까지의 합
def f(n):
    if n <= 1:
        return 1
    else:
        return n + f(n-1)

n = 100
print(f(n))


### 1 ~ 5까지의 곱
def f(n):
    if n <= 1:
        return 1
    else:
        return n * f(n-1)

n = 5
print(f(n))


### 재귀함수3 : 재귀함수 사례 ==============================

### 2진수 구하기

print(bin(11))   # 2진수
print(oct(11))   # 8진수
print(hex(11))   # 16진수


### 2진수 구하기 - 반복문

x = int(input('2진수로 바꿀 숫자를 입력하세요: '))
result = ''
while True:
    if x % 2 == 0:
        result += '0'
    else:
        result += '1'
    x = x // 2
    if x == 1:
        result += str(x)
        # 역순출력
        print(result[::-1])
        break

# 11 -> 1011


### 2진수 구하기 - 재귀함수

def 이진수구하기(입력값):
    if 입력값 < 2:
        return str(입력값)
    else:
        return str(이진수구하기(입력값//2)) + str(입력값%2)
    
이진수구하기(11)   # 1011
이진수구하기(10)   # 1011


### 문자열뒤집기 - 반복문

result = ''
for i in 'leeyuna':
    result = i + result
print(result)
# anuyeel


### 문자열뒤집기 - 재귀함수

def 문자열뒤집기(문자열):
    if 문자열 == '':
        return None
    else:
        문자열뒤집기(문자열[1:])
        print(문자열[0])

str(문자열뒤집기('leeyuna'))
# anuyeel


### 같은 방법
def 문자열뒤집기(문자열):
    if 문자열 != '':
        문자열뒤집기(문자열[1:])
        print(문자열[0])

'''
1) ABC -> else: -> 문자열뒤집기(‘BC’)
2) BC -> 문자열뒤집기(‘C’)
3) C -> 문자열뒤집기(‘’)
4) ‘’->if : return None
5) C -> 문자열뒤집기(‘’)
    print('C')
7) BC -> 문자열뒤집기('C')
    print('B')
8) ABC -> 문자열뒤집기('BC')
    print('A') 
'''


### 문자로 표현된 숫자 전부 더하기 - 반복문

x = 0
for i in '2231':
    x += int(i)
print(x)
    

### 문자로 표현된 숫자 전부 더하기 - 재귀함수

def 문자열뒤집기(문자열):
    if 문자열 == '':
        return 0
    else:
        print(문자열)
        return int(문자열[0]) + int(문자열뒤집기(문자열[1:]))
 
문자열뒤집기('2231')


### 재귀함수4 : 재귀함수 보강예제 ==============================

### 무한 순환 함수
def helloworld():
    return helloworld()

helloworld()


### 순환을 끝낼 수 있는 조건 추가
def 숫자세기(숫자):
    if 숫자 <= 0:
        print('끝')
    else:
        print(숫자)
        숫자세기(숫자 - 1)

숫자세기(10)


### 피보나치 - 반복문
# 0 1 1 2 3 5 8 13 21

a = 0
b = 1
for i in range(10):
    print(b)
    a, b = b, a+b


### 피보나치 - 재귀함수
# f(n) = f(n-1) + f(n-2)
def 피보나치(숫자):
    if 숫자 == 0 or 숫자 == 1:
        return 1
    return 피보나치(숫자-1) + 피보나치(숫자 - 2)

피보나치(5)

'''
피보나치(5) = 피보나치(4) + 피보나치(3) = 5 + 3 = 8
피보나치(4) = 피보나치(3) + 피보나치(2) = 3 + 2 = 5
피보나치(3) = 피보나치(2) + 피보나치(1) = 2 + 1 = 3
피보나치(2) = 피보나치(1) + 피보나치(0) = 1 + 1 = 2
피보나치(1) = 1
'''


###
def f(n):
    if n <= 1:
        return 1
    else:
        return n * f(n-1)

print(f(5))

# f(5) = 5 * f(4)
#      = 5 * 4 * f(3)
#      = 5 * 4 * 3 * f(2)
#      = 5 * 4 * 3 * 2 * 1


###
# for문
s = 0
for i in [100, 200, 300, 400]:
    s += i
    
print(s)

# 재귀함수
def s(l):
    if len(l) == 1:
        return l[0]
    else:
#        return l[0] + s(l[1:])
        return l[0] + s(l[1:])

s([100, 200, 300, 400])


###
# for문
def f(n, e):
    result = 1
    for i in range(e):
        result *= n
    return result

print(f(2, 6))

# 재귀함수
def f(n, e):
    if e == 1:
        return n
    else:
        return n * f(n, e-1)
    
f(2, 6)


### comma를 세 자리마다 찍기
def comma(s):
    if len(s) < 3:
        return s
    else:
        return comma(s[:len(s) - 3]) + ',' + s[len(s) -3:]

comma('10000000')

# built-in 함수
n = 999999999
n = format(n, ',')
print(n)
# 999,999,999


### 선택정렬 ==============================

입력값 = [5, 10, 66, 77, 54, 1, 32, 11, 15, 2]
정렬된리스트 = []

while 입력값:
    정렬된리스트.append(min(입력값))
    입력값.pop(입력값.index(min(입력값)))
    print(정렬된리스트)

print("최종값 : ", 정렬된리스트)


### min, append 등 함수 없이 계산하는 방법

def 최솟값(l):
    최소 = l[0]
    index = 0
    count = 0
    for i in l:
        if 최소 > i:
            최소 = i
            index = count
        count += 1
    return 최소

print(최솟값(입력값))


def 최솟값_인덱스(l):
    비교값 = l[0]
    index = 0
    for i in range(len(l)):
        if l[i] < 비교값:
            index = i
            비교값 = l[i]
    return index

print(최솟값_인덱스(입력값))


while 입력값:
    정렬된리스트.append(최솟값(입력값))
    입력값.pop(최솟값_인덱스(입력값))
    print(정렬된리스트)

print("최종값 : ", 정렬된리스트)


### 삽입정렬 ==============================

입력값 = [5, 10, 66, 77, 54, 32, 11, 15]
정렬된리스트= []

def 삽입값이_들어갈_인덱스(정렬된리스트, 삽입값):
    for i in range(len(정렬된리스트)):
        if 삽입값 < 정렬된리스트[i]:
            return i
    return len(정렬된리스트)
    

while 입력값:
    삽입값 = 입력값.pop(0)
    인덱스 = 삽입값이_들어갈_인덱스(정렬된리스트, 삽입값)
    print('인덱스: ', 인덱스, ', 삽입값: ', 삽입값, ', 정렬된리스트: ', 정렬된리스트, )
    정렬된리스트.insert(인덱스, 삽입값)

print(정렬된리스트)


### 병합정렬 ==============================

def 병합정렬(입력리스트):
    입력리스트의길이 = len(입력리스트)
    결과값 = []
    if 입력리스트의길이 <= 1:
        return 입력리스트
    중간값 = 입력리스트의길이 // 2
    그룹1 = 병합정렬(입력리스트[:중간값])
    그룹2 = 병합정렬(입력리스트[중간값:])
    print('그룹1 : {}, 그룹2 : {}\n'.format(그룹1, 그룹2))
    while 그룹1 and 그룹2:
        if 그룹1[0] < 그룹2[0]:
            결과값.append(그룹1.pop(0))
        else:
            결과값.append(그룹2.pop(0))
    print('결과1: ', 결과값)
    while 그룹1:
        결과값.append(그룹1.pop(0))
    print('결과2: ', 결과값)
    while 그룹2:
        결과값.append(그룹2.pop(0))
    print('결과3: ', 결과값)    
    return 결과값
    
병합정렬([5, 10, 66, 77, 54, 32, 11, 15])


### 퀵정렬 ==============================

g_list = []

def 퀵정렬(입력리스트):
    global g_list
    입력리스트의길이 = len(입력리스트)
    if 입력리스트의길이 <= 1:
        return 입력리스트
    피벗값 = 입력리스트.pop(0)
    그룹1 = []
    그룹2 = []
    for i in range(입력리스트의길이):
        if 입력리스트[i] < 피벗값:
            그룹1.append(입력리스트[i])
        else:
            그룹2.append(입력리스트[i])
    result = '그룹1: {}, 피벗: {}, 그룹2: {}'.format(그룹1, 피벗값, 그룹2)
    g_list.append(result)
    return 퀵정렬(그룹1) + [피벗값] + 퀵정렬(그룹2)

입력값 = [66, 77, 54, 32, 10, 5, 11, 15]
print(퀵정렬(입력값))

for i in g_list:
    print(i)
