
# 문제1 암호를 해독해라!

text = ['   + -- + - + -   ',
        '   + --- + - +   ',
        '   + -- + - + -   ',
        '   + - + - + - +   ']

########## 방법1
# 0과 1밖에 없기 때문에 일단 +는 1, -는 0으로 바꾸기
# strip() : 양 옆 공백 없애기
# replace는 중첩해서 사용 가능
for i in text:
    print(i.strip().replace(' ','').replace('+','1').replace('-','0'))
    
# replace 예시
'011011011'.replace('0','!').replace('!','+').replace('+','~')
'111'.zfill(10)    # 10자리 수 만들기 (default = 0) : '0000000111'

# int , 2 : 문자를 이진법으로 인식해서 10진법으로 바꿔줌
for i in text:
    print(int(i.strip().replace(' ','').replace('+','1').replace('-','0'), 2))

# built-in function
# ord() : 문자 > 숫자
# chr() : 숫자 > 문자
l = []
for i in text:
    l.append(chr(int(i.strip().replace(' ','').replace('+','1').replace('-','0'), 2)))

# 정답 : JEJU
''.join(l)


########## 방법2
# 한줄코딩 방법
[chr(int(i.strip().replace(' ','').replace('+','1').replace('-','0'), 2)) for i in text ]

# list comprehension 예시
[ i for i in range(10) ]
[ i for i in range(10) if i % 2 == 0 ]
[ f'{i} X {j} = {i*j}' for i in range(2, 10) for j in range(1, 10) ] # 구구단
# 첫 번째 for에 두 번째 for가 속해 있음 (병렬관계 X)


########## 방법3
s = [i.strip().replace(' ','').replace('+','1').replace('-','0') for i in text]
s

# 방법 3-1) lambda x : x 값이 들어오면 오른쪽과 같이 바꿔라
list(map(lambda x : chr(int(x, 2)), s))
''.join(list(map(lambda x : chr(int(x, 2)), s)))

# 방법 3-2)
def f(x):
    return chr(int(x, 2))

list(map(f, s))
''.join(list(map(f,s)))
# 원래 map과 zip을 많이 사용함
