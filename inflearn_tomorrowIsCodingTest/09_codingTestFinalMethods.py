
# 해당 object의 내장 method를 보는 방법
dir([1, 2, 3, 4])

format(10000000000, ',')
# '10,000,000,000'

format(10000000000, 'E')
# '1.000000E+10'

# 16진수
format(10000000000, 'x')
# '2540be400'

format(10000000, '!>020,.4f')
# '!!!!!10,000,000.0000'

###
def yuna(value):
    if value % 2 == 0:
        return True
    else:
        return False

list(filter(yuna, range(20)))
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

list(filter(lambda x: x % 2 == 0, range(20)))
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

[i for i in range(20) if i % 2 == 0]
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

###
len([1, 2, 3, 4])    # 4

###
# map(function, value)
list(map(yuna, range(20)))   # 결과값 전체 반환

list(map(lambda x: x % 2 == 0, range(20)))
list(map(lambda x: x ** 2, range(20)))

list(zip(['a', 'b', 'c', 'd'], [1, 2, 3, 4], [10, 20, 30, 40], 'ABCD'))
# [('a', 1, 10, 'A'), ('b', 2, 20, 'B'), ('c', 3, 30, 'C'), ('d', 4, 40, 'D')]

###
max([1, 2, 3, 4])   # 4
min([1, 2, 3, 4])   # 1

###
reversed()    # 역순이지 역정렬은 아님

l = [10, 5, 4, 3, 7, 6]
l.sort()      # 리스트를 직접 만져서 정렬
sorted(l)     # 리스트를 직접 만지지 않음

testCaseOne = ['abc', 'def', 'hello world', 'hello', 'python']
testCaseTwo = 'Life is too short. You need python'.split()
testCaseThree = list(zip('anvfe', [1, 2, 5, 4, 3]))

sorted(testCaseOne, key = len, reverse = True)
# ['hello world', 'python', 'hello', 'abc', 'def']

sorted(testCaseTwo, key = str.lower)
# ['is', 'Life', 'need', 'python', 'short.', 'too', 'You']

sorted(testCaseThree, key = lambda x:x[0])
# [('a', 1), ('e', 3), ('f', 4), ('n', 2), ('v', 5)]

sorted(testCaseThree, key = lambda x:x[1])
# [('a', 1), ('n', 2), ('e', 3), ('f', 4), ('v', 5)]
# -> dictionary 형태 = dictionary 정렬할 때도 사용 가능

###
5 in [1, 2, 3, 4, 5]
5 not in [1, 2, 3, 4, 5]


''' list
append : 1개요소추가
clear  : 전체 삭제
copy   : 어떤 list를 function으로 넘기기 전 원본 복사
count  : 개수
extend : 요소를 많이 추가할 때
index  : 어떤 요소를 찾을 때
insert : 해당 자리에 요소를 넣을 때
pop    : 뒤에서 값을 꺼낼 때
remove : 어떤 요소를 지울 때
         (Big O: N -> 시간이 많이 걸리기 때문에,
         시간 측정 시험의 경우 del 쓰는 게 더 좋음)
reverse : 역순
sort    : 정렬
'''

l = [1, 2, 3, 4, 5]

def listChange(x):
    x[0] = 1000

listChange(l.copy())
l       # [1, 2, 3, 4, 5]

listChange(l)
l       # [1000, 2, 3, 4, 5]

###
l = []
l.append(10)
l.append(20)
l.append(30)
l.pop(0)
# 10 : 가장 먼저 들어온 데이터가 가장 먼저 나가는 구조 = Queue

l = []
l.append(10)
l.append(20)
l.append(30)
l.pop()
# 30 : 가장 늦게 들어온 데이터가 가장 먼저 나가는 구조 = Stack


''' tuple
count
index
'''
t = (1, 2, 3)
dir(t)


''' dictionary
clear
copy
fromkeys
get
items
keys
pop
popitem
setdefault
update
values
'''

d = {'one':'하나', 'two':'둘'}
dir(d)

d.keys()
# dict_keys(['one', 'two'])

d.values()
# dict_values(['하나', '둘'])

d.items()
# dict_items([('one', '하나'), ('two', '둘')])

del d['one']
d
# {'two': '둘'}


''' set
add
clear
copy
difference          : 차집합 ex) 집합1.difference(집합2) = 집합1 - 집합2
difference_update
discard
intersection        : 교집합
intersection_update
isdisjoint
issubset
issuperset
pop
remove            : 요소 제거
symmetric_difference
symmetric_difference_update
union             : 합집합
update            : 한 번에 많은 데이터 추가
'''

s = set('11122345666')
s
# {'1', '2', '3', '4', '5', '6'}

s.add(7)
s
# {'1', '2', '3', '4', '5', '6', 7}

s.discard(7)
s
# {'1', '2', '3', '4', '5', '6'}

# 요소가 있는지 확인
'1' in s

###
판콜에이 = {'A', 'B', 'C'}
타이레놀 = {'A', 'B', 'D'}

# 차집합
print(판콜에이.difference(타이레놀))
# {'C'}

# 교집합
print(판콜에이.intersection(타이레놀))
# {'A', 'B'}

# 같은 원소 개수
print(len(판콜에이.intersection(타이레놀)))
# 2

# 합집합
print(판콜에이.union(타이레놀))
# {'D', 'C', 'A', 'B'}


''' 문제
단톡방에 x마리의 동물이 대화를 하고 있습니다.
각각의 동물들이 톡을 전송할 때마다 서버에는 아래와 같이 저장됩니다.
serverData = '개리 라이캣 개리 개리 라이캣 자바독 자바독 파이 썬'

1. 단톡방에는 모두 몇 마리의 동물이 있을까요? 톡은 무조건 1회 이상 전송합니다.
2. 단톡방에 동물들마다 몇 번의 톡을 올렸을까요?
'''

serverData = '개리 라이캣 개리 개리 라이캣 자바독 자바독 파이 썬'

# 전체 동물 숫자 : 5
len(set(serverData.split()))

d = {}
for i in set(serverData.split()):
    print(i, serverData.split().count(i))
    d[i] = serverData.split().count(i)

# 파이 1
# 자바독 2
# 라이캣 2
# 개리 3
# 썬 1

d   # {'파이': 1, '자바독': 2, '라이캣': 2, '개리': 3, '썬': 1}

for i in '1 2 3 4 5 6 7'.split():
    print(int(i))
# 1, 2, 3, 4, 5, 6, 7
    
[int(i) for i in '1 2 3 4 5 6 7'.split()]
# [1, 2, 3, 4, 5, 6, 7]

list(map(int, '1 2 3 4 5 6 7'.split()))
# [1, 2, 3, 4, 5, 6, 7]

