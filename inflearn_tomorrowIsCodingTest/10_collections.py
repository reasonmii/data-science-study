
########## ========== namedtuple ========== ##########

from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(11, y=22)       # instantiate with positional or keyword arguments
p[0] + p[1]               # indexable like the plain tuple (11, 22)

p
# Point(x=11, y=22)

p.x   # 11
p.y   # 22

p[x]   # Error

i, j = p
i, j
# (11, 22)

Point
dir(Point)

d = {
     'x' : 100,
     'y' : 200
     }

p = Point(**d)
p
# Point(x=100, y=200)

p.x   # 100

p._asdict()
# OrderedDict([('x', 100), ('y', 200)])

p._fields
# ('x', 'y')

p._replace
# <bound method Point._replace of Point(x=100, y=200)>

# 다른 값으로 변경 가능
re_p = p._replace(x = 1000)
re_p
# Point(x=1000, y=200)

# p 자체가 변경되지는 않기 때문에 immutable
p
# Point(x=100, y=200)

p.index(100)   # 0
p.index(200)   # 1

p.index('x')   # error
p.index(x)     # error

p.count(100)   # 1
p.count(200)   # 1
p.count(300)   # 0

p.count(x)     # error


### namedtuple을 많이 사용하는 예제
from collections import namedtuple

# comma로 변수명 나열이 가능하기 때문에 csv 파일을 가져올 때 유용
# 기술 : namedtuple의 이름
# 변수명 : 기술이름, 자격증, 연차
기술명세 = namedtuple('기술', '기술이름, 자격증, 연차')
이유나 = 기술명세('파이썬', '정보처리기사', '3')
이유나
# 기술(기술이름='파이썬', 자격증='정보처리기사', 연차='3')
dir(이유나)


''' 구조체 선언
python 3.7부터 사용 가능
'''

from dataclasses import dataclass

@dataclass
class Point:
    x: int = None
    y: int = None

print(Point())
# Point(x=None, y=None)

p = Point(10, 20)
p
# Point(x=10, y=20)

# unpacking은 불가능
i, j = p     # error

p.x, p.y     # (10, 20)

# 사용 가능한 method를 보면
# x, y만 있고 count, index는 없음
dir(p)



########## ========== deque ========== ##########

from collections import deque

a = [10, 20, 30, 40, 50]
d = deque(a)
d
# deque([10, 20, 30, 40, 50])

dir(d)

d.append(100)
d
# deque([10, 20, 30, 40, 50, 100])

d.appendleft(1000)
d
# deque([1000, 10, 20, 30, 40, 50, 100])

temp = d.pop()
temp          # 100
d
# deque([1000, 10, 20, 30, 40, 50])

temp = d.popleft()
temp         # 1000
d
# deque([10, 20, 30, 40, 50])

# 두 칸 회전시키기
# 알고리즘 문제에서 매우 유용
d.rotate(2)
d
# deque([40, 50, 10, 20, 30])

d.rotate(-1)
d
# deque([50, 10, 20, 30, 40])



########## ========== ChainMap ========== ##########

from collections import ChainMap

oneDict = {'one': 1, 'two': 2, 'three': 3}
twoDict = {'four': 4}

chain = ChainMap(oneDict, twoDict)
chain
# ChainMap({'one': 1, 'two': 2, 'three': 3}, {'four': 4})

dir(chain)

'one' in chain   # True
'four' in chain  # True
'five' in chain  # false

len(chain)   # 4

chain.values()
# ValuesView(ChainMap({'one': 1, 'two': 2, 'three': 3}, {'four': 4}))

chain.keys()
# KeysView(ChainMap({'one': 1, 'two': 2, 'three': 3}, {'four': 4}))

chain.items()
# ItemsView(ChainMap({'one': 1, 'two': 2, 'three': 3}, {'four': 4}))

chain[0]          # error
chain['oneDict']  # error

chain.maps
# [{'one': 1, 'two': 2, 'three': 3}, {'four': 4}]

chain.maps[0]   # {'one': 1, 'two': 2, 'three': 3}
chain.maps[1]   # {'four': 4}

# 리스트 연결하기
one = [1, 2, 3, 4]
two = [5, 6, 7, 8]

three = ChainMap(one, two)
three
# ChainMap([1, 2, 3, 4], [5, 6, 7, 8])

6 in three    # True

three.maps[0]   # [1, 2, 3, 4]
three.maps[1]   # [5, 6, 7, 8]



########## ========== Counter ========== ##########

from collections import Counter

a = [1, 1, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 8, 8, 8]
c = Counter(a)
c
# Counter({1: 3, 2: 1, 3: 1, 4: 1, 5: 3, 6: 1, 7: 1, 8: 4})

for i in c:
    print(i)
    
#1 2 3 4 5 6 7 8

# error
for i, j in c:
    print(i, j)    

for i in c.elements():
    print(i)

#1 1 2 3 4 5 5 5 5 6 7 8 8 8 8

c.keys()
# dict_keys([1, 2, 3, 4, 5, 6, 7, 8])

# 각 키의 개수
c.values()
# dict_values([3, 1, 1, 1, 3, 1, 1, 4])

c.items()
# dict_items([(1, 3), (2, 1), (3, 1), (4, 1), (5, 3), (6, 1), (7, 1), (8, 4)])

for i, j in c.items():
    print(i, j)

#1 3
#2 1
#3 1
#4 1
#5 3
#6 1
#7 1
#8 4

c.most_common()
# [(8, 4), (1, 3), (5, 3), (2, 1), (3, 1), (4, 1), (6, 1), (7, 1)]


###
s = 'hello, world'
sc = Counter(s)
sc

#Counter({'h': 1,
#         'e': 1,
#         'l': 3,
#         'o': 2,
#         ',': 1,
#         ' ': 1,
#         'w': 1,
#         'r': 1,
#         'd': 1})

# 요소 더하기
sc.update('hello')
sc
# hello, worldhello : 이렇게 추가된 것

#Counter({'h': 2,
#         'e': 2,
#         'l': 5,
#         'o': 3,
#         ',': 1,
#         ' ': 1,
#         'w': 1,
#         'r': 1,
#         'd': 1})

# 요소 빼기
sc.subtract('hello')
sc

#Counter({'h': 1,
#         'e': 1,
#         'l': 3,
#         'o': 2,
#         ',': 1,
#         ' ': 1,
#         'w': 1,
#         'r': 1,
#         'd': 1})

sc.subtract('hello')
sc

#Counter({'h': 0,
#         'e': 0,
#         'l': 1,
#         'o': 1,
#         ',': 1,
#         ' ': 1,
#         'w': 1,
#         'r': 1,
#         'd': 1})

sc.subtract('hello')
sc

#Counter({'h': -1,
#         'e': -1,
#         'l': -1,
#         'o': 0,
#         ',': 1,
#         ' ': 1,
#         'w': 1,
#         'r': 1,
#         'd': 1})

sc.subtract(Counter('hello'))
sc

#Counter({'h': -2,
#         'e': -2,
#         'l': -3,
#         'o': -1,
#         ',': 1,
#         ' ': 1,
#         'w': 1,
#         'r': 1,
#         'd': 1})


### dictionary의 활용도는 명확하지는 않지만
### 아래와 같이 사용 가능
d = {'one' : 100, 'two' : 200, 'three' : 200}
s = Counter(d)
s
# Counter({'one': 100, 'two': 200, 'three': 200})
# -> 'one'의 개수가 100개인 것처럼 출력 됨

d = {'one' : '100', 'two' : '200', 'three' : '200'}
s = Counter(d)
s
# Counter({'one': '100', 'two': '200', 'three': '200'})



########## ========== OrderedDict ========== ##########

from collections import OrderedDict

oneDict = {'one':1, 'two':2, 'three':3}
d = OrderedDict(oneDict)
d
# OrderedDict([('one', 1), ('two', 2), ('three', 3)])

dir(d)


### 맨 마지막으로 보내기 : default
d.move_to_end('one')
d
# OrderedDict([('two', 2), ('three', 3), ('one', 1)])

d.move_to_end('two', True)
d
# OrderedDict([('three', 3), ('one', 1), ('two', 2)])


### 맨 앞으로 가져오기
d.move_to_end('one', False)
d
# OrderedDict([('one', 1), ('three', 3), ('two', 2)])

d.move_to_end('two', False)
d
# OrderedDict([('two', 2), ('one', 1), ('three', 3)])


### item 꺼내기
d.popitem()        # ('three', 3) : 맨 뒤에서 꺼내기 = default
d.popitem(True)    # ('one', 1) : 맨 뒤에서 꺼내기
d.popitem(False)   # ('two', 2) : 맨 앞에서 꺼내기



########## ========== defaultdict ========== ##########

from collections import defaultdict

# default = str
# 앞으로 모르는 key값이 들어왔을 때는 string 초기값을 입력
d = defaultdict(str)
 
d['one'] = '1'
d['two'] = '2'
d['three']
d
# defaultdict(str, {'one': '1', 'two': '2', 'three': ''})


# 앞으로 모르는 key값이 들어왔을 때는 list 초기값을 입력
d = defaultdict(list)
 
d['one'] = '1'
d['two'] = '2'
d['three']
d
# defaultdict(list, {'one': '1', 'two': '2', 'three': []})


# 앞으로 모르는 key값이 들어왔을 때는 int 초기값을 입력
d = defaultdict(int)
 
d['one'] = '1'
d['two'] = '2'
d['three']
d
# defaultdict(int, {'one': '1', 'two': '2', 'three': 0})

dir(d)


# 일반 dictionary에서는 에러가 나지만
# defaultdict은 아래와 같이 작성 가능
# default인 1을 모두 넣어줌
d = defaultdict(int)
for i in range(10):
    d[i] += 1

d
# defaultdict(int, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1})


# 특히 리스트의 경우 여러 개의 중복값을 저장하기 위한 용도로 많이 사용
강좌 = [
      ('인스타그램클론', 123),
      ('정규표현식', 23),
      ('MBIT페이지만들기', 1313),
      ('python부트캠프', 312),
      ('눈떠보니코딩테스트전날', 1623),
      ]

강좌

d = defaultdict(list)
for 강의, 수강생 in 강좌:
    if 수강생 < 100: d['십'].append(강의)
    elif 수강생 < 1000: d['백'].append(강의)
    elif 수강생 < 10000: d['천'].append(강의)

d

#defaultdict(list,
#            {'백': ['인스타그램클론', 'python부트캠프'],
#             '십': ['정규표현식'],
#             '천': ['MBIT페이지만들기', '눈떠보니코딩테스트전날']})



########## ========== UserDict, UserList, UserString ========== ##########

from collections import UserDict, UserList, UserString

class CustomDict(UserDict):
    def contain_value(self, values):
        # 해당 value가 있는지 확인
        return values in self.data.values()
    
d = CustomDict()
dir(d)

d['one'] = 1
d['two'] = 2

d
# {'one': 1, 'two': 2}

'one' in d   # True

d.data
# {'one': 1, 'two': 2}

type(d.data)
# dict

# 해당 value가 있는지 확인
d.contain_value(1)  # True

