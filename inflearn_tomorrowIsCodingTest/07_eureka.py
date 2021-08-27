
# 문제7 : Eureka!

cross = [[[1, 5, 0, 1, 0],
          [0, 1, 6, 7, 0],
          [6, 2, 3, 2, 1],
          [1, 0, 1, 1, 1],
          [0, 2, 0, 1, 0]],
         [[0, 3, 0, 1, 0],
          [1, 2, 5, 4, 4],
          [0, 0, 3, 0, 0],
          [1, 2, 5, 0, 1],
          [0, 0, 0, 0, 0]],
         [[3, 0, 1, 1, 8],
          [5, 0, 4, 5, 4],
          [1, 5, 0, 5, 1],
          [1, 2, 1, 0, 1],
          [0, 2, 5, 1, 1]],
         [[1, 0, 3, 3, 3],
          [5, 1, 2, 2, 4],
          [1, 5, 1, 2, 4],
          [4, 4, 1, 1, 1],
          [4, 4, 1, 1, 1]],
         [[1, 2, 0, 3, 3],
          [1, 2, 0, 2, 4],
          [1, 2, 0, 2, 4],
          [4, 2, 0, 0, 1],
          [8, 4, 1, 1, 0]],
         [[1, 0, 3, 0, 0],
          [1, 1, 0, 2, 4],
          [0, 0, 1, 2, 4],
          [4, 0, 1, 0, 1],
          [0, 0, 1, 0, 1]]]


### 작은 데이터로 연습해보기 ====================================================
# 1행1열 3에서 -> 5행5열 1까지 가는 길 출력하기

c = [[3, 0, 1, 1, 8],
     [5, 0, 4, 5, 4],
     [1, 5, 0, 5, 1],
     [1, 2, 1, 0, 1],
     [0, 2, 5, 1, 1]]

가중치누적값 = [[0 for i in range(5)] for i in range(5)]

# 어떤 경로를 통해 3 -> 5 갔는지 좌표 저장하기 위함
좌표저장 = [[[0, 0] for i in range(5)] for j in range(5)]

# 아래와 같이 곱하기가 아닌 위처럼 for문으로 행렬을 만드는 이유
# 아래처럼 만들면, 각 [ ] 안 같은 위치들끼리 주소를 공유하게 됨
# 따라서 1000을 x[0][0][0]에 넣을지라도
# x[0][0][0], x[1][0][0], x[2][0][0], x[3][0][0], x[4][0][0] 모두 값이 변경됨
x = [[[None]*3]*3]*2
x[0][0][0] = 1000
x

for i in range(5):
    for j in range(5):
        if i == 0 and j == 0:
            가중치누적값[0][0] = c[0][0]
            좌표저장[i][j] = [i, j]
        elif i == 0:
            가중치누적값[i][j] = 가중치누적값[i][j-1] + c[i][j]
            좌표저장[i][j] = [i, j-1]
        elif j == 0:
            가중치누적값[i][j] = 가중치누적값[i-1][j] + c[i][j]
            좌표저장[i][j] = [i-1, j]
        else:
            # 가중치누적값[i][j] = min(가중치누적값[i][j-1], 가중치누적값[i-1][j]) + c[i][j]
            if 가중치누적값[i][j-1] > 가중치누적값[i-1][j]:
                가중치누적값[i][j] = 가중치누적값[i-1][j] + c[i][j]
                좌표저장[i][j] = [i-1, j]
            else:
                가중치누적값[i][j] = 가중치누적값[i][j-1] + c[i][j]
                좌표저장[i][j] = [i, j-1]

c, 가중치누적값
좌표저장


### 2사분면, 4사분면 데이터를 활용해서 연습해보기 ====================================================
# 단, 이번에는 오른쪽 제일 위 8에서 왼쪽 마지막 8로 가는 것이기 때문에 간단한 버전과 방향이 다름

cross = [[[3, 0, 1, 1, 8],
          [5, 0, 4, 5, 4],
          [1, 5, 0, 5, 1],
          [1, 2, 1, 0, 1],
          [0, 2, 5, 1, 1]],
         [[1, 2, 0, 3, 3],
          [1, 2, 0, 2, 4],
          [1, 2, 0, 2, 4],
          [4, 2, 0, 0, 1],
          [8, 4, 1, 1, 0]]]

cross_ = cross[0] + cross[1]
cross_

가중치누적값 = [[0 for i in range(5)] for i in range(len(cross_))]
좌표저장 = [[[0, 0] for i in range(5)] for j in range(len(cross_))]

for i in range(len(cross_)):
    for j in range(4, -1, -1):
        print(i, j)
        if i == 0 and j == 4:
            가중치누적값[0][4] = cross_[0][4]
            # 바로 다음으로 가는 좌표랑 겹치는 숫자가 나오지 않게
            # 초기값을 아예 나올 수 없는 [99, 99]로 지정해 줌
            좌표저장[i][j] = [99, 99]
        elif i == 0:
            가중치누적값[i][j] = 가중치누적값[i][j+1] + cross_[i][j]
            좌표저장[i][j] = [i, j+1]
        elif j == 4:
            가중치누적값[i][j] = 가중치누적값[i-1][j] + cross_[i][j]
            좌표저장[i][j] = [i-1, j]
        else:
            if 가중치누적값[i][j+1] > 가중치누적값[i-1][j]:
                가중치누적값[i][j] = 가중치누적값[i-1][j] + cross_[i][j]
                좌표저장[i][j] = [i-1, j]
            else:
                가중치누적값[i][j] = 가중치누적값[i][j+1] + cross_[i][j]
                좌표저장[i][j] = [i, j+1]

cross_, 가중치누적값
좌표저장

좌표저장[len(cross_)-1][0]

k = 0
while True:
    if k == 0:
        i, j = 좌표저장[len(cross_)-1][0]
    else:
        i, j = 좌표저장[i][j]
    if i == 99 and j == 99:
        break
    k += 1
    print(i, j)
