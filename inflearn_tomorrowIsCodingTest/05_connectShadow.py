
# 문제5 : 그림자연결

graph = {100: set([67, 66]),
         67: set([100, 82, 63]),
         66: set([100, 73, 69]),
         82: set([67, 61, 79]),
         63: set([67]),
         73: set([66]),
         69: set([66, 65, 81]),
         61: set([82]),
         79: set([82, 87, 77]),
         65: set([69, 84, 99]),
         81: set([69]),
         87: set([79, 31, 78]),
         77: set([79]),
         84: set([65]),
         99: set([65]),
         31: set([87]),
         78: set([87])
        }

# 깊이우선탐색 : stack
# 너비우선탐색 : queue

def 깊이우선탐색(graph, start):
    방문 = []
    stack = [start]
    
    while stack:
        # 너비우선의경우 : pop(0)
        print(f'stack1: {stack}')
        n = stack.pop()
        if n not in 방문:
            방문.append(n)
            # 부모노드 빼기
            # = 이미 '방문'에 있는 node 모두 제거
            차집합 = graph[n] - set(방문)
            print(f'차집합: {차집합}')
            
            if len(차집합) == 0:
                방문 += stack
                break
            
            # stack.append(min(차집합))
            stack.append(max(차집합))
            
            print(f'visited: {방문}')
            print(f'stack2: {stack}')
            
    return 방문

깊이우선탐색(graph, 100)


# ASCII code로 변환
''.join([chr(i) for i in [66, 69, 65, 84]])
''.join([chr(i) for i in [67, 82, 79, 87, 78]])

