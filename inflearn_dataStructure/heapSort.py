def heapify(list, n, i):
    
    print("")
    print("heapify")
    print("list: ", list, ", n: ", n, ", i: ", i)
    
    root_largest = i
    left = 2 * i + 1    # node
    right = 2 * i + 2   # node
    print("root_largest: ", root_largest, ", left: ", left, ", right: ", right)
    
    # left 값이 존재하고
    # left node > root 일 때
    if left < n and list[i] < list[left]:
        print("left node is bigger. list[i] list[left]: ", list[i], list[left])
        root_largest = left
        
    # right 값이 존재하고
    # right node > root 일 때
    if right < n and list[root_largest] < list[right]:
        print("right node is bigger. list[i] list[right]: ", list[i], list[right])
        root_largest = right
    
    # left, right node와 바꿔야 할 노드를 찾았을 때
    if root_largest != i:
        
        print("list: ", list, ", n: ", n, ", root_largest: ", root_largest)
        list[i], list[root_largest] = list[root_largest], list[i]
        print("list: ", list)
        
        # 계속 heap의 형태를 갖출 때까지 실행
        heapify(list, n, root_largest)
    
    print("They are safe from 'if' condition")


def heapSort(list):
    n = len(list)
    
    # heap의 형태로 데이터 정렬
    for i in range(n, -1, -1):
        print("heapsort heapify")
        heapify(list, n, i)
        
    # root와 마지막 node 값을 비교하고 바꿔서 정렬
    # 바뀐 기준으로 다시 heapify 실행
    for i in range(n-1, 0, -1):
        print("heapsort swap")
        list[i], list[0] = list[0],  list[i]
        heapify(list, i, 0)
    
    print("Heapsort is over")
    

list = [4, 10, 3, 5, 1]
heapSort(list)
