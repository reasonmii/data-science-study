def zzogaeggi(list, low, high):
    
    # pivot 값 정하기
    pivot = list[high]
    
    # i는 pivot을 기준으로 list 정렬
    i = low - 1
    
    print("")
    print("pivot: ", pivot, "==========================")
    print("low: ", low, ", high: ", high, "i: ", i)
    
    # j를 통해서 lsit 훑기
    for j in range(low, high):
        print("i1: ", i, ", j1: ", j, ", list1: ", list)
        if list[j] < pivot:
            i += 1
            list[i], list[j] = list[j], list[i]
            print("i2: ", i, ", j2: ", j, ", list2: ", list)
    
    # pivot이 들어갈 위치 바꾸기
    list[i+1], list[high] = list[high], list[i+1]
    print("list3: ", list)
    
    # pivot의 위치
    return i + 1

    
def quickSort(list, low, high):
    
    if low < high:
        
        # pivot 기준으로 쪼개기 위해 pivot 위치 가져오기
        pivot_position = zzogaeggi(list, low, high)
        
        quickSort(list, low, pivot_position -1)
        quickSort(list, pivot_position, high)
        

list = [10, 80, 30, 90, 40, 50, 70]
n = len(list)
quickSort(list, 0, n-1)

print(list)
