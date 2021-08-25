def bubbleSort(list):
    
    # 크기 = 5
    length = len(list)
    
    for i in range(0, length - 1):
        
        swap = False
        
        for j in range(0, length - 1 - i):
            
            print("list1: ", list, "i = ", i, ", j = ", j)
            
            if(list[j] > list[j + 1]) :
                list[j], list[j + 1] = list[j + 1], list[j]
                print("list2: ", list)
                swap = True
        
        # 이미 list 정렬이 잘 되어 있는 경우
        # if 조건에 걸리지 않아서, swap은 여전히 false
        if swap == False:
            break
            
    
list = [5,1,4,2,8]
bubbleSort(list)
print(list)
