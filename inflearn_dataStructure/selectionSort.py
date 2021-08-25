def selectionSort(list):
    
    # -1 : 마지막 숫자는 자동적으로 배치가 완료되기 떄문
    for i in range(len(list) - 1):
        
        print()
        print("i: ", i)
        # 가장 작은 값의 위치
        min_index = i
        
        # 리스트에 있는 값을 하나하나 보며
        # 가장 작은 값 찾기
        for j in range(i+1, len(list)):
            
            if list[min_index] > list[j]:
                
                print("list[min_index]: ", list[min_index], ", list[j]: ", list[j])
                
                min_index = j
        
        list[i], list[min_index] = list[min_index], list[i]
        print("list: ", list)
        

list = [64, 25, 12, 22, 11]
selectionSort(list)
print(list)
