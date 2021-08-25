def mergeSort(list):
    
    # 완전히 다 쪼개질 때까지 실행
    if len(list) > 1:
        
        # 쪼개기
        mid = len(list) // 2
        
        left = list[:mid]
        right = list[mid:]
        
        print("")
        print("left1: ", left, ", right1: ", right)

        # 왼쪽, 오른쪽으로 쪼개기        
        mergeSort(left)
        mergeSort(right)
        
        print("")
        print("left2: ", left, ", right2: ", right)
        
        # 합치는 과정
        i = 0; j = 0; k = 0
        
        print("i1: ", i, ", j1: ", j, ", k1: ", k)
        
        # 일단 임시로 정렬
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                list[k] = left[i]
                i += 1
            else:
                list[k] = right[j]
                j += 1    
            k += 1
            print("list1: ", list)
            print("i2: ", i, ", j2: ", j, ", k2: ", k, ", left3: ", left, ", right3: ", right)
        
        while i < len(left):
            list[k] = left[i]
            i += 1
            k += 1
        
        print("i3: ", i, ", j3: ", j, ", k3: ", k, ", left4: ", left, ", right4: ", right)
        
        while j < len(right):
            list[k] = right[j]
            j += 1
            k += 1  
        
        print("i4: ", i, ", j4: ", j, ", k4: ", k, ", left5: ", left, ", right5: ", right)
        print("list2: ", list)

list = [38, 27, 43, 3, 9, 82, 10]
mergeSort(list)
print(list)
