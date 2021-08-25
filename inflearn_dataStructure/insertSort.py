def insertSort(list):
    
    for i in range(1, len(list)):
        key = list[i]
        j = i - 1
        
        print("key :", key)
        
        while j >= 0 and key < list[j]:
            list[j+1] = list[j]
            print("list1 :", list)
            j -= 1
        
        print("list2 :", list)
        list[j + 1] = key
        print("list3 :", list)
        

list = [12, 11, 13, 5, 6]
insertSort(list)
print(list)
