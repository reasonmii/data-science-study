hash_table = [[] for _ in range(10)]
print(hash_table)

def insert(hash_table, key, value):
    hash_key = key % len(hash_table)
    
    # 중복확인
    key_exists = False
    bucket = hash_table[hash_key]
    print("hash_table: ", hash_table)
    print("key, value, hash_key: ", key, value, hash_key)
    print("bucket1: ", bucket)
    
    # 한 bucket 안에 여러 개가 있는지 반복문으로 돌리면서 확인
    for i, kv in enumerate(bucket):
        k, v = kv
        if key == k:
            # 중복을 확인한 경우
            key_exists = True
            break
    # 중복이 있을 때
    if key_exists:
        bucket[i] = ((key, value))
    else:
        bucket.append((key, value))
        print("bucket2: ", bucket)
    
    
insert(hash_table, 10000000, '민수') # hash_key = 0     
insert(hash_table, 10000005, '철수') # hash_key = 5
insert(hash_table, 20000000, '민지') # hash_key = 0

print(hash_table)
