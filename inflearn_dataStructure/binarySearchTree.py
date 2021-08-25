class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# 추가하기
def insert(node, val):
    
    # root가 비어있을 때
    # node를 root로 만들기
    if node is None:
        return Node(val)
    
    # root의 val 값이 더 큰 경우
    if val < node.val:
        node.left = insert(node.left, val)
    
    # root의 val 값이 더 작은 경우
    else:
        node.right = insert(node.right, val)
    
    # root 값
    return node

# 작은 순서대로 출력
def inorder(root):
    if root is not None:
        
        inorder(root.left)
        print(root.val)
        inorder(root.right)

# 가장 작은 값 출력
def minimum(node):
    
    # 노드가 가장 왼쪽으로
    while (node.left is not None):
        node = node.left
    
    # 가장 왼쪽 return
    return node

# 삭제하기
def remove(root, val):
    
    # 삭제할 위치 찾기
    
    # 삭제할 값이 root 값보다 작을 경우
    if val < root.val:
        # 왼쪽으로 이동
        root.left = remove(root.left, val)
        
    # 삭제할 값이 root 값보다 클 경우
    elif val > root.val:
        root.right = remove(root.right, val)
    
    # 삭제할 node를 찾아서 작업 시작
    else:
        # 자식이 1개일 경우
        if root.left is None:
            temp_node = root.right
            return temp_node
        elif root.right is None:
            temp_node = root.left
            return temp_node
        
        # 자식이 2개일 경우
        temp_node = minimum(root.right)
        
        # root의 오른쪽에서 가장 왼쪽 작은 값 찾기
        root.val = temp_node.val
        
        # root의 오른쪽에서 가장 왼쪽 작은 값 삭제
        # root 부분에 temp_node.val 값 넣기
        # temp_node 삭제하면 BST 형태가 됨
        root.right = remove(root.right, temp_node.val)
    
    return root

  
root = None
root = insert(root, 50)
root = insert(root, 30)
root = insert(root, 20)
root = insert(root, 40)
root = insert(root, 70)
root = insert(root, 60)
root = insert(root, 80)

inorder(root)   # 20 30 40 50 60 70 80
print("minimum: ", minimum(root))  # 20

root = remove(root, 50)
print("root: ", root.val)
