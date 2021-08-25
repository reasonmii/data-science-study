# 방법1
list = ["a", "b", "c", "d"]
print(list)

list.append("e")
print(list)

list.pop()
print(list)

# 방법2
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        
class Stack:
    
    # Head 만들기
    def __init__(self):
        self.head = None
    
    # Stack 쌓기
    def push(self, data):
        
        # Stack이 비어있을 때
        if self.head is None:
            self.head = Node(data)
        
        # Stack이 차 있을 때
        else:
            new_node = Node(data)
            new_node.next = self.head
            self.head = new_node
            
    # Stack에서 뺄 때
    def pop(self):
        
        # Stack이 비어있을 때
        if self.head is None:
            return Node
        
        # Stack이 차 있을 때
        else:
            popped = self.head.data
            self.head = self.head.next
            return popped

s = Stack()

s.push("a")        
s.push("b") 
s.push("c") 
s.push("d") 
s.push("e") 
print(s)

print(s.pop())  # e
print(s.pop())  # d
print(s.pop())  # c
print(s.pop())  # b
print(s.pop())  # a
