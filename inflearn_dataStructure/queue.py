class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# Queue 만들기
class Queue:
    
    def __init__(self):
        self.haed = None
        self.tail = None
    
    # 비어 있는지 확인
    def isEmpty(self):
        if self.head is None:
            return True
        else:
            return False
    
    # add
    def enqueue(self, data):
        
        # Queue가 비어 있을 때
        if self.tail is None:
            self.head = self.tail = Node(data)
        
        # Queue가 차 있을 때
        else:
            # 새로운 node를 생성해서 뒤에 붙이고
            self.tail.next = Node(data)
            # tail의 위치를 조정
            self.tail = self.tail.next
    
    # remove
    def dequeue(self):
    
        # Queue가 비어 있을 때
        if self.head is None:
            return None
        
        # Queue가 차 있을 때
        else:
            dequeue_data = self.head.data
            self.head = self.head.next
            return dequeue_data

q = Queue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)
print(q.dequeue())
print(q.dequeue())
print(q.dequeue())
print(q.isEmpty())
print(q.dequeue())
print(q.dequeue())
