# 영화티켓 모듈 만들기

# 일반 가격
def price(people):
    print("{}명 가격은 {}원 입니다.".format(people, people * 10000))
    
# 조조할인 가격

def price_morning(people):
    print("{}명 조조 할인 가격은 {}원 입니다.".format(people, people * 6000))

# 군인할인 가격
def price_soldier(people):
    print("{}명 군인 할인 가격은 {}원 입니다.".format(people, people * 4000))

