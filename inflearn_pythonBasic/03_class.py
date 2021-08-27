
'''참고강의
Inflearn 파이썬 무료강의 (기본편) - 6시간 뒤면 나도 개발
'''


''' book 어서와 파이썬은 처음이지
객체 지향 프로그래밍 Object-oriented programming
1. 개념
1) 객체(object) : 텔레비전
2) 객체상태(state) = 객체의 속성 : 채널번호, 볼륨, 전원상태
3) 객체동작(behavior) = 객체의 기능 : 켜기, 끄기, 채널 변경, 볼륨 변경

2. In software,
   Object = instance variable (state) + method (behavior)
1) instance variable : 객체 안의 변수 
   int channelNo;
   int volumne;
   bool on Off
2) method : 객체의 동작을 나타내는 부분
   turnOn()
   turnOff()
   changeChannel()
   changeVolume()

3. python에서는 모든 것이 객체
1) Class : 객체에 대한 설계도
2) member : class 내 instance variable과 method

4. 캡슐화 Encapsulation
   클래스에 의해 제공되는 method는 클래스의 공용 인터페이스 (public interface)
   개발자는 어떤 method가 어떻게 작업하는지만 알면 됨
   이렇게 public interface만 제공하고 구현 세부 사항을 감추는 것을 캡슐화라고 함
'''


'''생성자 Constructor
객체가 생성될 때 객체를 기본값으로 초기화하는 특수한 method

예시
아래 counter 클래스에는 인스턴스 변수가 count 뿐
따라서, count만 생성하고 추기화하면 됨
생성자의 첫 번째 매개변수는 self여야 하고, self는 현재 초기화되고 있는 객체임

파이썬 환경에서 주의할 점
1) 변수를 초기화하면 동시에 변수가 생성됨
2) 클래스 당 하나의 생성자만 허용
   이 부분이 불편할 수 있지만 매개변수의 기본값을 줄 수 있는 기능을 사용하면
   어느정도 보완 가능
'''

### -------------------------------------- < Create the Game >
# Explain the 'class' function, using Starcraft
# Unit : Starcraft's characters

from random import *

# Create the 'General Unit'
class Unit:
    def __init__(self, name, hp, speed):
        self.name = name
        self.hp = hp
        self.speed = speed
        print("{} 유닛이 생성되었습니다.".format(name))       
        
    def move(self, location):
        #print("[지상 유닛 이동]")
        print("{} : {} 방향으로 이동합니다. [속도 {}]".format(self.name, location, self.speed))
        
    def damaged(self, damage):
        print("{} : {} 데미지를 입었습니다.".format(self.name, damage))
        self.hp -= damage
        if self.hp <= 0:
            print("{} : 파괴되었습니다.".format(self.name))        

# Unit class - instance (객체) = marin1, marin2, tank    
marin1 = Unit("마린", 40, 15)
marin2 = Unit("마린", 40, 15)
tank = Unit("탱크", 150, 60)
wraith1 = Unit("레이스", 80, 70)
print("유닛 이름 : {}, 속도 : {}".format(wraith1.name, wraith1.speed))

# 변수 추가 가능
# but, 해당 객체에만 추가되고 다른 객체에는 반영 안됨
wraith2 = Unit("빼앗은 레이스", 80, 60)
wraith2.clocking = True

if wraith2.clocking == True:
    print("{}는 현재 클로킹 상태입니다.".format(wraith2.name))


# Create the 'Attack Unit'
### 상속 : 위에서 사용한 class를 다시 사용
### Create Method : attack, damaged, etc.
class AttackUnit(Unit):
    def __init__(self, name, hp, speed, damage):
        Unit.__init__(self, name, hp, speed)    # Unit 초기화
        self.damage = damage
        
    def attack(self, location):
        print("{} : {} 방향으로 적군을 공격 합니다. [공격력 {}]".format(self.name, location, self.damage))

# 지상 unit
firebat1 = AttackUnit("파이어벳", 50, 16, 20)
firebat1.attack("5시")

# 공격 2번 받았다고 가정
firebat1.damaged(20)     # 파이어벳 : 20 데미지를 입었습니다.
firebat1.damaged(30)     # 파이어벳 : 30 데미지를 입었습니다. 파이어벳 : 파괴되었습니다. 


# Create the 'Marine Unit'
class Marine(AttackUnit):
    def __init__(self):
        AttackUnit.__init__(self, "마린", 40, 1, 5)
        
    # 스팀팩 : 일정 시간 동안 이동하고 공격 속도를 증가, 체력 10 감소
    def stimpack(self):
        if self.hp > 10:
            self.hp -= 10
            print("{} : 스팀팩을 사용합니다. (HP 10 감소)".format(self.name))
        else:
            print("{} : 체력이 부족하여 스팀팩을 사용하지 않습니다.".format(self.name))


# Create the 'Tank Unit'            
class Tank(AttackUnit):
    # 시즈모드 : 탱크를 고정시켜 더 높은 파워로 공격 가능, 이동 불가
    seize_developed = False
    
    def __init__(self):
        AttackUnit.__init__(self, "탱크", 150, 1, 35)
        self.seize_mode = False
        
    def set_seize_mode(self):
        
        # 현재 시즈모드가 아닐 때 > 시즈모드
        if self.seize_mode == False:
            print("{} : 시즈모드로 전환합니다.".format(self.name))
            self.damage *= 2
            self.seize_mode = True
                
        # 현재 시즈모드일 때 > 해제
        else:
            print("{} : 시즈모드를 해제합니다.".format(self.name))
            self.damage /= 2
            self.seize_mode = False


# Create the 'Flying Unit'
### 다중 상속
class Flyable:
    def __init__(self, flying_speed):
        self.flying_speed = flying_speed
        
    def fly(self, name, location):
        print("{} : {} 방향으로 날아갑니다. [속도 {}]".format(name, location, self.flying_speed))

# 공중 공격 유닛
class FlyableAttackUnit(AttackUnit, Flyable):
    def __init__(self, name, hp, damage, flying_speed):
        AttackUnit.__init__(self, name, hp, 0, damage)  # 지상 speed = 0
        Flyable.__init__(self, flying_speed)
        
    def move(self, location):
        #print("[공중 유닛 이동]")
        self.fly(self.name, location)        
        
# 공중 unit
valkyrie = FlyableAttackUnit("발키리", 200, 6, 5)
valkyrie.fly(valkyrie.name, "3시")
battlecruiser = FlyableAttackUnit("배틀크루저", 500, 25, 3)
battlecruiser.move("9시")

# 지상 unit
vulture = AttackUnit("벌쳐", 80, 10, 20)
vulture.move("11시")


# Create the 'Wraith Unit'
# It attacks others while flying        
class Wraith(FlyableAttackUnit):
    def __init__(self):
        FlyableAttackUnit.__init__(self, "레이스", 80, 20, 5)
        self.clocked = False
        
    def clocking(self):
        if self.clocked == True:
            # 클로킹 : 상대방이 못 보도록 하는 것
            print("{} : 클로킹 모드 해제합니다.".format(self.name))
            self.clocked == False
        else:
            print("{} : 클로킹 모드 설정합니다.".format(self.name))
            self.clocked == True
        

### pass : 아무 것도 안 하고 일다 넘어간다
# supply 디폿 : 건물 1개 당 8개 유닛 생성 가능
class BuildingUnit(Unit):
    def __init__(self, name, hp, location):
        pass
    
supply_depot = BuildingUnit("서플라이디폿", 500, "7시")


###
def game_start():
    print("[알림] 새로운 게임을 시작합니다.")
    
def game_over():
    print("Player : GG")
    print("[Player] 님이 게임에서 퇴장하셨습니다.")



### -------------------------------------- < Real Game Start >
game_start()

# 마린 3기 생성
m1 = Marine()
m2 = Marine()
m3 = Marine()

# 탱크 2기 생성
t1 = Tank()
t2 = Tank()

# 레이스 1기 생성
w1 = Wraith()

# 유닛 일괄 관리 : 생성된 모든 유닛 append
attack_units = []
attack_units.append(m1)
attack_units.append(m2)
attack_units.append(m3)
attack_units.append(t1)
attack_units.append(t2)
attack_units.append(w1)

# 전군 이동
for unit in attack_units:
    unit.move("1시")
    
# 탱크 시즈모드 개발
Tank.seize_developed = True
print("[알림] 탱크 시즈 모드 개발이 완료되었습니다.")

# 공격 모드 준비
# 마린 : 스팀팩, 탱크 : 시즈모드, 레이스 : 클로킹
for unit in attack_units:
    # 이 unit은 marine class의 instance인지 확인
    if isinstance(unit, Marine):
        unit.stimpack()
    elif isinstance(unit, Tank):
        unit.set_seize_mode()
    elif isinstance(unit, Wraith):
        unit.clocking()
 
# 전군 공격
for unit in attack_units:
    unit.attack("1시")

# 전군 피해
for unit in attack_units:
    unit.damaged(randint(5,21))   # 공격은 랜덤으로 받음 (5~20)
    
# 게임 종료
game_over()



### -------------------------------------- < Super >
# 상속받을 때 활용 가능
# super() 로 쓰고 괄호 써 줘야 하고, self 쓰지 않음
class BuildingUnit(Unit):
    def __init__(self, name, hp, location):
        #Unit.__init__(self, name, hp, 0)  -> super() 쓸 때는 이렇게 쓰면 안 됨
        super().__init(name, hp, 0)
        self.location = location
        

# Super 문제점
# 두 개 이상의 class 상속 받을 때, 순서 상 마지막 class만 상속 받음
# 따라서 둘 다 호출하려면 super 쓰면 안 됨
class Unit:
    def __init__(self):
        print("Unit 생성자")

class Flyable:
    def __init__(self):
        print("Flayable 생성자")

# Super 사용하는 방법
class FlyableUnit(Unit, Flyable):
    def __init__(self):
        super().__init__()
        
dropship = FlyableUnit()    # Unit 생성자

class FlyableUnit(Flyable, Unit):
    def __init__(self):
        super().__init__()
        
dropship = FlyableUnit()    # Flayable 생성자
     
# 실제로 다중 호출 할 때 사용해야 하는 방법
class FlyableUnit(Unit, Flyable):
    def __init__(self):
        Unit.__init__(self)
        Flyable.__init__(self)
        
dropship = FlyableUnit()    # Unit 생성자, Flayable 생성자



### -------------------------------------- < Quiz >
# 부동산 program

class House:
    # 매물 초기화
    def __init__(self, location, house_type, deal_type, price, completion_year):
        self.location = location
        self.house_type = house_type
        self.deal_type = deal_type
        self.price = price
        self.completion_year = completion_year
        
    # 매물 정보 표시
    def show_detail(self):
        print(self.location, self.house_type, self.deal_type, self.price, self.completion_year)

houses = []
house1 = House("강남", "아파트", "매매", "10억", "2010년")
house2 = House("마포", "오피스텔", "전세", "5억", "2007년")
house3 = House("송파", "빌라", "월세", "500/50", "2000년") 

houses.append(house1)
houses.append(house2)
houses.append(house3)

print("총 {}대의 매물이 있습니다.".format(len(houses)))
for house in houses:
    house.show_detail()
    
