### python

<b>python 파일 생성</b>
- **사용자 입력**이 필요한 파일로 생성
```
from random import randint

min_number = int(input('Please enter the min number: '))
max_number = int(input('Please enter the max number: '))

if (max_number < min_number): 
  print('Invalid input - shutting down...')
else:
  rnd_number = randint(min_number, max_number)
  print(rnd_number)
```

<b>Dockfile</b>
```
FROM python

WORKDIR /app

COPY . /app

CMD ["python", "rng.py"]
```

<b>Terminal</b>
- `docker build .`
- `docker run [id]` -> error 발생 : 사용자 입력을 받아야 하는 파일이기 때문
  - `docker run --help`
- 실행방법1 : `docker run -it [id]`    
  - `-i` interactive mode : attach mode가 아닌 경우에도 container에 입력 가능
  - `-t` pseudo TTY 할당 : terminal 생성
  - `-it` : container에 입력하면 terminal에 노출
- 실행방법2 : `docker start -a -i [name]`
  - `-a` : attach mode
  - `-i` : interactive mode
- `docker stop [name]`
