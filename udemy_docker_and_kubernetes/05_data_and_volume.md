사용파일 : data-volumnes-01-starting-setup

<b>Dockerfile 생성</b>
```
FROM node:14

WORKDIR /app

COPY package.json .

RUN npm install

COPY . .

EXPOSE 80

CMD ["node", "server.js"]
```

<b>Terminal</b>
- `docker build -t feedback-node .`
- `docker run -p 3000:80 -d --name feedback-app --rm feedback-node:latest`
  - `-d` : detach mode로 실행 - terminal은 계속 사용하기 위함
  - `--name` : container name
  - `--rm` : automatically remove container

<b>인터넷</b>
- 주소창 : http://localhost:3000/
- 화면에서 title, document text 작성 - save
  - ex) title : awesome, text : this is awesome
- 주소창 : http://localhost:3000/feedback/awesome.txt
  - 화면결과 : this is awesome
  - 'server.js' 파일에 `app.use('/feedback', express.static('feedback'));` 코드 덕분
  - ★ but, Docker container에만 존재하는 것
    - local 'feedback' 폴더에서는 안 보임
    - 기존 코드를 Dockerfile `COPY . .`으로 local에 복사한 후 화면에서 작성한 내용이니 local에 연동 안 되어 있음
    - = container는 isolated 되어 있음
- title 또 'awesome' 입력하는 경우 : 이미 존재하는 이름이라고 알림 (overwritten 방지)
  - 'server.js' 파일에 `await fs.writeFile(tempFilePath, content); ~~` 

<b>container 새로 시작하는 경우</b>
- `docker stop feedback-app`
- `docker run -p 3000:80 -d --name feedback-app feedback-node:latest`
- 인터넷 주소창 : http://localhost:3000/
  - http://localhost:3000/feedback/awesome.txt : error 발생
    - Cannot GET /feedback/awesome.txt
    - `stop` 때문이 아니라 위에서 이전에 `run` 할 때 `--rm`으로 container를 삭제했기 때문
    - 지금 `run` 코드에는 `--rm` 없음
  - 다시 화면으로 돌아가서 'awesome', 'this is awesome' 입력
    - 결과확인 : http://localhost:3000/feedback/awesome.txt
- `docker stop feedback-app`
- `docker start feedback-app`
- 인터넷 주소창 : http://localhost:3000/feedback/awesome.txt 입력
  - `--rm`으로 container 제거한 적이 없으니 데이터 보존되어 있음

---

### volume
- 온라인 화면에서 입력한 데이터 persist 하게 만들기

<b>Dockerfile 생성</b>
- `VOLUME` : 'feedback' 내용을 inside of my container에 저장
  - 'feedback' 내용 : 인터넷 주소창 들어간 후 입력 내용
  - server.js 코드 `const finalFilePath = path.join(__dirname, 'feedback', adjTitle + '.txt');` 부분 참고

```
FROM node:14

WORKDIR /app

COPY package.json .

RUN npm install

COPY . .

EXPOSE 80

VOLUME [ "/app/feedback" ]

CMD ["node", "server.js"]
```

<b>방법1 : 실패</b>
- Terminal
  - `docker build -t feedback-node:volumes .`
  - `docker run -d -p 3000:80 --rm --name feedback-app feedback-node:volumes`
- 인터넷 : 에러발생
  - 주소창 : http://localhost:3000/
  - 화면에서 title, document text 작성 - save
    - ex) title : awesome, text : one more time!
    - 계속 loading 중이고 작동이 안 됨

<b>방법2 : 실패</b>
- Terminal : 문제해결
  - `docker logs feedback-app` 결과로 문제 확인
    - UnhandledPromiseRejectionWarning: cross-device link not permitted, rename '/app/temp/hello.txt' -> '/app/feedback/hello.txt'
    - 'server.js' 파일에서 `await fs.rename(tempFilePath, finalFilePath);` 이 부분 때문
    - 수정 : `await fs.copyFile(tempFilePath, finalFilePath);`
      - final에 경로 복사하고 temp 경로는 삭제
    - 아랫줄 추가 : `await fs.unlink(tempFilePath);`
  - `docker stop feedback-app`
  - `docker rmi feedback-node:volumes`
  - `docker build -t feedback-node:volumes .`
  - `docker run -d -p 3000:80 --rm --name feedback-app feedback-node:volumes`
- 인터넷 주소창 : http://localhost:3000/
  - 화면에서 title, document text 작성 - save
    - ex) title : awesome, text : one more time!
  - 주소창 : http://localhost:3000/feedback/awesome.txt
    - 결과 : one more time
- container 삭제/중단 후 재실행
  - `docker stop feedback-app`
    - 앞에 코드에서 `--rm` 있었으니 중단하면 container 삭제됨
  - `docker run -d -p 3000:80 --rm --name feedback-app feedback-node:volumes`
    - 이전과 같은 image를 기반으로 하지만 새로운 container 생성
  - 주소창 : http://localhost:3000/feedback/awesome.txt
    - error 발생 : 결과 안 나옴
- `docker volume ls`
- `docker stop feedback-app`

<b>방법3 : 성공!</b>
- Terminal
  - `docker build -t feedback-node:volumes .`
  - named volume 생성 : `docker run -d -p 3000:80 --rm --name feedback-app -v feedback:/app/feedback feedback-node:volumes`
- 인터넷 주소창 : http://localhost:3000/
  - 화면에서 title, document text 작성 - save
    - ex) title : awesome, text : awesome!
  - 주소창 : http://localhost:3000/feedback/awesome.txt
    - 결과 : awesome!
- Terminal
  - `docker stop feedback-app`
    - `run` 할 때 `--rm` 옵션 사용했으니 container 삭제됨
  - `docker volume ls` : container 없어졌지만 여전히 'feedback' volume 존재
  - 같은 volume name과 경로 사용해서 다시 `run`
    - `docker run -d -p 3000:80 --rm --name feedback-app -v feedback:/app/feedback feedback-node:volumes`
- 인터넷 주소창 : http://localhost:3000/
  - 주소창 : http://localhost:3000/feedback/awesome.txt
  - 아까 입력했던 'awesome!' 그대로 있음

