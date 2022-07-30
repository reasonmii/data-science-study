- 사용파일 : data-volumnes-01-starting-setup

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
  - 결과 : 'this is awesome' 이라고 화면에 나옴
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

