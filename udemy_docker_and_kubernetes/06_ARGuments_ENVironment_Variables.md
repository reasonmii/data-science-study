### ENVironment variables

<b>ENVironment variables 지정하기</b>
- 방법1) `run` command에서 직접 작성
  - `--env [변수]=[값]`, `-e [변수]=[값]`
  - 환경변수가 여러 개인 경우 : `-e [변수]=[값] -e [변수]=[값]` 이런 식으로 계속 이어 쓰면 됨
- 방법2) '.env' 파일 만들어서 작성
  - 환경 변수 데이터가 보안 데이터이고 Dockerfile에 포함하고 싶지 않은 경우
    - 자격 증명, 개인 키 등
    - 이렇게 하지 않으면 값이 image에 포함되어 모든 유저가 `docker history [image]`를 통해 값 확인 가능
  - source control 사용하는 경우 해당 파일을 source control 저장소의 일부분으로 commit 하지 않도록 주의
  - '.env' 파일 내용 입력 : `PORT=8000`
  - `run` command : `--env PORT=8000` 대신 `--env-file ./.env` 작성

<b>'PORT'를 환경변수로 지정하기</b>
- server.js : 마지막 `app.listen(80);` 변경 -> `app.listen(process.env.PORT);`
- Dockerfile
  - `PORT` 변수 선언 (`COPY . .` 밑에 작성) : `ENV PORT 80` (default 값 설정하는 것)
  - `EXPOSE 80` 수정 -> `EXPOSE $PORT`
    - `$` 의미 : environment variables
- Terminal
  - `docker build -t feedback-node:env .`
  - `docker run -d --rm  -p 3000:80 --name feedback-app -v feedback:/app/feedback -v "/Users/reasonmii/udemy_docker/data-volumes-01-starting-setup:/app:ro" -v /app/temp -v /app/node_modules  feedback-node:env`
  - `docker stop feedback-app`
- hard code `80`이 아닌 `$PORT`를 사용했기 때문에 PORT 값 직접 지정도 가능
  - `docker run -d --rm -p 3000:8000 --env PORT=8000 --name feedback-app -v feedback:/app/feedback -v "/Users/reasonmii/udemy_docker/data-volumes-01-starting-setup:/app:ro" -v /app/temp -v /app/node_modules  feedback-node:env`
    - `--env PORT=8000` : PORT값을 8000으로 한다
    - `-p 3000:8000` : 8000으로 연결한다
  - 결과 : 에러 없이 잘 작동

---

### ARGuments
- 위왁 같이 환경변수 사용하는 경우 Dockerfile에 `ENV PORT 80`으로 default 값이 고정되어 있음

<b>default 값 고정하기 싫은 경우</b>
- Dockerfile
  - `FROM node:14` 아래에 `ARG DEFAULT_PORT=80` 작성
    - 이 argument는 Dockerfile 내부에서 `CMD` 명령 부분 빼고 모든 곳에서 사용 가능
  - 수정 : `ENV PORT 80` -> `ENV PORT $DEFAULT_PORT`
- Terminal
  - 같은 이미지에 대해 아래와 같이 PORT 값 다르게 해서 여러 버전 만들기 가능
    - `docker build -t feedback-node:web-app .`
    - `docker build -t feedback-node:dev --build-arg DEFAULT_PORT=8000 .`
