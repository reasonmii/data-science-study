<b>ENVironment variables 지정하기</b>
- 방법1) `run` command에서 직접 작성
  - `--env [변수]=[값]`, `-e [변수]=[값]`
  - 환경변수가 여러 개인 경우 : `-e [변수]=[값] -e [변수]=[값]` 이런 식으로 계속 이어 쓰면 됨
- 2) '.env' 파일 만들어서 작성
  - 내용 입력 : `PORT=8000`
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

