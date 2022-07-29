### code list
- `docker --help`

<b>보기</b>
- `docker ps` : 실행 중인 모든 container
- `docker ps -a` : all containers from past
- `docker images` : 모든 images (repo, tag, id, created time, size)
- `docker image inspect [image id]` : 이미지 구성 상세 확인
  - image 전체 id, 생성시간
  - 이미지를 기반으로 실행될 container 구성,  ENTRYPOINT
  - 사용 중인 OS
  - 이미지 내 여러 layer들

<b>실행</b>
- `docker run [id]`
- `docker start [name]`

<b>중단</b>
- `docker stop [name]`

<b>삭제</b>
- `docker rm [name1] [name2] ...` : container 삭제
  - `docker ps -a` 하면 너무 많아서 지저분하니까 정리하기 위함
  - running container인 경우 먼저 stop 필요
- `docker rmi [image id1] [id2] ...`
  - `docker images`로 id 확인
  - 이미지 내부 모든 레이어 삭제
  - 관련 container가 먼저 삭제되어 있어야 image 삭제 가능
- `docker image prune`
  - 현재 실행 중인 container에서 사용되지 않는 모든 image 삭제
- ★ `docker run --rm [id]` : container 실행 중지되면 아예 삭제
   - ex) `docker run -p 3000:80 -d --rm [image id]`
     - `-d` : start a container in detached mode
     - `--rm` : automatically remove the container when it exists
   - `docker stop [name]`
   - `docker ps -a` : container 이력에 없음

<b>복사</b>
- `docker cp [복사할 파일] [목적지]` : 실행 중인 컨테이너나 실행 중인 컨테이너 밖으로 폴더/파일 복사 (copy)
- 1) local 폴더에서 컨테이너로 복사 : `[container name]:/[new name]`
  - ex) `docker cp dummy/. boring_vaughan:/test`,
    - `dummy/.` : dummy 폴더 내 모든 파일
  - ex) 'docker cp dummy/test.txt boring_vaughan:/test`  
- 2) 컨테이너에서 local 폴더로 복사 : `[local folder]`
  - ex) `docker cp boring_vaughan:/test dummy`
    - 'dummy' local 폴더에 test 폴더 생성하고 안에 있는 'test.txt' 파일 복사
  - ex) `docker cp boring_vaughan:/test/test.txt dummy`
    - 'dummy' local 폴더에 'test.txt' 파일 복사
- container가 log file을 많이 생성하는 경우 `docker cp`로 local에 옮겨서 자세히 보고  가능

  
