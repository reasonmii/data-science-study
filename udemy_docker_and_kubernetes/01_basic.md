<b>Image</b>
- template (setup, logic, codes)
  - 읽기/쓰기 access 권한이 있는 instance를 실행하는 container의 blueprint
  - 이미지의 모든 명령은 cache 가능한 layer 생성
    - 레이어는 이미지 재구축, 공유
- name : tag
  - name : defines a group of possible more specialized images ex) "node"
  - tag (optional) : defines a specialized image **within a group** of images ex) "14"
- image push/pull 하는 곳 : Docker Hub, Private Registry

<b>Container</b>
- run based on images
- 여러 컨테이너가 서로 간섭하지 않고 동일한 이미지에 기반해서 실행 가능

<b>Docker Hub</b>
- image 공유 사이트
- ★ 보통 이곳에서 base image 가져오고 내 code를 add 함
- ex) 'node' 검색 - docker official images
  - Terminal : `docker run node`
    - download latest node image automatically
    - run with container
  - `docker run -it node` : interactive node terminal

<b>VS code</b>
- 상단 view - extensions - Docker 검색 - install
- 작업 폴더에 'Dockerfile' 생성
- Terminal - New Terminal
  - `[ctrl] + C 두 번` : 실행 중인 작업 
  - `clear` : terminal 창 깨끗하게

<b>공유</b>
- Everyone who has an image, can create containers based on image!
  - 서로 공유할 때 container가 아닌 image를 공유하는 것
- 1) **Share a Dockerfile** + surrounding files/folders (source code)
  - `docker build .`
- 2) Share a Built Image
  - Download an image, run a container based on it
  - no build step required, everythin gis included in the image already! (source code 필요X)
  - 보통 이 로 대부분 공유해서 작업

<b>Data</b>
- Application (code + environment)
  - read-only, hence stored in images
- Temporary App Data
  - ex) entered user input
  - read + write, temporary, hence stored in containers
- Permanent App Data
  - ex) user accounts
  - read + write, permanent, stored with containers & **volumnes**

<b>External Data Storage</b>
- Volumes : managed by docker
  - anonymous volumes : container가 존재하는 동안만 존재
  - named volumes : container가 shutdown 되어도 존재함
    - 영구적으로 필요한데 edit할 필요는 없는 data에 유용
- Bind Mounts : Managed by you
  - Great for persistent, editable data ex) source code

<b>volumes</b>
- container와는 별개로 host machine hard drive에 있는 것
  - Docker에서 관리하므로 호스트 폴더(컨테이너 내부 경로 매핑)가 어디 있는지 반드시 알 
- container가 shutdown 되어도 살아 있음
  - If a container (re-)start and mounts a volume, any data inside of that volume is available in the container
- container에 mount 해서 사용
- container는 volume data 읽기/쓰기 가능
- 익명 볼륨 : 외부 경로보다 컨테이너 내부 경로의 우선순위를 높이는데 사용

<b>anonymous volume</b>
- Dockerfile `VOLUME` 코드로 생성하거나 `-v`로 생성
  - `VOLUME ["/app/node_modules"]`
  - `-v /app/node_modules`
    - `-v /app/data ...`
- created specifically for a single container
- survives container shutdown/restart unless `--rm` is used
- container 간 공유 불가, 재사용 불가
- 컨테어너에 이미 존재하는 특정 데이터를 잠그는데 유용
- 데이터가 다른 모듈로 덮어쓰기되는 것 방지

<b>named volume</b>
- `-v [volume name]:/app/node_modules`
- Dockerfile에서 생성 불가, `-v`로 생성
- created in general (특정 container X)
- survives container shutdown/restart
  - 삭제 : Docker CLI 사용
- container 간 공유 가능
- restart를 통해 같은 컨테이너에서 재사용 가능

<b>bind mounts</b>
- `-v "[PATH]:/app"
- 유저가 알고 있으며 특정 host machine 상 경로에 있고 일부 컨테이너 내부 경로에 매핑됨
  - Location on host file system, not tied to any specific container
- 목적 : 컨테이너에 live data를 제공하기 위함 (rebuild 필요X)
- survives container shutdown/restart
  - host machine에서 삭제
- container 간 공유 가능
- restart를 통해 같은 컨테이너에서 재사용 가능

<b>ARGuments</b>
- Docker supports build-time ARGuments
- Available inside of Dockerfile, NOT accessible in CMD or any application code
- Set on image build (`docker build`) via `--build-arg`

<b>ENVironment</b>
- Docker supports runtime ENVironment variables
- Available inside of Dockerfile & in application code
- Set via ENV in Dockerfile or via `--env` on `docker run`


