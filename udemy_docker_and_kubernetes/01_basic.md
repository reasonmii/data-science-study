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
- Bind Mounts : Managed by you

<b>volumes</b>
- container와는 별개로 host machine hard drive에 있는 것
- container가 shutdown 되어도 살아 있음
  - If a container (re-)start and mounts a volume, any data inside of that volume is available in the container
- container에 mount 해서 사용
- container는 volume data 읽기/쓰기 가능
