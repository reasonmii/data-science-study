<b>Image</b>
- template (setup, logic, codes)
  - 읽기/쓰기 access 권한이 있는 instance를 실행하는 container의 blueprint
  - 이미지의 모든 명령은 cache 가능한 layer 생성
    - 레이어는 이미지 재구축, 공유
- name : tag
  - name : defines a group of possible more specialized images ex) "node"
  - tag (optional) : defines a specialized image **within a group** of images ex) "14"

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
