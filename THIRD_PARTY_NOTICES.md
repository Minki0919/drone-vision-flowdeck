# 오픈소스 출처 및 라이선스

확인일: 2026-11-03. 코드의 기존 출처 주석, 설치 패키지 메타데이터와 원본 저장소를 기준으로 정리했습니다.
직접 포함한 외부 파일과 실행 시 설치하는 라이브러리를 구분합니다.
이 목록은 주요 구성 요소의 출처 기록이며 모든 전이 의존성의 완전한 목록은 아닙니다.

## 1. 저장소에 직접 포함한 외부 자원

### Bitcraze Crazyflie 시뮬레이션 자원

- 원본: [bitcraze/crazyflie-simulation](https://github.com/bitcraze/crazyflie-simulation)
- 기존 모델 문서에 기록된 원본 커밋: `7e93752dbc803af2488c1db46bb79b3da55f5d8c`
- 원본 모델: [simulator_files/gazebo/crazyflie/model.sdf](https://github.com/bitcraze/crazyflie-simulation/blob/7e93752dbc803af2488c1db46bb79b3da55f5d8c/simulator_files/gazebo/crazyflie/model.sdf)
- 포함 위치: `vendor/crazyflie-simulation/meshes/`의 기체·프로펠러 메시, 텍스처, Blender 자원.
- 파생 모델: `ros2_ws/src/dynamic_drone/models/crazyflie_2_1/model.sdf` 및 관련 Gazebo 월드.
- 라이선스: MIT. 원본 고지는 [vendor/crazyflie-simulation/LICENSE](vendor/crazyflie-simulation/LICENSE)에 그대로 보존했습니다.
- 원본 라이선스 파일의 저작권 표기: Copyright (c) 2022 Bitcraze.
- 프로젝트 적용 내용: 토픽 이름, 접촉 센서, 초기 위치, 제어 이득, 메시 경로와 미션 환경 설정.
- 자세한 변경 기록: [기체 모델 README](ros2_ws/src/dynamic_drone/models/crazyflie_2_1/README.md).

원본 커밋과 모델 경로는 기존 문서와 공개 원본에서 확인했습니다.
모든 로컬 메시의 원본 대비 무수정 여부까지 확인한 것은 아닙니다.
현재 포함 파일의 정확한 바이트는 `transfer_manifest.json`의 SHA-256으로 식별합니다.

## 2. 설치해서 사용하는 주요 라이브러리

아래 라이브러리의 전체 소스나 가상환경은 이 저장소에 복사하지 않습니다.
설치된 배포본에서 확인한 라이선스·고지 원문은 [licenses/third_party](licenses/third_party/)에 보존했습니다.
사본의 패키지 버전, 원래 파일 경로와 해시는 [라이선스 목록](licenses/third_party/index.json)에 기록했습니다.

| 구성 요소 | 공식 출처 | 확인한 버전 / 라이선스 | 사용 위치·목적 |
| --- | --- | --- | --- |
| OpenCV | [opencv/opencv](https://github.com/opencv/opencv), [opencv/opencv-python](https://github.com/opencv/opencv-python) | 설치 wheel `5.0.0.93`; OpenCV 본체 Apache-2.0, Python 패키징의 보존된 LICENSE는 MIT, 포함 제3자 고지 별도 | ArUco, PnP, 윤곽·타원, 광류, 센서 화면 |
| Ultralytics | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | `8.4.174`, 설치 배포본 AGPL-3.0 | `perception/yolo.py`의 YOLO 추론 |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | `2.10.0+cu128`, BSD-3-Clause 및 포함 고지 | `perception/depth.py`의 깊이 신경망 |
| torchvision | [pytorch/vision](https://github.com/pytorch/vision) | `0.25.0+cu128`, BSD-3-Clause | 모델 실행 환경 의존성 |
| NumPy | [numpy/numpy](https://github.com/numpy/numpy) | `2.5.2`, 메타데이터: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | 행렬·좌표·광류·제어 수치 연산 |
| PyYAML | [yaml/pyyaml](https://github.com/yaml/pyyaml) | `6.0.3`, MIT | 설정·ROS 브리지 YAML 처리 |
| ROS 2 rclpy | [ros2/rclpy](https://github.com/ros2/rclpy) | 설치된 Jazzy `package.xml`: Apache License 2.0 | ROS 노드·구독·발행 |
| ros_gz_bridge | [gazebosim/ros_gz](https://github.com/gazebosim/ros_gz) | 설치된 Jazzy `package.xml`: Apache 2.0 | Gazebo↔ROS 메시지 연결 |
| gz_sim_vendor | [gazebo-release/gz_sim_vendor](https://github.com/gazebo-release/gz_sim_vendor) | 설치된 Jazzy `package.xml`: Apache License 2.0 | Gazebo 실행 환경 공급 패키지 |

설치 환경과 requirements에 적힌 패키지가 반드시 모두 동일하게 설치된 것은 아닙니다.
`opencv-contrib-python`은 requirements에 선언되어 있지만 이번 환경의 별도 배포 메타데이터에서는
확인되지 않아 설치된 것으로 표시하지 않았습니다.

공식 라이선스 안내: [OpenCV](https://opencv.org/license/), [Ultralytics](https://www.ultralytics.com/license),
[PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE).
출처 표기는 라이선스 조건을 대체하지 않습니다. 특히 Ultralytics의 설치 배포본은 AGPL-3.0이므로
프로젝트 전체가 일괄 MIT라고 해석하지 않습니다. 별도 상용 라이선스 보유 여부는 확인되지 않았습니다.

## 3. 전달받은 코드·모델 및 미확인 출처

| 항목 | 현재 확인된 근거 | 확인되지 않은 부분 |
| --- | --- | --- |
| `perception/aruco_detection.py` | 파일 주석에 사용자 제공 `ArUco_marker_detection.py`의 영상 탐지 부분을 통합했다고 기록. 공통 사전·서브픽셀 보정 사용, 고정 토픽·해상도·ID별 크기는 그대로 가져오지 않음 | 원본 작성자, 저장소 URL, 원본 버전, 사용 허락·라이선스 |
| `models/mission_yolo_v3/rings.pt` | 기존 manifest의 전달 파일 `YOLO_ring_v2/weights/best.pt`, SHA-256 `a68ce7070a34e46e952f7a80f6afebc172a29dd1a61944c5b4a654f549674e47`; 실행 프레임워크는 Ultralytics | 전달받은 가중치의 별도 배포 조건, 초기 가중치·데이터셋의 정확한 출처와 사용 조건 |
| `perception/depth.py`, `DATA/sensor_only_20261011_050229/depth/depth.pt` | 코드에 논문에서 착안한 소형 흑백 encoder-decoder이며 공개 모델 자체가 아니라고 기록. 학습 메타데이터의 `paper_reproduction=false` | 참고 논문의 정확한 제목·저자·DOI/URL. 특정 논문이나 외부 사전학습 가중치 출처로 임의 단정하지 않음 |

확인되지 않은 항목은 오픈소스 사용 허락이 확인된 것으로 표시하지 않았습니다.
원본 주소·저자·라이선스를 확보하면 이 표와 해당 파일 주석을 함께 보완해야 합니다.

## 4. 프로젝트에서 작성·통합한 부분의 범위

미션 상태 전이, 영상 중심 PID, Flow Deck 모의 보조, 센서 정책, 실행 스크립트,
센서 화면 및 보고서 문서는 이 프로젝트의 구현·통합 영역입니다.
이 설명은 기존 모든 파일의 단독 저작권이나 완전한 독창성을 증명하는 진술이 아닙니다.
기존 `package.xml`/`setup.py`에는 MIT 표기가 있으나, 이번 정리에서 전체 프로젝트에 새 라이선스를
부여하거나 외부 자원의 원래 라이선스를 변경하지 않았습니다.

## 5. 앞으로 외부 코드를 추가할 때 남길 기록

원본 저장소·파일 URL, 커밋/버전, 작성자·저작권 고지, 라이선스 원문 위치,
복사한 로컬 경로, 수정 내용과 날짜를 기록하고 원본 고지를 보존합니다.
라이브러리를 호출하는 경우와 코드를 직접 복사·수정한 경우를 구분합니다.
