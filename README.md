# 드론 자율비행 시뮬레이션 — OpenCV · YOLO · Flow Deck

Crazyflie 2.1+를 대상으로 전방 카메라와 하방 광류·거리·IMU를 이용하는
ROS 2 / Gazebo 자율비행 개발 프로젝트입니다. PC에서 실행하는 시뮬레이션
참조 구현이며 실기체 GAP8/STM32 펌웨어는 포함하지 않습니다.

## 구현한 기능

| 구분 | 구현 내용 |
| --- | --- |
| 1단계 | ArUco 1~6 순차 방문, 알려진 마커 크기의 PnP 위치 추정, 영상 중심 PID 정렬 |
| 2단계 | YOLO 링 검출, OpenCV 타원 윤곽 검증과 광류 추적, 하방 ToF 거리 변화로 통과 확인 |
| 3단계 | 깃발 마커 기반 왕복 차량 추종, 상대 속도에 따른 하강, ToF와 접촉에 의한 착륙 확인 |
| Flow Deck 보조 | 하방 영상·거리·IMU로 속도 계산, 정지 중 미끄러짐 보정, 탐색 높이 유지, 착지 감속 |
| 센서 화면 | 전방 영상, 학습 깊이, 하방 영상·ToF·광류 속도·IMU 및 갱신 상태 표시 |

제공 지도나 Gazebo 정답 위치를 기체 제어 입력으로 사용하지 않습니다.
접촉 정보는 시뮬레이션 착륙 확인에 사용합니다.

## 실행

기준 환경: Ubuntu 24.04 x86_64, Python 3.12, ROS 2 Jazzy, Gazebo Harmonic.
저장소 루트에서, 필요한 의존성이 준비된 환경으로 실행합니다.

```bash
bash scripts/run_all_windows.sh --front-mission-flight --seed 100002 --domain 207
```

1단계만 확인하려면 `--stop-after-stage1`, 창 없이 실행하려면 `--headless`를 추가합니다.
센서 창을 보려면 `--headless`를 사용하지 않습니다.
종료: 터미널 `Ctrl+C` 또는 카메라·센서 창의 `Q`/`Esc`.

새 PC 설치 과정과 기존 검증 기록은 [설치·이전 안내](README_TRANSFER_CURRENT_KO.md),
미션 세부 동작은 [미션 설명](competition/README.md)을 참고하세요.
설치 스크립트는 시스템 패키지와 가상환경을 변경하므로 새 PC에서 내용을 확인하고 실행합니다.
이번 Git 정리 과정에서는 의존성 설치·시뮬레이션·테스트를 실행하지 않았습니다.

## 폴더 구성

```text
competition/                     미션 프로필과 실행 설명
scripts/runtime/                 Gazebo 월드 생성, ROS 연결, 실행 및 센서 화면
scripts/setup/                   설치·빌드·검증 도구
ros2_ws/src/dynamic_drone/        인식·제어·센서 정책과 월드·모델 설정
models/mission_yolo_v3/          링 검출 가중치와 모델 이력
DATA/sensor_only_20261011_050229/depth/
                                실행용 깊이 모델과 학습 메타데이터
assets/                         바닥 시각 자원
vendor/crazyflie-simulation/     기체 메시와 원본 라이선스
tests/                          기존 오프라인 회귀 테스트
docs/                           보고서용 기능·수식, Flow Deck 코드 안내
```

## 모델과 기록 관리

- `models/mission_yolo_v3/rings.pt`: 현재 링 검출 모델, 약 5.5 MB.
- `DATA/sensor_only_20261011_050229/depth/depth.pt`: 기본 실행에 필요한 깊이 모델, 약 4.3 MB.
- 두 모델은 Git에 포함합니다. 학습 데이터셋은 포함하지 않습니다.
- 가상환경, 캐시, ROS 빌드 결과, `.env`, 실행별 `DATA` 로그·영상은 제외합니다.
- 기존 파일을 로컬에서 삭제하거나 이동하지 않습니다. 새 실행 결과도 로컬에 보존됩니다.
- `transfer_manifest.json`은 Git 정리 시 포함한 프로젝트 파일의 SHA-256 목록입니다.
  코드 변경 후에는 이 목록도 갱신해야 기존 `scripts/setup/verify.py`의 무결성 검사가 일치합니다.
- 이전 상세 문서가 참조하는 실행 로그는 로컬 자료이며 Git 복제본에는 없습니다.

## 보고서 자료

- [OpenCV·Flow Deck 기능과 수식](docs/보고서_OpenCV_기능과수식.txt)
- [Flow Deck 코드와 센서 화면](docs/FlowDeck_코드와_센서화면_안내.txt)

## 검증 상태와 라이선스

외부 코드·기체 자원·주요 라이브러리의 출처와 사용 위치는
[오픈소스 출처 및 라이선스](THIRD_PARTY_NOTICES.md)에 정리했습니다.
원문 고지 사본은 [licenses/third_party](licenses/third_party/)에서 확인할 수 있습니다.
전달받은 마커 코드·가중치와 참고 논문 중 출처가 확정되지 않은 부분도 구분해 기록했습니다.

기존 오프라인 검증 기록은 상세 문서와 모델 manifest에 보존되어 있습니다.
전체 미션 성공·최신 변경 후 Gazebo 비행·실기체 성능을 검증 완료로 간주하지 않습니다.
Flow Deck은 하방 카메라·거리 센서·IMU의 모의 구현이며 PMW3901 하드웨어 드라이버가 아닙니다.

외부 기체 자원의 라이선스는 [vendor 라이선스](vendor/crazyflie-simulation/LICENSE)를 보존합니다.
프로젝트 전체에 새 라이선스를 부여하지 않았습니다. 모델의 배포 조건은 해당 출처를 확인해야 합니다.
