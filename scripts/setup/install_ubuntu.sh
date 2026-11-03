#!/usr/bin/env bash
# 파일 역할: 새 Ubuntu PC에 필요한 의존성과 가상환경 설치
# Run only on the destination PC. No system upgrade, driver replacement, or ~/.bashrc edits.
set -e
# 파트 1: 프로젝트 위치와 지원 OS 확인
transfer_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$transfer_root"
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 24.04 || "$(uname -m)" != x86_64 ]]; then
  echo 'This bundle requires Ubuntu 24.04 x86_64 / Python 3.12. Do not reuse it unchanged on 22.04 or ARM.' >&2
  exit 1
fi
if [[ ! -f transfer_manifest.json || ! -f requirements-transfer-ai.txt ]]; then
  echo 'Run this from the EXTRACTED transfer bundle, not the original training directory.' >&2
  exit 1
fi
if [[ "${1:-}" == --check-platform ]]; then
  echo 'Supported destination: Ubuntu 24.04 x86_64'; exit 0
fi
echo 'Installing ROS Jazzy + Gazebo Harmonic and isolated Python environments on this PC.'
# 파트 2: 설치 권한 확인과 시스템 의존성 준비
sudo -v
sudo apt-get update
sudo apt-get install -y curl ca-certificates software-properties-common python3 python3-venv
sudo add-apt-repository -y universe
# 파트 3: ROS 저장소가 없을 때만 등록
if ! apt-cache show ros-jazzy-ros-base >/dev/null 2>&1; then
  transfer_temp="$(mktemp -d -t drone-ros-source-XXXXXXXX)"
  echo "ROS repository setup files: $transfer_temp"
  curl --proto '=https' --tlsv1.2 -fsSL \
    https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    -o "$transfer_temp/release.json"
  transfer_deb_url="$(python3 -c 'import json,sys; a=json.load(open(sys.argv[1]))["assets"]; print(next(x["browser_download_url"] for x in a if x["name"].startswith("ros2-apt-source_") and x["name"].endswith("noble_all.deb")))' "$transfer_temp/release.json")"
  if [[ "$transfer_deb_url" != https://github.com/ros-infrastructure/ros-apt-source/releases/download/* ]]; then
    echo 'Unexpected ROS repository package URL; stopped.' >&2; exit 1
  fi
  curl --proto '=https' --tlsv1.2 -fsSL "$transfer_deb_url" -o "$transfer_temp/ros2-apt-source.deb"
  sudo dpkg -i "$transfer_temp/ros2-apt-source.deb"
fi
sudo apt-get update
# 파트 4: ROS·Gazebo·영상 처리 의존성 설치
sudo apt-get install -y \
  ros-jazzy-ros-base ros-jazzy-ros-gz ros-jazzy-ros-gz-interfaces \
  ros-jazzy-gz-sim-vendor ros-jazzy-gz-tools-vendor ros-jazzy-gz-gui-vendor \
  ros-jazzy-geometry-msgs ros-jazzy-nav-msgs ros-jazzy-sensor-msgs \
  ros-jazzy-rosgraph-msgs ros-jazzy-tf2-msgs ros-jazzy-std-srvs \
  python3-colcon-common-extensions python3-setuptools python3-yaml \
  python3-numpy python3-opencv python3-psutil python3-pytest \
  libgl1 libegl1 libgl1-mesa-dri libxcb-cursor0 mesa-utils
# 파트 5: 이 프로젝트 안에 독립 가상환경 구성
for transfer_env in .venv .venv-ai; do
  if [[ -e "$transfer_env" && ! -f "$transfer_env/pyvenv.cfg" ]]; then
    echo "Refusing to replace non-venv path: $transfer_env" >&2; exit 1
  fi
  python3 -m venv --system-site-packages "$transfer_env"
done
# Apt's full OpenCV 4.6 supplies preview/HOG. Do not pip-install OpenCV in .venv.
# 파트 6: Python 영상 처리·모델 추론 의존성 설치
.venv/bin/python -m pip install 'setuptools==79.0.1' 'numpy==1.26.4' 'PyYAML==6.0.3'
.venv-ai/bin/python -m pip install 'setuptools==79.0.1'
transfer_torch_version="$(python3 -c 'import json; print(json.load(open("transfer_manifest.json"))["runtime_versions"]["torch"])')"
.venv-ai/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==$transfer_torch_version" 'torchvision==0.25.0+cpu'
.venv-ai/bin/python -m pip install -r requirements-transfer-ai.txt
# Do not source activate.sh until colcon has built THIS checkout.
source /opt/ros/jazzy/setup.bash
# 파트 7: 현재 체크아웃의 ROS 패키지 빌드
colcon build --base-paths ros2_ws/src --build-base ros2_ws/build \
  --install-base ros2_ws/install --packages-select dynamic_drone --symlink-install
# 파트 8: 환경 연결 후 파일·모듈 검사; 비행은 자동 실행하지 않음
source scripts/activate.sh
.venv-ai/bin/python scripts/setup/verify.py
echo '설치 완료. 시험비행은 자동으로 시작하지 않았습니다.'
echo '실행: bash scripts/run_all_windows.sh --front-mission-flight --seed 100002'
