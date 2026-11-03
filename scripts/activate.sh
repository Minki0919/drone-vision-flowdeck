#!/usr/bin/env bash
# 파일 역할: 현재 프로젝트의 ROS·Python 실행 환경 연결
set -e

# 파트 1: 현재 체크아웃 경로와 Python 경로 정책
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export PATH="/usr/bin:${PATH}"

# 파트 2: ROS Jazzy 환경 연결
source /opt/ros/jazzy/setup.bash

# 파트 3: 현재 프로젝트의 빌드 결과가 있으면 연결
if [[ -f "${PROJECT_ROOT}/ros2_ws/install/setup.bash" ]]; then
  source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
fi

# 호환 경로: 해당 펌웨어 빌드가 존재할 때만 Python 모듈 경로 연결
FIRMWARE_BUILD="${PROJECT_ROOT}/vendor/crazyflie-firmware/build"
if [[ -f "${FIRMWARE_BUILD}/cffirmware.py" ]]; then
  export PYTHONPATH="${FIRMWARE_BUILD}:${PYTHONPATH:-}"
fi

# 파트 4: 현재 프로젝트의 가상환경 연결
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  source "${PROJECT_ROOT}/.venv/bin/activate"
  VENV_SITE="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages"
  export PYTHONPATH="${VENV_SITE}:${PYTHONPATH:-}"
fi

# Prefer this checkout after a folder move; installed colcon/venv activation
# files may still reference the previous absolute path.
export PYTHONPATH="${PROJECT_ROOT}/ros2_ws/src/dynamic_drone:${PYTHONPATH:-}"

# 파트 5: 프로젝트 위치와 ROS 로그 저장 위치 설정
export DYNAMIC_DRONE_ROOT="${PROJECT_ROOT}"
export ROS_LOG_DIR="${PROJECT_ROOT}/.logs/ros"
mkdir -p "${ROS_LOG_DIR}"
echo "Dynamic Drone environment: ROS ${ROS_DISTRO}, ${PROJECT_ROOT}"
