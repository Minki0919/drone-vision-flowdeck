#!/usr/bin/env bash
# 파일 역할: 현재 ROS 패키지 빌드
set -e

# 파트 1: 프로젝트 위치와 Python 실행 환경
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONNOUSERSITE=1
export PATH="/usr/bin:${PATH}"
# 파트 2: ROS Jazzy 환경 연결
source /opt/ros/jazzy/setup.bash
cd "${PROJECT_ROOT}/ros2_ws"
# 파트 3: 소스 패키지 빌드; 비행은 실행하지 않음
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
