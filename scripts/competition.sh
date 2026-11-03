#!/usr/bin/env bash
# 파일 역할: 전체 미션 실행 입구와 실행 환경 연결
set -eo pipefail
# 파트 1: 이 프로젝트 경로 확인
competition_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$competition_root"
# 파트 2: 허용된 전체 미션 명령인지 검사
if [[ "${1:-}" != mission-flight ]]; then
  echo '사용법: bash scripts/competition.sh mission-flight [--seed 100002] [--domain 207]' >&2
  exit 2
fi
shift
# 파트 3: ROS·Python 경로 연결
source scripts/activate.sh
export PYTHONPATH="$competition_root/ros2_ws/src/dynamic_drone:$competition_root/scripts:${PYTHONPATH:-}"
# 파트 4: 전체 미션과 화면 실행
exec .venv-ai/bin/python -u scripts/runtime/flight.py --full-mission "$@"
