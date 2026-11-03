#!/usr/bin/env bash
# 파일 역할: 프로젝트 위치 확인 후 전체 실행 옵션 전달
set -euo pipefail
# 파트 1: 실행 위치를 이 프로젝트로 고정
all_windows_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$all_windows_root"
# 파트 2: 독립 AI 가상환경으로 실행 옵션 전달
exec .venv-ai/bin/python -u scripts/runtime/launch.py "$@"
