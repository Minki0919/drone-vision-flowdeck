#!/usr/bin/env python3
# 파일 역할: 실행 옵션을 받아 전체 미션 실행 명령으로 연결
"""현재 전방 카메라 전체 미션과 화면을 실행합니다."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import argparse
import os
from pathlib import Path

# 파트: 공통 경로·좌표계·설정값
ROOT = Path(__file__).resolve().parents[2]


# 처리: 입력 옵션을 검사하고 이 파일의 실행 흐름 시작
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--front-mission-flight', action='store_true', help='전체 미션 실행')
    parser.add_argument('--domain', type=int, default=207)
    parser.add_argument('--seed', type=int, default=100002)
    parser.add_argument('--depth-model', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--stop-after-stage1', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.domain <= 232:
        parser.error('ROS domain must be 0..232')
    command = ['bash', str(ROOT/'scripts/competition.sh'), 'mission-flight',
               '--domain', str(args.domain), '--seed', str(args.seed), '--device', args.device]
    if args.depth_model is not None:
        command += ['--depth-model', str(args.depth_model)]
    for option in ('headless', 'stop_after_stage1'):
        if getattr(args, option):
            command.append('--'+option.replace('_', '-'))
    os.execvp('bash', command)


if __name__ == '__main__':
    main()
