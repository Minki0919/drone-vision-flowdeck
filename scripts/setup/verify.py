#!/usr/bin/env python3
# 파일 역할: 설치 파일·의존성·설정 검사; 비행은 실행하지 않음
"""Verify the current source bundle and imports without starting a simulation."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import hashlib
import importlib
import json
from pathlib import Path
import shutil

# 파트: 공통 경로·좌표계·설정값
ROOT = Path(__file__).resolve().parents[2]


# 처리: 입력 옵션을 검사하고 이 파일의 실행 흐름 시작
def main():
    manifest = json.loads((ROOT / 'transfer_manifest.json').read_text())
    if manifest['schema'] != 'front_vision_current_transfer_v1':
        raise RuntimeError('This check requires the current front-vision bundle manifest')
    failures = []
    for name, expected in manifest['files_sha256'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append(name)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    if failures:
        raise RuntimeError('Bundle differs from exported source: ' + ', '.join(failures))
    for name in ('rclpy', 'ros_gz_interfaces', 'sensor_msgs.msg', 'geometry_msgs.msg',
                 'rosgraph_msgs.msg', 'torch', 'torchvision', 'ultralytics',
                 'dynamic_drone.control.mission',
                 'dynamic_drone.control.flow_deck',
                 'dynamic_drone.control.marker_odometry'):
        importlib.import_module(name)
    import cv2
    import numpy as np
    import torch
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('OpenCV ArUco module missing')
    if not (hasattr(cv2.aruco, 'ArucoDetector') or hasattr(cv2.aruco, 'detectMarkers')):
        raise RuntimeError('OpenCV ArUco detector missing')
    if not shutil.which('gz'):
        raise RuntimeError('gz command missing; source ROS Jazzy and install Gazebo')
    profile = json.loads((ROOT / 'competition/profile.json').read_text())
    for key in ('ground_pc_flight_control', 'provided_map_control', 'simulator_truth_control'):
        if profile['prohibited'].get(key) is not True:
            raise RuntimeError('Required sensor/control prohibition missing: ' + key)
    print(f'File checksums and imports OK: torch={torch.__version__}, OpenCV={cv2.__version__}, numpy={np.__version__}')
    print('No Gazebo, camera windows, training or flight started. Flight success is not verified.')


if __name__ == '__main__':
    main()
