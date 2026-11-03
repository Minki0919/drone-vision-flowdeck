#!/usr/bin/env python3
# 파일 역할: 대회 준비 설정과 금지 기능 검증
"""Competition preparation status; never connects to or commands an aircraft."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import json, math
from pathlib import Path
# 파트: 공통 경로·좌표계·설정값
ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/'competition/profile.json'
REQUIRED_BANS=('ground_pc_flight_control','provided_map_control','simulator_truth_control')

# 처리: 설정 읽기와 센서·비행 방식 제한 확인
def load_profile():
    cfg=json.loads(PROFILE.read_text())
    if cfg.get('schema')!='competition_front_vision_flow_v2':raise ValueError('Unknown competition profile')
    if any(cfg.get('prohibited',{}).get(k) is not True for k in REQUIRED_BANS):raise ValueError('Competition sensor/control prohibition cannot be disabled')
    assistance=cfg.get('sensor_assistance',{})
    if (type(assistance.get('flow_deck')) is not bool
            or assistance.get('mode')!='bounded_flow_assistance_and_ToF_mission_verification'
            or assistance.get('imu_usage')!='flow_rotation_and_ToF_tilt_compensation'):
        raise ValueError('Flow deck assistance must use the bounded full-mission mode')
    if cfg['target']['camera_direction']!='front' or cfg['target']['image_format']!='grayscale':raise ValueError('Front grayscale camera required')
    if cfg['competition']['mission_sign']!='marker' or cfg['competition']['arrow_instructions'] is not False:raise ValueError('Competition profile uses markers; arrow instructions are not confirmed')
    if cfg.get('landing_policy') != dict(reference='existing_flag_and_downward_ToF', allow_additional_markers=False,
            descent_trigger='relative_speed_matched', completion='ToF_deck_entry_near_surface_and_simulation_contact',
            on_unobservable='hold_except_brief_final_descent'):
        raise ValueError('Truck completion requires ToF deck-entry/near-surface evidence and simulated deck contact')
    preview=cfg['development_preview']
    size=preview.get('route_marker_size_m',.6)
    if isinstance(size,bool) or not isinstance(size,(int,float)) or not math.isfinite(size) or not .05<=size<=1.5:
        raise ValueError('development_preview.route_marker_size_m must be 0.05..1.5 metres (black ArUco square side)')
    if preview['static_camera_only'] is not True or preview['flight_commands'] is not False:raise ValueError('Preview may not enable flight')
    if (preview['camera_width'],preview['camera_height'])!=(320,320) or not 1<=preview['camera_fps']<=30:raise ValueError('Invalid preview camera configuration')
    return cfg
