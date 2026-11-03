# 파일 역할: 카메라 전용 월드 설정과 호환 계산 함수
"""Front camera + generic ToF simulation contracts; no bottom image input."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import math
import numpy as np

# 파트: 공통 경로·좌표계·설정값
TOF_SCHEMA='gazebo_front_tof_residual_ppo_v1'
TOF_STATE_SIZE=44


# 처리: 호환용 거리 배열에서 유효한 값 추출
def scan_distance(ranges, minimum, maximum):
    if not ranges:
        return None
    value=float(ranges[len(ranges)//2])
    return value if math.isfinite(value) and minimum<=value<maximum else None


# 처리: 호환용 관측·설정으로 목표 계산; 현행 착륙 확인을 대체하지 않음
def front_landing_target(observation, when, velocity, config):
    age=when-observation['stamp']
    if age<-.01:
        raise ValueError('Future flag measurement')
    point=np.asarray(observation['world_position'])+np.asarray(velocity)*max(0.,age)
    return point-np.asarray(config['flag_follow']['marker_offset'])+np.asarray(config['front_tof']['landing_offset'])


# 처리: 호환 월드에서 금지된 하방 센서 제거
def configure_front_tof_world(world, cfg):
    """Legacy caller compatibility: front camera only, no deck sensors."""
    drone=world.find("model[@name='crazyflie_2_1']")
    body=drone.find("link[@name='base_link']")
    for sensor in list(body.findall("sensor")):
        if sensor.get("name") in ("landing_camera", "downward_tof", "flow_proxy_camera"):
            body.remove(sensor)
