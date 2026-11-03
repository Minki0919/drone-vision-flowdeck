# 파일 역할: 사용자가 허용한 Flow 보조 센서와 기존 경로의 센서 구성 검사
"""Only the explicitly enabled full-mission Flow assist may use deck sensors."""

FLOW_SENSORS = {
    'flow_deck_camera': ('camera', '/vision_trial/flow/image'),
    'flow_deck_range': ('gpu_lidar', '/vision_trial/flow/range'),
    'flow_deck_imu': ('imu', '/vision_trial/flow/imu'),
}
FLOW_BRIDGES = {
    '/vision_trial/flow/image': ('sensor_msgs/msg/Image', 'gz.msgs.Image'),
    '/vision_trial/flow/camera_info': ('sensor_msgs/msg/CameraInfo', 'gz.msgs.CameraInfo'),
    '/vision_trial/flow/range': ('sensor_msgs/msg/LaserScan', 'gz.msgs.LaserScan'),
    '/vision_trial/flow/imu': ('sensor_msgs/msg/Imu', 'gz.msgs.IMU'),
}

# 처리: 센서 이름·설정에서 금지 항목 식별
def _forbidden(value):
    value=str(value).lower().replace('-', '_')
    return any(token in value for token in (
        'flow_deck','flowdeck','flow_proxy','optical_flow','downward_tof',
        'down_tof','/tof','/flow/', 'pmw3901','vl53l1','vl53l0'))


# 처리: 센서 구성의 금지 항목 검사
def validate_config(cfg):
    research=cfg.get('research',{})
    if research.get('optical_flow') or research.get('downward_tof') or cfg.get('flow_deck'):
        raise RuntimeError('Legacy Flow/ToF controllers remain disabled; use full-mission Flow assistance')


# 처리: 실행 월드의 센서·모델 항목 검사
# 실행 환경에도 금지 항목을 검사; 설정 파일만 확인하고 통과시키지 않음
def validate_sdf(root, *, allow_flow_deck=False):
    for node in root.iter():
        if node.tag in ('sensor','plugin','model','link','include'):
            values=list(node.attrib.values())+[node.findtext('topic',''),node.findtext('uri','')]
            if any(_forbidden(value) for value in values):
                expected = FLOW_SENSORS.get(node.get('name'))
                if (allow_flow_deck and node.tag == 'sensor' and expected is not None
                        and (node.get('type'), node.findtext('topic')) == expected):
                    continue
                raise RuntimeError('Forbidden Flow deck / downward ToF in generated SDF: '+str(values))


# 처리: ROS↔Gazebo 연결에 금지 센서 토픽이 포함됐는지 검사
def validate_bridges(mappings, *, allow_flow_deck=False):
    for row in mappings:
        if any(_forbidden(value) for value in row):
            if (allow_flow_deck and len(row) == 5 and row[1] in FLOW_BRIDGES
                    and tuple(row[2:4]) == FLOW_BRIDGES[row[1]] and row[4] == 'GZ_TO_ROS'
                    and row[0].endswith('/flow/'+row[1].rsplit('/', 1)[-1])):
                continue
            raise RuntimeError('Forbidden Flow deck / downward ToF ROS bridge: '+str(row))
