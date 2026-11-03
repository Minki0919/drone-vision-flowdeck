# 파일 역할: 설정으로 Gazebo SDF 환경 생성 및 센서 검사
"""Latest-course residual PPO contracts, independent of ROS/Gym imports.

The perception/state-machine guide sees images and own odometry, not truck
truth. PPO learns bounded corrections, not the perception or safety rules.
"""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from dynamic_drone.scene.variation import apply_scene_variant
from dynamic_drone.scene.camera_config import configure_front_tof_world

# 파트: 공통 경로·좌표계·설정값
SCHEMA = 'gazebo_visual_course_residual_ppo_v1'
STAGES = ('navigation', 'gates', 'landing', 'full')
PHASES = ('INIT', 'TAKEOFF', 'ARUCO_ACQUIRE', 'ARUCO_MOVE',
          'DIRECTION_SHIFT', 'DIRECTION_HOVER', 'FORWARD_TO_WALL',
          'OBSTACLE_BYPASS', 'RING_ACQUIRE', 'RING_APPROACH', 'RING_CROSS',
          'S3_FLAG_FOLLOW', 'S3_PAD_APPROACH', 'S3_PAD_ALIGN', 'S3_PAD_DESCEND',
          'S3_TOUCHDOWN', 'S3_RIDE_VERIFY', 'ABORT_LAND', 'FAILED', 'COMPLETE')
STATE_SIZE = 15 + len(PHASES) + 4 + 4
EVENT_REWARDS = dict(ARUCO_REACHED=5, DIRECTION_HOVER=3,
                     OBSTACLE_CLEARED=8, RING_PASSED=15,
                     TRUCK_SPEED_MATCHED=10, S3_TOUCHDOWN=15,
                     S3_RIDE_VERIFY=15, COMPLETE=80)










# 처리: 설정을 SDF에 반영하고 금지 센서 검사 후 저장
def write_course_world(root, output, cfg):
    from dynamic_drone.safety.sensor_policy import validate_config,validate_sdf
    validate_config(cfg)
    tree=ET.parse(Path(root)/'ros2_ws/src/dynamic_drone/worlds/gazebo_missions_vision.sdf')
    world=tree.getroot().find('world')
    world.set('name','rl_training')
    speed=float(cfg.get('simulation_target_real_time_factor',1.))
    if not np.isfinite(speed) or not 1<=speed<=8:raise ValueError('Simulation speed must be 1..8')
    physics=world.find('physics')
    if physics is None:raise ValueError('Training world needs explicit fixed-step physics')
    factor=physics.find('real_time_factor')
    if factor is None:factor=ET.SubElement(physics,'real_time_factor')
    factor.text=str(speed)
    # Change wall-clock pacing only; retain the 1ms integration step and all
    # camera/IMU/ToF capture-time synchronization.
    baseline=yaml.safe_load((Path(root)/'ros2_ws/src/dynamic_drone/config/gazebo_mission.yaml').read_text())
    apply_scene_variant(world,cfg,baseline)
    if cfg.get('sensor_suite')=='front_tof':
        configure_front_tof_world(world,cfg)
    drone=world.find("model[@name='crazyflie_2_1']")
    drone.find(".//sensor[@name='front_camera']/camera/image/format").text='L8'
    drone.find('pose').text=' '.join(map(str,(*cfg['start_position'],0,0,0)))
    # Only the copied training world uses explicit image capture. Periodic
    # rendering can skip a camera when physics is rapidly stepped and paused.
    for sensor in drone.findall(".//sensor[@type='camera']"):
        camera=sensor.find('camera')
        triggered=camera.find('triggered')
        if triggered is None:
            triggered=ET.SubElement(camera,'triggered')
        triggered.text='true'
        topic=camera.find('trigger_topic')
        if topic is None:
            topic=ET.SubElement(camera,'trigger_topic')
        topic.text=sensor.findtext('topic')+'/trigger'
    if 'research' in cfg:
        from dynamic_drone.scene.environment import apply_research
        apply_research(world,cfg)
    if 'mission_plan' in cfg:
        from dynamic_drone.scene.markers import apply
        apply(world,cfg,root)
    # Same walls, two navigation codes, four marker-free rings and truck.
    validate_sdf(tree.getroot())
    ET.indent(tree)
    tree.write(output,encoding='utf-8',xml_declaration=True)








