#!/usr/bin/env python3
# 파일 역할: Gazebo·ROS·카메라·제어기를 연결하는 전체 실행 흐름
"""Simulation-only fixed-marker flight reference; NOT competition onboard code."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

from mission_profile import ROOT, load_profile
from preview_world import stop


# 처리: SDF 항목을 생성하거나 기존 값을 수정
def set_text(parent, tag, value):
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    child.text = str(value)


# 처리: 렌더링용 월드 생성; 좌표는 제어 입력으로 사용하지 않음
def make_world(directory, marker_id, marker_size, *, unknown_seed=None):
    """Only scene builder knows fixture coordinates. They never enter the pilot."""
    import yaml
    from dynamic_drone.scene.markers import board
    from dynamic_drone.safety.sensor_policy import validate_sdf
    from gazebo_assets import relocate_meshes
    cfg = yaml.safe_load((ROOT/'ros2_ws/src/dynamic_drone/config/gazebo_mission.yaml').read_text())
    tree = ET.parse(ROOT/'ros2_ws/src/dynamic_drone/worlds/gazebo_missions_vision.sdf')
    world = tree.find('world')
    world.set('name', 'vision_marker_trial')
    for model in list(world.findall('model')):
        if model.get('name') not in ('ground_plane', 'crazyflie_2_1'):
            world.remove(model)
    # A cleared localization trial, not the obstacle/ring/truck course.
    marker_position = [3.4, 0., .8]
    if unknown_seed is not None:
        import random
        rng = random.Random(unknown_seed)
        marker_size = rng.uniform(.55, .95)
        marker_position = [rng.uniform(3.5, 4.0), rng.uniform(-.35, .35), rng.uniform(.65, .85)]
        # Simulator fixture metadata, never read by the observer or pilot.
        (directory/'scene_truth_diagnostic_only.json').write_text(json.dumps(dict(
            seed=unknown_seed, marker_size_m=marker_size, marker_position_m=marker_position,
            controller_access=False), indent=2))
    board(world, marker_id, marker_position, 0., marker_size)
    scene = world.find('scene')
    if scene is None:
        scene = ET.SubElement(world, 'scene')
    set_text(scene, 'ambient', '.65 .65 .65 1')
    set_text(scene, 'background', '.7 .8 .9 1')
    drone = world.find("model[@name='crazyflie_2_1']")
    set_text(drone, 'pose', '0 0 .022 0 0 0')
    set_text(drone, 'static', 'false')
    for plugin in list(drone.findall('plugin')):
        if plugin.get('filename') not in ('gz-sim-multicopter-motor-model-system',
                                         'gz-sim-multicopter-control-system'):
            drone.remove(plugin)
    for link in drone.findall('link'):
        for sensor in list(link.findall('sensor')):
            if sensor.get('name') not in ('front_camera', 'mission_contact'):
                link.remove(sensor)
    front = drone.find(".//sensor[@name='front_camera']")
    camera = front.find('camera')
    for tag in ('triggered', 'trigger_topic'):
        item = camera.find(tag)
        if item is not None:
            camera.remove(item)
    set_text(front, 'topic', '/vision_trial/image')
    set_text(front, 'update_rate', 20)
    set_text(camera, 'camera_info_topic', '/vision_trial/camera_info')
    set_text(camera.find('image'), 'width', 320)
    set_text(camera.find('image'), 'height', 320)
    set_text(camera.find('image'), 'format', 'L8')
    set_text(drone.find(".//sensor[@name='mission_contact']"), 'topic', '/vision_trial/contact')
    # Match the fixed extrinsic calibration used by the reference localizer.
    body_pose = [float(v) for v in drone.find("link[@name='base_link']/pose").text.split()]
    offset = cfg['front_camera_offset']
    set_text(front, 'pose', ' '.join(map(str, [offset[i]-body_pose[i] for i in range(3)]+[0, 0, 0])))
    validate_sdf(tree.getroot())
    path = directory/'world.sdf'
    ET.indent(tree)
    tree.write(path, encoding='utf-8', xml_declaration=True)
    relocate_meshes(path, ROOT)
    (directory/'trial_contract.json').write_text(json.dumps(dict(
        mode='simulation_reference_only', camera='front_gray_320',
        stationary_marker_id=marker_id, marker_size_m=marker_size if unknown_seed is None else None,
        marker_size_status='development_assumption_not_official' if unknown_seed is None else 'unknown_to_controller',
        position_source='front_camera_PnP_relative_to_takeoff' if unknown_seed is None else 'none_image_servo_only',
        simulator_odometry_input=False, provided_map_input=False,
        low_level='Gazebo MulticopterVelocityControl uses simulated rigid-body state',
        hardware_firmware=False, full_mission=False, collision_contact='termination_only'), indent=2))
    return path, offset


# 처리: 허용된 ROS↔Gazebo 토픽만 연결
def bridge_config(directory, *, ros_namespace, image_only=False, full_mission=False, flow_deck=False):
    import yaml
    from dynamic_drone.safety.sensor_policy import validate_bridges, FLOW_BRIDGES
    rows = [
        (ros_namespace+'/image', '/vision_trial/image', 'sensor_msgs/msg/Image', 'gz.msgs.Image', 'GZ_TO_ROS'),
        (ros_namespace+'/camera_info', '/vision_trial/camera_info', 'sensor_msgs/msg/CameraInfo', 'gz.msgs.CameraInfo', 'GZ_TO_ROS'),
        (ros_namespace+'/clock', '/clock', 'rosgraph_msgs/msg/Clock', 'gz.msgs.Clock', 'GZ_TO_ROS'),
        (ros_namespace+'/contact', '/vision_trial/contact', 'ros_gz_interfaces/msg/Contacts', 'gz.msgs.Contacts', 'GZ_TO_ROS'),
        (ros_namespace+'/cmd', '/crazyflie_2_1/gazebo/command/twist', 'geometry_msgs/msg/Twist', 'gz.msgs.Twist', 'ROS_TO_GZ'),
        (ros_namespace+'/enable', '/crazyflie_2_1/enable', 'std_msgs/msg/Bool', 'gz.msgs.Boolean', 'ROS_TO_GZ'),
    ]
    if image_only:
        rows = [row for row in rows if row[2] != 'sensor_msgs/msg/CameraInfo']
    if full_mission:
        rows.append((ros_namespace+'/truck_cmd', '/model/mission_vehicle/cmd_vel', 'geometry_msgs/msg/Twist', 'gz.msgs.Twist', 'ROS_TO_GZ'))
    if flow_deck:
        if not full_mission:
            raise ValueError('Flow assistance is available only in the full mission')
        rows += [(ros_namespace+'/flow/'+topic.rsplit('/', 1)[-1], topic, *types, 'GZ_TO_ROS')
                 for topic, types in FLOW_BRIDGES.items()]
    validate_bridges(rows, allow_flow_deck=flow_deck)
    keys = ('ros_topic_name', 'gz_topic_name', 'ros_type_name', 'gz_type_name', 'direction')
    path = directory/'bridge.yaml'
    path.write_text(yaml.safe_dump([dict(zip(keys, row)) for row in rows]))
    return path


# 처리: 입력 옵션을 검사하고 이 파일의 실행 흐름 시작
def main():
    # 파트 1: 실행 옵션·깊이 모델 경로·센서 제한 검사
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', type=int, default=210)
    parser.add_argument('--marker-id', type=int, default=7)
    parser.add_argument('--marker-size-m', type=float)
    parser.add_argument('--rise-m', type=float)
    parser.add_argument('--stand-off-m', type=float)
    parser.add_argument('--unknown-size', action='store_true', help='Image centering/area only; no metric pose')
    parser.add_argument('--seed', type=int, default=100002, help='Unknown-size scene generator only')
    parser.add_argument('--target-area', type=float, default=.06, help='Unknown-size target marker area / image area')
    parser.add_argument('--use-depth', action='store_true', help='Use learned marker depth for stopping in unknown-size mode')
    parser.add_argument('--depth-model', type=Path)
    parser.add_argument('--stop-distance', type=float, default=2., help='Predicted marker depth stopping target, not measured distance')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--full-mission', action='store_true', help='Experimental ArUco obstacle/ring/truck mission; no legacy sensors')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--stop-after-stage1', action='store_true',
                        help='Simulation check: stop owned session when markers 1..6 complete')
    args = parser.parse_args()
    if args.stop_after_stage1 and not args.full_mission:
        parser.error('--stop-after-stage1 requires --full-mission')
    if args.full_mission:
        args.unknown_size = True
        args.use_depth = True
    if args.use_depth and not args.unknown_size:
        parser.error('--use-depth requires --unknown-size')
    if args.depth_model is not None and not args.use_depth:
        parser.error('--depth-model requires --use-depth')
    if args.use_depth:
        args.depth_model = (args.depth_model or ROOT/'DATA/sensor_only_20261011_050229/depth/depth.pt').resolve()
        if not args.depth_model.is_file():
            parser.error('Depth model missing: '+str(args.depth_model))
    if not 0 <= args.domain <= 232 or not 0 <= args.marker_id < 50:
        parser.error('Invalid ROS domain or DICT_4X4_50 marker ID')
    if args.unknown_size:
        if any(value is not None for value in (args.marker_size_m, args.rise_m, args.stand_off_m)):
            parser.error('Unknown size cannot use marker-size, metric rise or metric stand-off arguments')
    else:
        args.marker_size_m = .8 if args.marker_size_m is None else args.marker_size_m
        args.rise_m = .65 if args.rise_m is None else args.rise_m
        args.stand_off_m = 2. if args.stand_off_m is None else args.stand_off_m
        if not .6 <= args.marker_size_m <= 1.0:
            parser.error('Cleared trial marker size must be .6..1m')
    # 파트 2: 대회 설정과 금지 기능 확인
    profile = load_profile()  # Preserve hard sensor and real-aircraft control prohibitions.
    import cv2
    import numpy as np
    from dynamic_drone.perception.marker_pose import FrontMarkerLocalizer, MarkerFlightTrial
    from dynamic_drone.control.marker_servo import UnknownMarkerObserver, UnknownMarkerFlight
    from dynamic_drone.perception.images import decode_image, gray_image
    from gazebo_assets import renderer_environment
    from marker_evidence import MarkerEvidence
    try:
        pilot = UnknownMarkerFlight(args.target_area) if args.unknown_size else MarkerFlightTrial(args.rise_m, args.stand_off_m)
        if args.use_depth:
            from dynamic_drone.control.depth_servo import DepthMarkerFlight
            pilot = DepthMarkerFlight(args.stop_distance)
        if args.full_mission:
            from dynamic_drone.control.mission import FrontVisionMission
            pilot = FrontVisionMission(marker_size=profile['development_preview']['route_marker_size_m'])
    except ValueError as error:
        parser.error(str(error))
    flow_assist = None
    if args.full_mission and profile['sensor_assistance']['flow_deck']:
        from dynamic_drone.control.flow_deck import FlowDeckAssist
        flow_assist = FlowDeckAssist()
        pilot.tof_sensor = flow_assist
    if args.full_mission and flow_assist is None:
        parser.error('ToF passage/landing verification requires sensor_assistance.flow_deck=true')
    key = uuid.uuid4().hex[:8]
    ros_namespace = '/vision_trial_'+key
    predictor = None
    if args.use_depth:
        import torch
        from dynamic_drone.perception.depth import Predictor
        torch.set_num_threads(2)
        predictor = Predictor(args.depth_model, device=args.device)
    prefix = 'depth_marker_flight_' if args.use_depth else 'unknown_marker_flight_' if args.unknown_size else 'vision_flight_'
    if args.full_mission:
        prefix = 'front_full_mission_'
    # 파트 3: 이번 실행의 결과 폴더와 환경 생성
    directory = ROOT/'DATA'/(prefix+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+key)
    directory.mkdir(parents=True)
    if args.full_mission:
        from mission_world import make_mission_world
        world, offset = make_mission_world(directory, profile, args.seed)
    else:
        world, offset = make_world(directory, args.marker_id, args.marker_size_m,
                                   unknown_seed=args.seed if args.unknown_size else None)
    bridge = bridge_config(directory, ros_namespace=ros_namespace,
        image_only=args.unknown_size and not args.full_mission,
        full_mission=args.full_mission, flow_deck=flow_assist is not None)
    if predictor is not None:
        (directory/'depth_model.json').write_text(json.dumps(dict(
            path=str(predictor.path), sha256=predictor.hash, device=predictor.device,
            input='current_front_grayscale_only', output='predicted_optical_depth_m',
            stop_distance_m=args.stop_distance, new_scene_accuracy_verified=False,
            model_forward_ms='CUDA synchronized; preprocessing/copy excluded',
            vision_pipeline_ms='marker detection plus prediction, ROI checks and transfer'), indent=2))
    environment = renderer_environment(dict(os.environ), os.environ.get('DYNAMIC_DRONE_RENDER_GPU', 'auto'))
    environment.update(GZ_PARTITION='vision_flight_'+key, ROS_DOMAIN_ID=str(args.domain),
                       ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')
    for name in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        if 'cv2' in environment.get(name, ''):
            environment.pop(name)
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo, Imu, LaserScan
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool
    from rosgraph_msgs.msg import Clock
    from ros_gz_interfaces.msg import Contacts
    processes, streams = [], []
    node = None
    command_pub = None
    enable_pub = None
    truck_pub = None
    truck_motion_started = None
    state = dict(image=None, info=None, clock=None, clock_wall=None, image_wall=None,
                 collision=None, departed=False, flow_image=None, tof_wall=None,
                 flow_image_wall=None, flow_imu_wall=None)
    started = time.monotonic()
    status = 'starting'
    localizer = UnknownMarkerObserver(args.marker_id) if args.unknown_size else None
    if args.full_mission:
        from dynamic_drone.perception.yolo import MissionYOLO
        from dynamic_drone.control.mission import MissionObserver
        localizer = MissionObserver(MissionYOLO(ROOT, device=predictor.device))
    last_stamp = None
    last_flow_stamp = None
    last_control = None
    last_valid_wall = None
    last_depth_valid_wall = None
    last_valid_frame = None
    last_depth_valid_frame = None
    armed = False
    armed_at = None
    output = None
    inference_ms = None
    depth_forward_ms = None
    depth_view = None
    display_depth_stamp = None
    display_depth_forward_ms = None
    display_depth_only = True
    camera_canvas = None
    depth_canvas = None
    flow_canvas = None
    flow_gray = None
    last_command = [0., 0., 0., 0.]
    last_print = 0.
    last_diagnostic = 0.
    window = 'VISION FLIGHT | SIMULATION ONLY | '+('Full mission marker localization' if args.full_mission
              else 'UNKNOWN SIZE image servo' if args.unknown_size else 'Front marker localization')
    depth_window = 'DEPTH ESTIMATE | Front grayscale model'
    flow_window = 'FLOW DECK | Downward camera + ToF + IMU'

    # 처리: 추정 깊이를 화면 색상으로 변환; 거리 측정이나 제어 입력을 만들지 않음
    def depth_color(prediction):
        invalid = ~np.isfinite(prediction)
        color = np.uint8(255*(1.-np.clip(np.nan_to_num(prediction, nan=8., posinf=8., neginf=0.), 0., 8.)/8.))
        picture = cv2.applyColorMap(color, cv2.COLORMAP_TURBO)
        picture[invalid] = (96, 96, 96)
        return cv2.resize(picture, (320, 320))
    evidence = MarkerEvidence(directory/'marker_evidence', 'DICT_4X4_50',
                              source='Gazebo_front_marker_flight_reference', action='visual_flight_trial')

    # 처리: 카메라 메시지와 실제 수신 시간을 기록
    def image_received(message):
        state['image'] = message
        state['image_wall'] = time.monotonic()

    # 처리: Gazebo 시뮬레이션 시간과 수신 시간을 기록
    def clock_received(message):
        value = message.clock.sec+message.clock.nanosec*1e-9
        if state['clock'] is None or value != state['clock']:
            state['clock_wall'] = time.monotonic()
        state['clock'] = value

    def flow_info_received(message):
        if (message.width, message.height) == (128, 128) and message.distortion_model in ('', 'plumb_bob'):
            try:
                flow_assist.configure(message.k)
                return
            except ValueError:
                pass
        flow_assist.matrix = None
        flow_assist.invalidate_flow('flow_camera_calibration_unavailable')

    def flow_image_received(message):
        state['flow_image'] = message
        state['flow_image_wall'] = time.monotonic()

    def flow_range_received(message):
        stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        distance = message.ranges[len(message.ranges)//2] if message.ranges else float('nan')
        flow_assist.update_range(stamp, distance)
        state['tof_wall'] = time.monotonic()

    def tof_readout(now, wall_now):
        if flow_assist is None:
            return dict(enabled=False, state='DISABLED'), 'DOWN ToF: DISABLED'
        fresh = (state['tof_wall'] is not None and wall_now-state['tof_wall'] < .25
                 and state['clock_wall'] is not None and wall_now-state['clock_wall'] < .5)
        data = flow_assist.range_telemetry(now, transport_fresh=fresh)
        distance, age = data['distance_m'], data['age_s']
        value = '--' if distance is None else f'{distance:.3f} m ({distance*1000:.0f} mm)'
        age_text = '--' if age is None else f'{age:.3f} s'
        return data, f"DOWN ToF: {value} | {data['state']} | age: {age_text}"

    def flow_imu_received(message):
        state['flow_imu_wall'] = time.monotonic()
        stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        q = message.orientation
        flow_assist.update_attitude(stamp, [q.x, q.y, q.z, q.w]
            if message.orientation_covariance[0] != -1. else [0., 0., 0., 0.])

    # 처리: 접촉 정보는 충돌 종료·차량 착지 확인에만 사용; 위치 제어 입력 아님
    def contacts_received(message):
        if args.full_mission and (pilot.stage1_active() or pilot.ring_stage_active()):
            return
        for contact in message.contacts:
            collision_names = [contact.collision1.name, contact.collision2.name]
            names = '|'.join(collision_names)
            contact_now = (state['clock'] if state['clock_wall'] is not None
                           and time.monotonic()-state['clock_wall'] < .5 else None)
            if args.full_mission and pilot.confirm_truck_contact(collision_names,
                    [(n.x, n.y, n.z) for n in contact.normals], contact_now):
                continue
            if not state['departed'] and 'ground_plane' in names:
                continue  # Initial ground support only; never used as a height input.
            state['collision'] = names

    try:
        # 파트 4: ROS 구독·명령 발행과 Gazebo 프로세스 연결
        rclpy.init()
        node = rclpy.create_node('vision_flight_'+key)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(Image, ros_namespace+'/image', image_received, qos)
        if not args.unknown_size or args.full_mission:
            node.create_subscription(CameraInfo, ros_namespace+'/camera_info', lambda m: state.update(info=m), qos)
        node.create_subscription(Clock, ros_namespace+'/clock', clock_received, qos)
        node.create_subscription(Contacts, ros_namespace+'/contact', contacts_received, qos)
        command_pub = node.create_publisher(Twist, ros_namespace+'/cmd', 1)
        enable_pub = node.create_publisher(Bool, ros_namespace+'/enable', 1)
        if args.full_mission:
            truck_pub = node.create_publisher(Twist, ros_namespace+'/truck_cmd', 1)
        if flow_assist is not None:
            node.create_subscription(Image, ros_namespace+'/flow/image', flow_image_received, qos)
            node.create_subscription(CameraInfo, ros_namespace+'/flow/camera_info', flow_info_received, qos)
            node.create_subscription(LaserScan, ros_namespace+'/flow/range', flow_range_received, qos)
            node.create_subscription(Imu, ros_namespace+'/flow/imu', flow_imu_received, qos)
        commands = [('gazebo', ['gz', 'sim', '-r', *(['-s'] if args.headless else []), str(world)]),
                    ('bridge', ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge', '--ros-args', '-p', 'config_file:='+str(bridge)])]
        for name, command in commands:
            stream = (directory/(name+'.log')).open('w')
            streams.append(stream)
            processes.append(subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                              stderr=subprocess.STDOUT, start_new_session=True))
        output = (directory/'vision_metrics.jsonl').open('a', buffering=1)
        if args.full_mission:
            print('전체 환경: 1 순차 마커 방문 → 2 링 → 3 왕복 트럭 추종. 깃발 거리 1.5m 유지, 상대 속도 0.05m/s 이하이면 즉시 하강하며 계속 추종.', flush=True)
            print(f'1단계: OpenCV PnP, 마커 한 변 {pilot.route_marker_size:.3f}m. 전방 마커 관측으로 위치를 계산하며 깊이 화면은 표시 전용.', flush=True)
            if flow_assist is not None:
                print('Flow/ToF: 정지 보정·높이 유지·착지 감속. 링은 하단 거리 감소→지면 회복으로 통과 확인, 차량은 ToF 적재함 진입·근접과 접촉으로 착륙 확인.', flush=True)
        elif predictor is not None:
            print(f'깊이 모드 활성: 화면 면적 대신 추정 거리 {args.stop_distance:.2f}m로 감속·정지. 고도 추정은 없음.', flush=True)
        else:
            print(('크기 미상 시험: 영상 중심으로 상승·정렬 → 화면 점유율 기준 접근 → 3초 정지. 거리·고도 추정 없음.'
                   if args.unknown_size else '전방 마커 기반 시뮬 시험: 이륙 → 접근 → 3초 정지. 실기체 연결 없음.'), flush=True)
        print('기록: '+str(directory), flush=True)
        print('ROS 실행 토픽: '+ros_namespace, flush=True)
        # 파트 9: 카메라·깊이 추정 결과 화면 표시
        if not args.headless:
            if Path('/usr/share/fonts/truetype/dejavu').is_dir():
                os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts/truetype/dejavu'
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 780, 540)
            cv2.moveWindow(window, 1100, 20)
            if predictor is not None:
                cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(depth_window, 780, 330)
                cv2.moveWindow(depth_window, 1100, 695)
            if flow_assist is not None:
                from flow_sensor_view import render_flow_panel
                cv2.namedWindow(flow_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(flow_window, 800, 660)
                cv2.moveWindow(flow_window, 20, 60)
        gray = np.zeros((320, 320), np.uint8)
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        while rclpy.ok():
            wall_now = time.monotonic()
            ring_stage = args.full_mission and pilot.ring_stage_active()
            if any(p.poll() is not None for p in processes):
                status = 'process_exited'
                break
            rclpy.spin_once(node, timeout_sec=.005)
            now = state['clock']
            if state['collision'] and not ring_stage:
                status = 'collision_terminated'
                break
            flow_image = state['flow_image']
            if flow_assist is not None and flow_image is not None and now is not None:
                flow_stamp = flow_image.header.stamp.sec+flow_image.header.stamp.nanosec*1e-9
                if flow_stamp != last_flow_stamp and 0 <= now-flow_stamp <= .15:
                    last_flow_stamp = flow_stamp
                    try:
                        decoded_flow = gray_image(decode_image(flow_image))
                        flow_gray = decoded_flow if decoded_flow.shape == (128, 128) else None
                        flow_assist.update_image(decoded_flow, flow_stamp)
                    except (ValueError, cv2.error):
                        flow_gray = None
                        flow_assist.invalidate_flow('flow_image_decode_or_tracking_failed')
            info = state['info']
            if args.full_mission and info is not None and pilot.odometry is None:
                if info.width != 320 or info.height != 320 or info.distortion_model not in ('', 'plumb_bob'):
                    raise ValueError('Unsupported mission CameraInfo')
                pilot.configure_camera(info, offset=offset)
                (directory/'camera_calibration.json').write_text(json.dumps(dict(
                    source='Gazebo_CameraInfo_not_real_camera_calibration', k=list(info.k), d=list(info.d),
                    size=[info.width, info.height], camera_offset_body=offset,
                    route_marker_side_m=pilot.route_marker_size,
                    marker_side_definition='black_ArUco_square_excluding_white_quiet_border',
                    hardware_calibration_verified=False), indent=2))
            if localizer is None and info is not None:
                if info.distortion_model not in ('', 'plumb_bob') or info.width != 320 or info.height != 320:
                    raise ValueError('Unsupported CameraInfo calibration')
                localizer = FrontMarkerLocalizer(info.k, info.d, (info.width, info.height),
                                                 args.marker_id, args.marker_size_m, offset)
                (directory/'camera_calibration.json').write_text(json.dumps(dict(
                    source='Gazebo_CameraInfo_not_real_camera_calibration', k=list(info.k), d=list(info.d),
                    size=[info.width, info.height], camera_offset_body=offset), indent=2))
            # 파트 5: 새 카메라 영상만 받아 인식·추론 수행
            message = state['image']
            if localizer is not None and message is not None and now is not None:
                stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
                # Image and /clock use independent queues. Wait for the matching
                # clock instead of discarding every slightly-ahead image forever.
                if stamp != last_stamp and stamp <= now:
                    last_stamp = stamp
                    gray = gray_image(decode_image(message))
                    before = time.perf_counter()
                    if now-stamp > .25 and not ring_stage:
                        localizer.reject('stale_or_future_camera_frame')
                    else:
                        if args.full_mission:
                            localizer.detect_rings = pilot.ring_stage_active()
                            localizer.ring_target_locked = pilot.phase == 'S2_CROSS_RING'
                        localizer.process(gray, stamp)
                    if predictor is not None:
                        if args.full_mission:
                            pilot.update_stage1_primary(localizer.latest)
                        # 마커·차량 단계 깊이는 표시 전용; 제어는 전방 마커 PnP 사용
                        if args.full_mission and (pilot.stage1_active() or pilot.truck_stage_active()):
                            # Display-only inference never updates the PnP pilot.
                            # Throttle it to 5Hz so the extra view stays lightweight.
                            depth_forward_ms = None
                            if not args.headless and (display_depth_stamp is None or stamp-display_depth_stamp >= .2):
                                display_prediction, display_depth_forward_ms = predictor.predict(gray)
                                depth_view = depth_color(display_prediction)
                                display_depth_stamp = stamp
                            display_depth_only = True
                        else:
                            predicted, depth_forward_ms = predictor.predict(gray)
                            display_depth_forward_ms = depth_forward_ms
                            pilot.depth.update(predicted, localizer.latest, gray.shape)
                            depth_view = depth_color(predicted)
                            display_depth_stamp = stamp
                            display_depth_only = False
                            if localizer.latest is not None:
                                cv2.polylines(depth_view, [localizer.latest.corners.astype(np.int32)], True, (255, 255, 255), 1)
                    inference_ms = 1000*(time.perf_counter()-before)
                    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    if localizer.latest is not None:
                        pose = localizer.latest
                        cv2.polylines(overlay, [pose.corners.astype(np.int32)], True, (80, 255, 80), 1)
                        if args.full_mission:
                            boxes = [m.corners for m in pose.markers.values()]
                            ids = np.array(list(pose.markers), dtype=np.int32).reshape(-1, 1)
                            if boxes:
                                cv2.aruco.drawDetectedMarkers(overlay, [b.reshape(1, 4, 2) for b in boxes], ids)
                            if pilot.matrix is not None:
                                camera_center = tuple(np.rint(pilot.matrix[:2, 2]).astype(int))
                                cv2.drawMarker(overlay, camera_center, (255, 0, 255), cv2.MARKER_CROSS, 14, 1)
                                locked = pilot.phase == 'S2_CROSS_RING'
                                for detected_ring in pose.visible_rings:
                                    box = np.rint(detected_ring['box']).astype(int)
                                    color = (0, 190, 255) if locked else (0, 255, 0)
                                    cv2.rectangle(overlay, tuple(box[:2]), tuple(box[2:]), color, 1)
                                    label = ('ring TRACK' if detected_ring.get('source') == 'verified_ring_optical_flow'
                                             else f"ring {detected_ring['confidence']:.2f}")
                                    cv2.putText(overlay, label,
                                        (max(0, box[0]), max(12, box[1]-4)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
                                if locked:
                                    cv2.putText(overlay, 'LOCKED TARGET | YOLO DISPLAY ONLY', (5, 16),
                                        cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 190, 255), 1)
                                target_id = pilot.target_marker_id if pilot.stage1_active() else 23 if pilot.truck_stage_active() else None
                                target = pose.markers.get(target_id)
                                center = (target.corners.mean(axis=0) if target is not None else
                                          pose.ring['box_center'] if not pilot.stage1_active() and pose.ring is not None else None)
                                if center is not None:
                                    center = tuple(np.rint(center).astype(int))
                                    cv2.circle(overlay, center, 3, (0, 255, 0), 1)
                                    cv2.line(overlay, camera_center, center, (0, 255, 0), 1)
                            if pilot.stage1_active() and pilot.frame is pose and pilot.odometry.valid:
                                for identifier, observation in pilot.markers.items():
                                    corner = observation['target'].corners[0]
                                    where = (max(2, int(corner[0])), max(15, int(corner[1])-5))
                                    distance = float(np.linalg.norm(observation['body'][:2]))
                                    method = 'PnP'
                                    if identifier in pose.tracked_marker_ids:
                                        method += ' / tracked'
                                    cv2.putText(overlay, f'{identifier}: {distance:.2f}m {method}', where,
                                                cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 255), 1)
                            evidence.record(gray, overlay, boxes, ids, stamp)
                        else:
                            evidence.record(gray, overlay, [pose.corners], np.array([[args.marker_id]]), stamp)
                        if args.full_mission and time.monotonic()-last_diagnostic >= 3.:
                            # Keep an actual recent image even when no ID decoded;
                            # recognition-only evidence cannot diagnose missed signs.
                            for name, image in (('latest_front_gray', gray), ('latest_front_overlay', overlay)):
                                if not cv2.imwrite(str(directory/(name+'.png')), image):
                                    raise OSError('Could not write front-image diagnostic')
                            (directory/'latest_front_observation.json').write_text(json.dumps(dict(
                                capture_simulation_seconds=stamp, observation=localizer.telemetry(),
                                mission=pilot.telemetry()), ensure_ascii=False, indent=2))
                            memory_status = pilot.telemetry()
                            if pilot.stage1_odometry.last_valid_marker_snapshot is not None:
                                (directory/'last_valid_marker_pose.json').write_text(json.dumps(dict(
                                    last_valid_marker=pilot.stage1_odometry.last_valid_marker_snapshot,
                                    current_pose_valid=pilot.stage1_odometry.valid,
                                    depth_position_fallback_enabled=False), ensure_ascii=False, indent=2))
                            (directory/'remembered_markers.json').write_text(json.dumps(dict(
                                capture_simulation_seconds=stamp,
                                source='observed_known_size_ArUco_OpenCV_PnP',
                                coordinate_frame='initial_body_pose_observed_marker_map',
                                memory_persistence='this_flight_only_no_cross_scene_reload',
                                metric_accuracy_verified=False,
                                markers=memory_status['remembered_markers'],
                                recognized_route_ids=memory_status['route_ID_records'],
                                completed_by_metric_visit=memory_status['aruco_completed_by_metric_visit'],
                                navigation=memory_status['marker_memory_navigation']),
                                ensure_ascii=False, indent=2))
                            last_diagnostic = time.monotonic()
                        if not args.unknown_size and pose.position[2] > .15:
                            state['departed'] = True
            # Stop on wall-time freeze as well as sensor simulation-time age.
            wall_now = time.monotonic()
            tof_data, tof_text = tof_readout(now, wall_now)
            # 파트 6: 영상·시뮬레이션 시간의 실제 갱신 여부 검사
            fresh_transport = (state['image_wall'] is not None and wall_now-state['image_wall'] < .25
                               and state['clock_wall'] is not None and wall_now-state['clock_wall'] < .5)
            if not ring_stage and localizer is not None and localizer.latest is not None:
                if not fresh_transport or now is None or not 0 <= now-localizer.latest.stamp <= .25:
                    localizer.reject('camera_or_clock_stale')
            pose = localizer.latest if localizer is not None and (fresh_transport or ring_stage) else None
            if pose is not None and pose.stamp != last_valid_frame:
                last_valid_frame = pose.stamp
                last_valid_wall = wall_now
            if predictor is not None and pose is None and not ring_stage:
                pilot.depth.reject('image_unavailable_or_stale')
            if predictor is not None and pilot.depth.latest is not None:
                if pilot.depth.latest['stamp'] != last_depth_valid_frame:
                    last_depth_valid_frame = pilot.depth.latest['stamp']
                    last_depth_valid_wall = wall_now
            if now is not None and (last_control is None or now-last_control >= .05):
                last_control = now
                # 파트 7: 현재 관측으로 명령 계산; 아래에서 ROS로 발행
                if args.full_mission:
                    pilot.tof_transport_fresh = bool(tof_data.get('valid'))
                command, reason = pilot.command(pose, now)
                requested_command = list(map(float, command))
                if flow_assist is not None:
                    searching = (pilot.phase == 'S2_FIND_RING' or (pilot.phase == 'S1_MARKERS'
                                 and (pose is None or pilot.target_marker_id not in pose.markers)))
                    command = flow_assist.apply(command, pilot.phase, now, searching=searching,
                        landing_observed=pilot.truck_status.get('observation_valid', False))
                last_command = list(map(float, command))
                if pilot.phase != 'ACQUIRE':
                    armed = not (args.full_mission and pilot.landing_confirmed)
                    if armed_at is None:
                        armed_at = now
                if args.unknown_size and armed_at is not None and now-armed_at > 2.:
                    # Initial ground-support grace, NOT an airborne/height estimate.
                    state['departed'] = True
                enable_pub.publish(Bool(data=armed))
                msg = Twist()
                msg.linear.x, msg.linear.y, msg.linear.z = map(float, command[:3])
                msg.angular.z = float(command[3])
                command_pub.publish(msg)
                if truck_pub is not None:
                    truck = Twist()
                    if pilot.truck_stage_active():
                        if truck_motion_started is None:
                            truck_motion_started = now
                        # Simulation actor only: its commanded motion is never a pilot input.
                        truck.linear.x = .15*math.cos(math.pi*(now-truck_motion_started)/12.)
                    truck_pub.publish(truck)
                record = dict(simulation_seconds=now, wall_seconds=wall_now-started,
                              phase=pilot.phase, reason=reason, command_body=list(map(float, command)),
                              mission_command_body=requested_command,
                              flow_deck=None if flow_assist is None else flow_assist.telemetry(),
                              tof=tof_data,
                              pose_inference_ms=inference_ms if predictor is None else None,
                              vision_pipeline_ms=inference_ms if predictor is not None else None,
                              transport_fresh=fresh_transport,
                              model_forward_ms=depth_forward_ms,
                              display_depth_stamp=display_depth_stamp,
                              display_depth_only=display_depth_only,
                              display_depth_forward_ms=display_depth_forward_ms,
                              localization=localizer.telemetry() if localizer else None,
                              image_servo=pilot.telemetry() if args.unknown_size else None)
                output.write(json.dumps(record)+'\n')
                if wall_now-last_print >= 1:
                    observed = ('center='+str(np.round(pose.center_error, 3))+f' | area={pose.area_fraction:.3f}'
                                if args.unknown_size and pose is not None else
                                'position='+str(np.round(pose.position, 3)) if pose is not None else 'observation unavailable')
                    if args.full_mission:
                        odom = pilot.odometry
                        observed = ('position='+str(np.round(odom.position, 3))
                                    if odom is not None and odom.valid and pose is not None
                                    else 'position unavailable')
                        if odom is not None:
                            observed += ' | PnP: '+odom.reason
                        if pilot.truck_stage_active() or pilot.landing_confirmed:
                            observed = ('flag body='+str(pilot.truck_status.get('flag_body_m'))+
                                ' | relative speed='+str(pilot.truck_status.get('relative_horizontal_speed_m_s')))
                    print(f"{pilot.phase} | {reason} | "+observed, flush=True)
                    if flow_assist is not None:
                        print(tof_text+' | ring: '+pilot.ring_tof.get('state', 'WAIT')+
                              ' | deck: '+pilot.truck_tof.get('state', 'WAIT'), flush=True)
                    if predictor is not None:
                        print('추정 깊이: '+json.dumps(pilot.depth.telemetry(), ensure_ascii=False), flush=True)
                    last_print = wall_now
                if pilot.phase in ('ABORT', 'COMPLETE'):
                    status = pilot.phase.lower()
                    break
                if args.stop_after_stage1 and pilot.phase == 'S2_FIND_RING':
                    status = 'stage1_complete'
                    break
            elif not fresh_transport and not ring_stage:
                command_pub.publish(Twist())
                enable_pub.publish(Bool(data=armed))
            # 1단계는 관측 기반 고도 제어와 마커 추종만 실행한다.
            stage1_marker_only = args.full_mission and pilot.stage1_active()
            runtime_limits_enabled = not stage1_marker_only and not (args.full_mission and pilot.ring_stage_active())
            if runtime_limits_enabled and armed and (last_valid_wall is None or wall_now-last_valid_wall > 2.):
                status = 'vision_lost_terminated'
                break
            if (predictor is not None and armed and runtime_limits_enabled
                    and not (args.full_mission and pilot.truck_stage_active())
                    and (last_depth_valid_wall is None or wall_now-last_depth_valid_wall > 2.)):
                status = ('visual_odometry_lost_terminated'
                          if args.full_mission and pilot.odometry is not None and not pilot.odometry.valid
                          else 'predicted_depth_lost_terminated')
                break
            if runtime_limits_enabled and not armed and wall_now-started > 60:
                status = 'depth_or_marker_acquisition_timeout' if predictor is not None else 'camera_or_marker_acquisition_timeout'
                break
            if runtime_limits_enabled and wall_now-started > (900 if args.full_mission else 180):
                status = 'trial_wall_timeout'
                break
            if not args.headless:
                banner = np.zeros((170, 960, 3), np.uint8)
                telemetry = localizer.telemetry() if localizer else {}
                lines = ['SIM REFERENCE ONLY | NO FLOW / ToF | NO PROVIDED MAP',
                         f'Phase: {pilot.phase} | '+telemetry.get('reason', 'waiting_camera_calibration'),
                         'Relative XYZ (m): '+str(telemetry.get('position_m')),
                         'Marker in body (m): '+str(telemetry.get('marker_body_m')),
                         'Z = rise from initial pose, NOT ground range | Q / Esc: stop']
                if args.unknown_size:
                    lines = ['UNKNOWN MARKER SIZE | IMAGE SERVO | NO METRIC POSITION',
                             f'Phase: {pilot.phase} | '+telemetry.get('reason', ''),
                             'Image center error (right/down): '+str(telemetry.get('center_error')),
                             f"Marker area: {telemetry.get('area_fraction')} | target: {args.target_area:.3f}",
                             'Distance / altitude: UNKNOWN | Q / Esc: stop']
                    cv2.drawMarker(overlay, (160, 160), (0, 200, 255), cv2.MARKER_CROSS, 16, 1)
                if predictor is not None:
                    depth_info = pilot.depth.telemetry()
                    distance = depth_info.get('estimated_marker_depth_m')
                    clearance = depth_info.get('estimated_front_clearance_m')
                    lines = ['LEARNED DEPTH TEST | UNKNOWN MARKER SIZE | ALTITUDE UNKNOWN',
                             f'Phase: {pilot.phase} | '+depth_info['reason'],
                             f'Estimated marker depth: {distance} m | target: {args.stop_distance:.2f} m',
                             f'Front clearance estimate: {clearance} m | red=near blue=far (0..8m)',
                             'Command forward/left/up/yaw: '+str(np.round(last_command, 3))]
                if args.full_mission:
                    mission = pilot.telemetry()
                    odom_info = mission['odometry'] or {}
                    current_position = (odom_info.get('estimated_position_m')
                                        if pose is not None and not pilot.truck_stage_active() else None)
                    banner = np.zeros((380, 960, 3), np.uint8)
                    lines = ['FRONT VISION + '+('FLOW DECK ASSIST' if flow_assist is not None else 'MARKER / RING CONTROL'),
                             f'Phase: {pilot.phase} | '+pilot.depth.reason,
                             'Decoded: '+str(telemetry.get('marker_detection', {}).get('fresh_decoded_ids', []))+
                             ' | Tracked: '+str(telemetry.get('marker_detection', {}).get('tracked_ids', []))+
                             ' | Geometry: '+str(mission['depth_usable_marker_ids']),
                             'Next: '+str(mission['next_route_marker_id'])+' | Active: '+str(mission['active_marker_id'])+' | Visited: '+str(mission['aruco_completed_by_metric_visit']),
                             'Remembered IDs: '+str(mission['remembered_marker_ids'])+' | Memory target: '+str(None if mission['marker_memory_navigation'] is None else mission['marker_memory_navigation']['marker_id']),
                             ('Flag body (m): '+str(mission['truck'].get('flag_body_m')) if pilot.truck_stage_active() else
                              'Current XYZ (m): '+('UNAVAILABLE' if current_position is None else str(np.round(current_position, 3)))),
                             ('Flow: '+str(flow_assist.telemetry().get('reason')) if flow_assist is not None else
                              'PnP rejected: '+str(odom_info.get('PnP_rejections', {}))),
                             ('S1: OpenCV PnP first | marker '+str(mission['stage1_marker_size_m'])+' m | marker-only position'
                              if pilot.phase in ('ACQUIRE', 'S1_TAKEOFF', 'S1_MARKERS') else
                              'Depth rejected IDs: '+str(mission['marker_depth_rejections'])),
                             f'Ring crossings (ToF): {pilot.crossed}/4 | '+mission['ring_tof'].get('state', 'WAIT'),
                             'Motion: '+str(np.round(last_command, 3))+' | '+(mission['stage1_action'] if pilot.phase == 'S1_MARKERS' else pilot.reason),
                             'Landing: '+('CONTACT CONFIRMED' if mission['landing_confirmed'] else
                                'DESCENDING' if pilot.phase == 'S3_DESCEND' else 'WAITING')+
                             ' | Deck ToF: '+mission['truck_tof'].get('state', 'WAIT'),
                             tof_text]
                for index, line in enumerate(lines):
                    cv2.putText(banner, line, (10, 25+index*30), cv2.FONT_HERSHEY_SIMPLEX, .5, (140, 240, 190), 1)
                camera_panel = np.zeros((480, 960, 3), np.uint8)
                camera_panel[:, 240:720] = cv2.resize(overlay if args.full_mission else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (480, 480))
                camera_canvas = np.vstack([banner, camera_panel])
                cv2.imshow(window, camera_canvas)
                if flow_assist is not None:
                    flow_canvas = render_flow_panel(flow_assist, flow_gray, last_flow_stamp,
                        state, now, wall_now, tof_data, pilot)
                    cv2.imshow(flow_window, flow_canvas)
                    if cv2.getWindowProperty(flow_window, cv2.WND_PROP_VISIBLE) < 1:
                        status = 'user_stopped'
                        break
                if predictor is not None:
                    depth_banner = np.zeros((105 if args.full_mission else 80, 640, 3), np.uint8)
                    age = None if display_depth_stamp is None or now is None else now-display_depth_stamp
                    state_text = ('WAITING' if age is None else 'STALE' if age > .5 or not fresh_transport else 'LIVE')
                    mode_text = 'DISPLAY ONLY | FRONT MARKER PnP CONTROL' if display_depth_only else 'PREDICTED DEPTH USED BY CONTROLLER'
                    cv2.putText(depth_banner, state_text+' | '+mode_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .42, (140, 240, 190), 1)
                    cv2.putText(depth_banner, 'red=near / blue=far (0..8m) | gray=invalid', (8, 45), cv2.FONT_HERSHEY_SIMPLEX, .42, (140, 240, 190), 1)
                    cv2.putText(depth_banner, 'Capture: '+str(display_depth_stamp)+' s | Q / Esc: stop', (8, 68), cv2.FONT_HERSHEY_SIMPLEX, .42, (140, 240, 190), 1)
                    if args.full_mission:
                        cv2.putText(depth_banner, tof_text, (8, 92), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 220, 255), 1)
                    depth_panel = np.zeros((320, 640, 3), np.uint8)
                    if depth_view is not None:
                        depth_panel[:, 160:480] = depth_view
                    depth_canvas = np.vstack([depth_banner, depth_panel])
                    cv2.imshow(depth_window, depth_canvas)
                    if cv2.getWindowProperty(depth_window, cv2.WND_PROP_VISIBLE) < 1:
                        status = 'user_stopped'
                        break
                if cv2.waitKey(1)&255 in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    status = 'user_stopped'
                    break
    except KeyboardInterrupt:
        status = 'user_stopped'
    except Exception as error:
        status = 'error: '+str(error)
        raise
    # 파트 10: 정지 명령·프로세스 종료·결과 기록
    finally:
        if command_pub is not None and enable_pub is not None and rclpy.ok():
            command_pub.publish(Twist())
            # Full mission owns a Gazebo-only velocity controller. Keep it
            # enabled until the owned physics server stops, rather than cutting
            # motors while an airborne simulated model is still live. This is
            # cleanup ordering, NOT a real-aircraft sensor-loss failsafe.
            enable_pub.publish(Bool(data=armed if args.full_mission else False))
        if truck_pub is not None and rclpy.ok():
            truck_pub.publish(Twist())
        # Stop owned Gazebo FIRST for full missions, then the bridge. Never
        # leave active physics running with motors disarmed during fault cleanup.
        shutdown_order = processes if args.full_mission else reversed(processes)
        for process in shutdown_order:
            stop(process)
        if args.full_mission and enable_pub is not None and rclpy.ok():
            enable_pub.publish(Bool(data=False))
        for stream in streams:
            stream.close()
        if output is not None:
            output.close()
        (directory/'result.json').write_text(json.dumps(dict(
            status=status, phase=pilot.phase, wall_seconds=time.monotonic()-started,
            collision=state['collision'], localization=localizer.telemetry() if localizer else None,
            stage1_success=(status == 'stage1_complete' and pilot.route_completion_order == list(range(1, 7)))
                           if args.full_mission else None,
            image_servo=pilot.telemetry() if args.unknown_size else None,
            flow_deck=None if flow_assist is None else flow_assist.telemetry(),
            full_mission_success=(status == 'complete' and pilot.landing_confirmed
                                  and pilot.crossed == 4 and pilot.route_completion_order == list(range(1, 7)))
                                 if args.full_mission else None,
            landing_reference=profile['landing_policy']['reference'] if args.full_mission else None,
            landing_attempted=pilot.landing_attempted if args.full_mission else False,
            landing_confirmed=pilot.landing_confirmed if args.full_mission else False,
            onboard_validated=False), ensure_ascii=False, indent=2))
        print('시험 종료: '+status+' | '+str(directory/'result.json'), flush=True)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if args.full_mission and not args.headless and status not in ('complete', 'stage1_complete', 'user_stopped'):
            # All owned simulation processes are stopped. Preserve diagnostic
            # windows only, explicitly labelled as static rather than live video.
            print('시뮬레이션 종료: 마지막 정지 결과 화면을 유지합니다. Q/Esc로 닫으세요.', flush=True)
            try:
                saved_views = [(window, camera_canvas), (depth_window, depth_canvas),
                               (flow_window, flow_canvas)]
                saved_views = [(name, picture.copy()) for name, picture in saved_views if picture is not None]
                for name, picture in saved_views:
                    picture[:80] = 0
                    cv2.putText(picture, 'STOPPED | STATIC LAST FRAME | NOT LIVE', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 170, 255), 1)
                    cv2.putText(picture, status+' | Q / Esc: close', (8, 52), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 170, 255), 1)
                while saved_views and all(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1 for name, _ in saved_views):
                    for name, picture in saved_views:
                        cv2.imshow(name, picture)
                    if cv2.waitKey(100)&255 in (27, ord('q')):
                        break
            except (KeyboardInterrupt, cv2.error):
                pass
        cv2.destroyAllWindows()
    if status not in ('complete', 'stage1_complete', 'user_stopped'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
