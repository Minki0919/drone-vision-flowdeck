# 파일 역할: 마커·링·트럭을 포함하는 비행 환경 조립
"""Full arena builder; generated geometry is never a navigation input."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
from copy import deepcopy
import json
import math
import xml.etree.ElementTree as ET
from preview_world import make_world as preview_world
from flight import make_world as trial_world, set_text
from dynamic_drone.safety.sensor_policy import validate_sdf
from dynamic_drone.scene.markers import board
from dynamic_drone.scene.environment import solid


# 처리: 1단계의 마커와 지지대 배치 구성
def simplify_marker_stage(world, marker_size=.6):
    """Stage 1 has route signs only; leave stage-2 gates/boundaries intact.

    Rendering geometry only; do not supply these positions to the pilot.
    """
    if not math.isfinite(marker_size) or not .05<=marker_size<=1.5:
        raise ValueError('Route marker size must be 0.05..1.5 metres')
    # board() adds a white quiet border one bit-cell wide on all sides.
    # The selected size refers to the black 6x6 encoded square, not its mount.
    panel_size = marker_size*(1.+2./6.)+.04
    post_height = max(.08, 1.2-panel_size/2.)
    for model in list(world.findall('model')):
        if model.get('name', '').startswith(('planned_obstacle_', 'side_obstacle_',
                                              'route_sign_panel_', 'route_sign_wall_', 'route_marker_',
                                              'easy_stage1_', 'between_markers_')):
            world.remove(model)
    pairs = []
    for index in range(6):
        identifier = index+1
        x = 3.7+3.5*index
        side = 1 if index % 2 == 0 else -1
        # Keep the existing sign route, but create no avoidance blocks here.
        marker_position = [x+1.2, -side*1.6, 1.2]
        # The board faces incoming flight. Its collision panel and narrow post
        # stay outside the central lane; no invisible/nonphysical signs.
        solid(world, f'easy_stage1_marker_panel_{identifier}',
              [marker_position[0]+.04, marker_position[1], 1.2],
              [.04, panel_size, panel_size], color=(.75, .75, .75))
        solid(world, f'easy_stage1_marker_post_{identifier}',
              [marker_position[0]+.04, marker_position[1], post_height/2.],
              [.08, .08, post_height], color=(.55, .55, .55))
        board(world, identifier, marker_position, 0., marker_size)
        pairs.append(dict(marker_id=identifier, marker_position_m=marker_position,
                          marker_size_m=marker_size, mounting_panel_size_m=panel_size))
    return dict(name='marker_only_sequential_aruco_v3', stages_modified=[1],
                obstacle_shape=None, obstacle_count=0,
                side_obstacle_count=0, inter_marker_obstacle_count=0,
                central_route_blocked_between_markers=False,
                marker_pairs=pairs, inter_marker_obstacles=[],
                sign_mounts_retained=True, stage2_geometry_changed=False,
                route_marker_size_m=marker_size, size_definition='black_ArUco_square_side_excluding_white_quiet_border',
                positions_use='rendering_and_diagnostics_only_not_controller_input')


# 처리: 기체와 단계별 환경을 합치고 계약 정보 기록
def add_flow_deck(world, aircraft):
    """Rendered downward sensor proxy; never an odometry/pose publisher."""
    body = aircraft.find("link[@name='base_link']")
    for name, kind, topic, rate in (
        ('flow_deck_camera', 'camera', 'image', 20),
        ('flow_deck_range', 'gpu_lidar', 'range', 30),
        ('flow_deck_imu', 'imu', 'imu', 100),
    ):
        sensor = ET.SubElement(body, 'sensor', name=name, type=kind)
        set_text(sensor, 'always_on', 'true')
        set_text(sensor, 'update_rate', rate)
        set_text(sensor, 'topic', '/vision_trial/flow/'+topic)
        set_text(sensor, 'pose', '0 0 0 0 0 0' if kind == 'imu' else '0 0 -.020 0 1.5707963267948966 0')
        if kind == 'camera':
            camera = ET.SubElement(sensor, 'camera')
            set_text(camera, 'horizontal_fov', .8)
            set_text(camera, 'camera_info_topic', '/vision_trial/flow/camera_info')
            image = ET.SubElement(camera, 'image')
            set_text(image, 'width', 128)
            set_text(image, 'height', 128)
            set_text(image, 'format', 'L8')
            clip = ET.SubElement(camera, 'clip')
            set_text(clip, 'near', .01)
            set_text(clip, 'far', 5.)
        elif kind == 'gpu_lidar':
            lidar = ET.SubElement(sensor, 'lidar')
            horizontal = ET.SubElement(ET.SubElement(lidar, 'scan'), 'horizontal')
            for tag, value in [('samples', 3), ('resolution', 1), ('min_angle', -.0087), ('max_angle', .0087)]:
                set_text(horizontal, tag, value)
            distance = ET.SubElement(lidar, 'range')
            for tag, value in [('min', .02), ('max', 2.), ('resolution', .001)]:
                set_text(distance, tag, value)
        else:
            ET.SubElement(sensor, 'imu')
    ET.SubElement(world, 'plugin', filename='gz-sim-imu-system', name='gz::sim::systems::Imu')
    # Reuse the existing nonsemantic floor texture so flow can measure motion.
    # These pixels contain no marker IDs or map coordinates for the controller.
    from dynamic_drone.scene.ground_texture import floor_mesh
    from mission_profile import ROOT
    texture = ET.SubElement(world, 'model', name='ground_visual_texture')
    set_text(texture, 'static', 'true')
    link = ET.SubElement(texture, 'link', name='floor_pattern')
    visual = ET.SubElement(link, 'visual', name='texture')
    set_text(visual, 'pose', '0 0 .002 0 0 0')
    mesh = ET.SubElement(ET.SubElement(visual, 'geometry'), 'mesh')
    set_text(mesh, 'uri', str(floor_mesh(ROOT)))


def make_mission_world(directory, profile, seed):
    # Reuse the working physical aircraft definition from the isolated trial.
    staging = directory/'aircraft_build'
    staging.mkdir()
    aircraft_path, camera_offset = trial_world(staging, 7, .8)
    aircraft = deepcopy(ET.parse(aircraft_path).find("world/model[@name='crazyflie_2_1']"))
    # Common aircraft attitude loop: gentler correcting torque, more rate
    # damping. Keep rotor thrust/reaction-torque constants and vertical gains.
    controller = aircraft.find("plugin[@name='gz::sim::systems::MulticopterVelocityControl']")
    set_text(controller, 'attitudeGain', '0.0008 0.0008 0.0004')
    set_text(controller, 'angularRateGain', '0.00024 0.00024 0.00012')
    set_text(controller, 'maximumLinearAcceleration', '0.35 0.35 1')
    world_path, _ = preview_world(directory, profile, seed, 'marker')
    tree = ET.parse(world_path)
    world = tree.find('world')
    old_drone = world.find("model[@name='crazyflie_2_1']")
    world.remove(old_drone)
    world.append(aircraft)
    flow_enabled = profile['sensor_assistance']['flow_deck']
    if flow_enabled:
        add_flow_deck(world, aircraft)
    marker_size = float(profile['development_preview'].get('route_marker_size_m',.6))
    stage1_layout = simplify_marker_stage(world, marker_size)
    # Explicit rendering-only audit. Do not pass these poses to a controller.
    arrows = [m.get('name') for m in world.findall('model') if 'arrow' in m.get('name', '').lower()]
    if arrows:
        raise ValueError('Arrow geometry is forbidden in the marker-only mission: '+str(arrows))
    marker_models = [m.get('name') for m in world.findall('model') if m.get('name', '').startswith('route_marker_')]
    if len(marker_models) != 6:
        raise ValueError('Expected six ArUco route signs in development mission')
    validate_sdf(tree.getroot(), allow_flow_deck=flow_enabled)
    ET.indent(tree)
    tree.write(world_path, encoding='utf-8', xml_declaration=True)
    (directory/'mission_contract.json').write_text(json.dumps(dict(
        stages=['aruco_marker_navigation', 'visual_ring_passage', 'truck_visual_follow_and_landing'],
        route_marker_ids=[1, 2, 3, 4, 5, 6], truck_flag_id=23,
        marker_size_known_to_controller=True, scene_coordinates_given_to_controller=False,
        development_route_marker_size_m=marker_size,
        development_marker_size_definition='black_ArUco_square_side_excluding_white_quiet_border',
        stage1_difficulty='marker_only', route_marker_order='strict_1_to_6',
        stage1_motion_policy='constant_forward_speed_with_marker_PID_alignment',
        image_center_controller='filtered_PID',
        stage1_forward_acceleration_m_s2=.20,
        stage1_forward_jerk_m_s3=.40,
        stage1_depth_usage='display_only_no_position_range_or_height_fallback',
        stage1_depth_position_fallback_enabled=False,
        stage1_navigation_uses_metric_marker_memory=True,
        stage1_pose_estimator='multi_marker_RANSAC_PnP_when_available',
        stage1_corner_tracking_max_seconds=.25,
        stage1_tracked_corners_count_as_new_ID=False,
        stage1_fallback_reference='last_valid_marker_snapshot_historical_not_live',
        stage1_marker_map_source='observations_only_no_fixture_positions_or_reload',
        stage1_all_route_IDs_detected_while_moving=True,
        stage1_target_estimated_height_m=1.5,
        stage1_visible_target_vertical_control='marker_center_to_camera_principal_point',
        image_center_tolerance_px=10.,
        stage1_altitude_reference='initial_body_pose_marker_PnP_not_ground_range',
        stage1_visible_target_forward_speed_m_s=.50,
        stage1_missing_target_forward_speed_m_s=0.,
        stage1_unknown_marker_search_half_angle_deg=50.,
        stage1_known_marker_search='face_remembered_position_not_ttl_limited',
        stage1_unknown_target_direction_forward_allowed=False,
        stage1_marker_stand_off_m=3.5,
        stage1_completion='observed_standoff_reached_then_current_target_decoded_centered',
        stage1_initial_cached_IDs_complete_later_targets=False,
        stage1_dwell_seconds=0., stage1_estimated_vicinity_visit_required=True,
        stage1_image_area_alone_completes_visit=False,
        stage1_physical_touch_required=False, stage1_metric_accuracy_verified=False,
        stage1_takeoff_mode='concurrent_ascent_and_marker_mission',
        stage1_obstacle_count=0, inter_marker_obstacle_count=0,
        arrow_instructions=False, flow_deck=flow_enabled, downward_tof=flow_enabled,
        flow_deck_assistance=profile['sensor_assistance'],
        ring_passage_verification='ToF_15cm_drop_then_6cm_recovery_for_0.30s',
        truck_deck_verification='flag_aligned_ToF_20cm_surface_step',
        truck_touchdown_verification='ToF_6cm_near_surface_plus_simulation_contact',
        flow_deck_sensor_model='downward_camera_range_IMU_simulation_proxy_not_hardware_driver',
        camera='front_grayscale', hardware_ready=False,
        landing_reference=profile['landing_policy']['reference'],
        landing_policy=profile['landing_policy'], landing_verifier_implemented=True,
        simulation_landing_contact_check=True,
        truck_motion='0.15*cos(pi*stage3_elapsed/12)_m_s_simulation_actor_only'), indent=2))
    # Preview scene.json still describes a static view; replace misleading flags.
    scene_path = directory/'scene.json'
    scene = json.loads(scene_path.read_text())
    scene.update(flight=True, static_model=False, mode='full_marker_mission_experimental')
    scene['stage1_layout'] = stage1_layout
    scene['scene_variant'].update(mission_obstacle_types=[], stage1_obstacle_types=[],
                                  mission_route_style='marker_only_sequential_aruco',
                                  stage1_obstacle_count=0, stage2_geometry_changed=True)
    scene['sensor_topics'] = ['/vision_trial/image']
    if flow_enabled:
        scene['sensor_topics'] += ['/vision_trial/flow/'+name for name in ('image', 'range', 'imu')]
    scene.pop('assumed_preview_position', None)
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2))
    return world_path, camera_offset
