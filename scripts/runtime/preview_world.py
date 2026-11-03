#!/usr/bin/env python3
# 파일 역할: 전방 카메라 환경 생성과 소유 프로세스 종료
"""Static Gazebo + front camera preview. No flight controller or command publisher."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import json, os, signal, subprocess
import xml.etree.ElementTree as ET
from mission_profile import ROOT


# 처리: 렌더링용 월드 생성; 좌표는 제어 입력으로 사용하지 않음
def make_world(directory,profile,seed,view):
    import numpy as np
    import yaml
    from dynamic_drone.scene.markers import configure
    from dynamic_drone.scene.world import write_course_world
    from dynamic_drone.scene.ring_course import add_ring_corridor
    from dynamic_drone.safety.sensor_policy import validate_sdf,validate_config
    from gazebo_assets import relocate_meshes
    base=yaml.safe_load((ROOT/'ros2_ws/src/dynamic_drone/config/gazebo_mission.yaml').read_text())
    variation=yaml.safe_load((ROOT/'ros2_ws/src/dynamic_drone/config/research_v2.yaml').read_text())
    variation['layout']=yaml.safe_load((ROOT/'ros2_ws/src/dynamic_drone/config/course_variation_v1.yaml').read_text())
    variation['route_sign_mode']='aruco';variation['mission_lighting_mode']=profile['development_preview']['lighting']
    # Legacy scene builder also sets dormant landing fields. They never reach
    # a controller and are discarded before saving this preview configuration.
    base['front_tof']={}
    previous=os.environ.get('DYNAMIC_ROUTE_SIGN_MODE');trial=os.environ.pop('DYNAMIC_ARROW_OBSTACLE_TRIAL',None)
    os.environ['DYNAMIC_ROUTE_SIGN_MODE']='aruco'
    try:cfg=configure(base,np.random.default_rng(seed),variation,'full')
    finally:
        if previous is None:os.environ.pop('DYNAMIC_ROUTE_SIGN_MODE',None)
        else:os.environ['DYNAMIC_ROUTE_SIGN_MODE']=previous
        if trial is not None:os.environ['DYNAMIC_ARROW_OBSTACLE_TRIAL']=trial
    cfg.pop('front_tof',None);cfg['sensor_suite']='front_camera_only'
    cfg['mission_plan']['route_marker_size']=float(profile['development_preview'].get('route_marker_size_m',.6))
    cfg['research'].update(optical_flow=False,downward_tof=False,private_depth_ground_truth=False)
    cfg['simulation_target_real_time_factor']=1.
    path=directory/'world.sdf';write_course_world(ROOT,path,cfg);relocate_meshes(path,ROOT)
    tree=ET.parse(path);world=tree.find('world');drone=world.find("model[@name='crazyflie_2_1']")
    # This is an inspection camera position, NOT a pose estimate or flight.
    position=[0.,0.,1.5]
    if view=='gates':position=[cfg['gate']['centers'][0][0]-3.,0.,1.5]
    elif view=='truck':position=[cfg['truck_start'][0]-3.,cfg['truck_start'][1],1.5]
    drone.find('pose').text=' '.join(map(str,[*position,0.,0.,0.]))
    static=drone.find('static')
    if static is None:static=ET.SubElement(drone,'static')
    static.text='true'
    for parent in list(drone.iter()):
        for child in list(parent):
            if child.tag=='plugin' or (child.tag=='sensor' and child.get('name')!='front_camera'):parent.remove(child)
    front=drone.find(".//sensor[@name='front_camera']")
    if front is None:raise RuntimeError('Front camera missing')
    front.find('update_rate').text=str(profile['development_preview']['camera_fps'])
    camera=front.find('camera')
    for key in ('triggered','trigger_topic'):
        element=camera.find(key)
        if element is not None:camera.remove(element)
    camera.find('image/width').text=str(profile['development_preview']['camera_width'])
    camera.find('image/height').text=str(profile['development_preview']['camera_height'])
    camera.find('image/format').text='L8'
    stage2_layout=add_ring_corridor(world,cfg['gate'])
    validate_config(cfg);validate_sdf(tree.getroot())
    tree.write(path,encoding='utf-8',xml_declaration=True)
    (directory/'scene.json').write_text(json.dumps(dict(seed=seed,view=view,assumed_preview_position=position,
        sensor_topics=[front.findtext('topic')],scene_parameters='development_only_not_official',
        flight=False,static_model=True,scene_variant=cfg['scene_variant'],
        stage2_layout=stage2_layout),ensure_ascii=False,indent=2))
    return path,front.findtext('topic')


# 처리: 이 실행에서 생성한 프로세스 그룹만 종료
def stop(process):
    if process is None:return
    # Only the process group created by this preview is addressed.
    try:os.killpg(process.pid,signal.SIGINT)
    except ProcessLookupError:return
    try:process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait(timeout=2)

