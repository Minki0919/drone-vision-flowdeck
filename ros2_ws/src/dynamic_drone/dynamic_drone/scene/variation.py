# 파일 역할: 환경 배치와 렌더링 조건 변화
"""Bounded scene variation; generated truth is for world building, not control."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np

# 파트: 공통 경로·좌표계·설정값
ENVIRONMENTS=('baseline','varied_v1','research_v2','mission_plan_v3')


# 처리: 환경 요소의 위치·크기 변화 설정
def vary_course(config, rng, profile):
    if profile.get('name')=='research_v2':
        from dynamic_drone.scene.environment import sample_research
        return sample_research(config,rng,profile)
    cfg=deepcopy(config)
    # 처리: 환경 조건에서 임의 값을 선택
    def sample(key):
        bounds=np.asarray(profile[key],dtype=float)
        if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>bounds[1]:
            raise ValueError(f'Invalid variation bounds: {key}')
        return float(rng.uniform(*bounds))
    y=sample('start_y')
    cfg['start_position'][1]=y
    lateral=sample('lateral_distance')
    cfg['visual_navigation']['lateral_distance']=lateral
    bypass=cfg['obstacle_avoidance']['width']/2+cfg['obstacle_avoidance']['clearance']
    marker1=[sample('marker1_x'),y,cfg['flight_height']]
    wall1=[marker1[0]+sample('wall1_ahead'),y+lateral,1.0]
    marker2=[wall1[0]+sample('marker2_after_wall1'),wall1[1]+bypass,cfg['flight_height']]
    wall2=[marker2[0]+sample('wall2_after_marker2'),y+bypass,1.0]
    cfg['visual_navigation']['markers'][0]['position']=marker1
    cfg['visual_navigation']['markers'][1]['position']=marker2
    cfg['obstacle_avoidance']['obstacles'][0]['position']=wall1
    cfg['obstacle_avoidance']['obstacles'][1]['position']=wall2
    gate_x=wall2[0]+sample('first_gate_after_wall2')
    low=y+sample('gate_low_y')
    high=low+sample('gate_zigzag')
    gates=[]
    for index in range(4):
        if index:
            gate_x+=sample('gate_spacing')
        gates.append([gate_x,low if index%2==0 else high,
                      cfg['flight_height']+sample('gate_height_delta')])
    cfg['gate']['centers']=gates
    cfg['truck_start']=[gate_x+sample('truck_after_gate4'),high+sample('truck_y_delta'),config['truck_start'][2]]
    cfg['scene_variant']=dict(name=profile['name'],course_start=list(cfg['start_position']),
        ambient=sample('ambient'),sun_diffuse=sample('sun_diffuse'),ground_gray=sample('ground_gray'))
    # Nearest-gate association remains valid and acquisition stays ahead of us.
    gaps=np.diff([point[0] for point in gates])
    if not (np.all(gaps>=4.) and np.all(gaps<=cfg['gate']['spacing_max'])):
        raise ValueError('Randomized gates violate supported ordering/spacing')
    if marker2[0]-wall1[0] < 2.5 or cfg['truck_start'][0]-gate_x < 3.:
        raise ValueError('Insufficient room to acquire the next target')
    return cfg


# 처리: 변형 환경 설정을 SDF에 반영
def apply_scene_variant(world, cfg, baseline):
    """Move model transforms, including models with world-valued link poses."""
    variant=cfg.get('scene_variant')
    if variant is None:
        return
    # 처리: 환경 모델을 새 배치 좌표로 이동
    def relocate(name, new, original):
        model=world.find(f"model[@name='{name}']")
        if model is None:
            raise ValueError(f'Missing required scene model: {name}')
        pose=model.find('pose')
        if pose is None:
            pose=ET.SubElement(model,'pose')
            previous=[0.]*6
        else:
            previous=list(map(float,pose.text.split()))
        delta=np.asarray(new)-np.asarray(original)
        pose.text=' '.join(map(str,[*(np.asarray(previous[:3])+delta),*previous[3:]]))
    for index in range(2):
        relocate(f'marker_{index+1}',cfg['visual_navigation']['markers'][index]['position'],
                 baseline['visual_navigation']['markers'][index]['position'])
        relocate(f'obstacle_wall_{index+1}',cfg['obstacle_avoidance']['obstacles'][index]['position'],
                 baseline['obstacle_avoidance']['obstacles'][index]['position'])
    for index in range(4):
        relocate(f'air_gate_{index+1}',cfg['gate']['centers'][index],baseline['gate']['centers'][index])
    relocate('mission_vehicle',cfg['truck_start'],baseline['truck_start'])
    # Curriculum airborne starts must not move the takeoff pad into the air.
    delta=np.asarray(variant['course_start'])-np.asarray(baseline['start_position'])
    for name in ('takeoff_pad','start_border_0','start_border_1','stage1_area','stage2_area','stage3_area'):
        relocate(name,delta,[0.,0.,0.])
    scene=world.find('scene')
    if scene is None:
        scene=ET.SubElement(world,'scene')
    ambient=scene.find('ambient')
    if ambient is None:
        ambient=ET.SubElement(scene,'ambient')
    ambient.text=' '.join(map(str,[variant['ambient']]*3+[1.]))
    world.find("light[@name='sun']/diffuse").text=' '.join(map(str,[variant['sun_diffuse']]*3+[1.]))
    material=world.find("model[@name='ground_plane']/link/visual/material")
    for key in ('ambient','diffuse'):
        material.find(key).text=' '.join(map(str,[variant['ground_gray']]*3+[1.]))
