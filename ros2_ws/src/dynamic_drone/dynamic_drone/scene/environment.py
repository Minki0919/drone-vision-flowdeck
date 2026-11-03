# 파일 역할: 장애물·조명·센서의 Gazebo 환경 구성
"""Seeded physical research scenes. Scene truth never enters the policy."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
from copy import deepcopy
import math
import xml.etree.ElementTree as ET
import numpy as np


# 처리: 환경의 장애물·조명 조건을 선택
def sample_research(config, rng, profile):
    from dynamic_drone.scene.variation import vary_course
    base=deepcopy(config)
    # 처리: 환경 조건에서 임의 값을 선택
    def sample(key):
        lo,hi=profile[key]
        if not np.isfinite([lo,hi]).all() or lo>hi:
            raise ValueError(f'Invalid research bounds: {key}')
        return float(rng.uniform(lo,hi))
    base['obstacle_avoidance']['width']=sample('wall_width')
    base['obstacle_avoidance']['thickness']=sample('wall_thickness')
    cfg=vary_course(base,rng,profile['layout'])
    variant=cfg['scene_variant']
    az=math.radians(sample('sun_azimuth_deg')); el=math.radians(sample('sun_elevation_deg'))
    variant.update(name='research_v2',ambient=sample('ambient'),sun_diffuse=sample('sun_diffuse'),
        sun_direction=[math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),-math.sin(el)],
        room_style=str(rng.choice(profile['room_styles'])),wall_height=sample('wall_height'),
        obstacle_types=[str(rng.choice(profile['obstacle_types'])) for _ in range(2)])
    for obstacle in cfg['obstacle_avoidance']['obstacles']:
        obstacle['position'][2]=variant['wall_height']/2
    if 'cylinder' in variant['obstacle_types']:
        cfg['obstacle_avoidance']['thickness']=max(cfg['obstacle_avoidance']['thickness'],cfg['obstacle_avoidance']['width'])
    diameter=sample('ring_diameter')
    cfg['gate'].update(shape='circle',width=diameter,height=diameter,max_vertical_offset=.35)
    cfg['research']=dict(sensor_stress={k:sample(k) for k in (
        'camera_delay_seconds','camera_dropout_probability','camera_noise_std',
        'occlusion_fraction','occlusion_probability')},private_depth_ground_truth=False,
        imu=True,optical_flow=False,downward_tof=False,firmware_backend=False)
    return cfg


# 처리: XML 항목의 텍스트 값을 설정
def text(parent, tag, value):
    node=parent.find(tag)
    if node is None:node=ET.SubElement(parent,tag)
    node.text=str(value);return node


# 처리: 충돌 형상과 렌더링 형상을 가진 구조물 생성
def solid(world,name,position,size,kind='box',color=(.45,.45,.45),rotation=(0,0,0)):
    model=ET.SubElement(world,'model',name=name)
    text(model,'static','true'); text(model,'pose',' '.join(map(str,[*position,*rotation])))
    link=ET.SubElement(model,'link',name='body')
    for tag in ('visual','collision'):
        obj=ET.SubElement(link,tag,name=tag)
        geometry=ET.SubElement(obj,'geometry')
        shape=ET.SubElement(geometry,kind)
        if kind=='box': text(shape,'size',' '.join(map(str,size)))
        else:
            text(shape,'radius',size[1]/2); text(shape,'length',size[2])
        if tag=='visual':
            material=ET.SubElement(obj,'material')
            for key in ('ambient','diffuse'): text(material,key,' '.join(map(str,[*color,1])))
    return model


# 처리: 장애물과 조명·재질 조건을 월드에 적용
def apply_research(world,cfg):
    if 'research' not in cfg: return
    variant=cfg['scene_variant']
    world.find("light[@name='sun']/direction").text=' '.join(map(str,variant['sun_direction']))
    if variant.get('lighting_mode')=='bright_fixed':
        text(world.find("light[@name='sun']"),'cast_shadows','false')
    for i,(obstacle,kind) in enumerate(zip(cfg['obstacle_avoidance']['obstacles'],variant['obstacle_types']),1):
        old=world.find(f"model[@name='obstacle_wall_{i}']"); world.remove(old)
        width=cfg['obstacle_avoidance']['width']; thickness=cfg['obstacle_avoidance']['thickness']; height=variant['wall_height']
        position=list(obstacle['position']); position[2]=height/2
        if kind=='panels':
            # Three touching panels form the same conservative obstacle envelope.
            for j in range(3):
                p=position.copy(); p[1]+=(j-1)*width/3
                solid(world,f'obstacle_wall_{i}_panel_{j}',p,[thickness,width/3,height],color=(.9,.25,.1))
        else:
            # A cylinder's diameter must match its avoidance envelope in X and Y.
            if kind=='cylinder':
                obstacle['collision_thickness']=width
            solid(world,f'obstacle_wall_{i}',position,[thickness,width,height],kind,color=(.9,.25,.1))
    # True circular openings, approximated by 48 overlapping physical rim segments.
    diameter=cfg['gate']['width']; radius=diameter/2; rim=cfg['gate']['frame_thickness']
    for i,center in enumerate(cfg['gate']['centers'],1):
        world.remove(world.find(f"model[@name='air_gate_{i}']"))
        for j in range(48):
            angle=2*math.pi*j/48
            p=[center[0],center[1]+(radius+rim/2)*math.cos(angle),center[2]+(radius+rim/2)*math.sin(angle)]
            solid(world,f'air_gate_{i}_rim_{j}',p,[rim,rim,2*(radius+rim)*math.sin(math.pi/48)*1.1],
                  color=(.025,.025,.025),rotation=(angle,0,0))
    style=variant['room_style']
    if style in ('corridor','warehouse'):
        # Background geometry remains outside the bounded flight envelope.
        for y in (-2.7,6.8):
            solid(world,f'room_side_{y}',[22,y,2.5],[46,.15,5],color=(.7,.72,.75))
    if style=='warehouse':
        for i,x in enumerate(range(5,36,5)):
            solid(world,f'clutter_box_{i}',[x,5.2,1.0],[1.0,1.0,2.0],color=(.35+.04*(i%3),.3,.22))
            solid(world,f'clutter_pillar_{i}',[x,-1.9,1.5],[.5,.5,3.],kind='cylinder',color=(.6,.6,.6))
    # Preserve the existing floor appearance.
    for i in range(36):
        for j,y in enumerate((-1.5,-.5,.5,1.5,2.5,3.5,4.5)):
            if (i+j)%2:
                solid(world,f'floor_tile_{i}_{j}',[i,y,.001],[.9,.9,.001],color=(.4,.4,.4))
    add_sensors(world,cfg)
    if cfg['research'].get('firmware_backend'):
        drone=world.find("model[@name='crazyflie_2_1']")
        for plugin in list(drone.findall('plugin')):
            if 'multicopter-control-system' in plugin.get('filename',''):drone.remove(plugin)
            elif 'multicopter-motor-model-system' in plugin.get('filename',''):
                text(plugin,'commandSubTopic','command/motor_speed')
                text(plugin,'maxRotVelocity',2618)
        drone.find("plugin[@name='gz::sim::systems::OdometryPublisher']/odom_publish_frequency").text='500'


# 처리: 환경 생성용 카메라 센서 항목 작성
def camera(link,name,topic,pose,kind='camera',size=128):
    sensor=ET.SubElement(link,'sensor',name=name,type=kind)
    text(sensor,'pose',pose); text(sensor,'always_on','true'); text(sensor,'update_rate',20)
    text(sensor,'topic',topic)
    c=ET.SubElement(sensor,'camera'); text(c,'horizontal_fov',1.2)
    im=ET.SubElement(c,'image'); text(im,'width',size);text(im,'height',size);text(im,'format','R8G8B8')
    clip=ET.SubElement(c,'clip');text(clip,'near',.01);text(clip,'far',30)
    text(c,'triggered','true');text(c,'trigger_topic',topic+'/trigger')
    return sensor


# 처리: 환경 센서 구성; 최종 실행 경로에서 금지 센서 검사
def add_sensors(world,cfg):
    drone=world.find("model[@name='crazyflie_2_1']");link=drone.find("link[@name='base_link']")
    sensor=ET.SubElement(link,'sensor',name='research_imu',type='imu')
    text(sensor,'always_on','true');text(sensor,'update_rate',200);text(sensor,'topic','/crazyflie_2_1/imu')
    imu=ET.SubElement(sensor,'imu')
    for kind,std in [('angular_velocity',.002),('linear_acceleration',.02)]:
        element=ET.SubElement(imu,kind)
        for axis in 'xyz':
            noise=ET.SubElement(ET.SubElement(element,axis),'noise',type='gaussian')
            text(noise,'mean',0);text(noise,'stddev',std)
    ET.SubElement(world,'plugin',filename='gz-sim-imu-system',name='gz::sim::systems::Imu')
    if cfg['research'].get('private_depth_ground_truth'):
        front=link.find("sensor[@name='front_camera']")
        camera(link,'training_depth','/crazyflie_2_1/training_rgbd',front.findtext('pose'),kind='rgbd_camera',size=128)
