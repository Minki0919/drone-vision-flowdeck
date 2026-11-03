# 파일 역할: 환경 표식·장애물 배치; 배치 좌표를 제어기에 전달하지 않음
"""Physical L/N/U/zigzag obstacles, closed sides, route arrows and image tags."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import os
import math
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from dynamic_drone.scene.environment import sample_research,solid,text
from dynamic_drone.scene.geometry import astar

# 파트: 공통 경로·좌표계·설정값
OBSTACLE_HEIGHT_LIMIT=2.0

# 처리: 환경 배치·표식·조명 설정 구성
def configure(base,rng,profile,stage='full'):
    cfg=sample_research(base,rng,profile)
    lighting=profile.get('mission_lighting_mode','bright_fixed')
    if lighting not in ('bright_fixed','randomized'):raise ValueError('Invalid mission_lighting_mode')
    if lighting=='bright_fixed':
        # Override AFTER sampling, preserving seeded obstacle/route RNG draws.
        # Lighting config is rendering-only, never a navigation/map input.
        ambient=float(profile.get('mission_bright_ambient',.65))
        diffuse=float(profile.get('mission_bright_sun_diffuse',.90))
        direction=np.asarray(profile.get('mission_bright_sun_direction',[.25,-.15,-.956556]),float)
        if not 0<=ambient<=1 or not 0<=diffuse<=1 or direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<.01:
            raise ValueError('Invalid bright mission lighting')
        cfg['scene_variant'].update(ambient=ambient,sun_diffuse=diffuse,sun_direction=(direction/np.linalg.norm(direction)).tolist(),lighting_mode='bright_fixed',sun_cast_shadows=False)
    else:cfg['scene_variant']['lighting_mode']='randomized'
    mirror=int(rng.choice([-1,1]));kinds=['L','N','U','zigzag','T','slalom'];rng.shuffle(kinds);kinds=kinds[:4]
    style='zigzag' if kinds[0]=='zigzag' else 's_curve' if kinds[0]=='L' else 'dogleg'
    ys={'zigzag':[0.,1.6,-1.5,1.5,-1.4,0.], 's_curve':[0.,.8,1.6,.8,-.8,0.], 'dogleg':[0.,1.5,1.5,-1.5,-1.5,0.]}[style]
    points=[[3.+i*4.,mirror*y,1.5] for i,y in enumerate(ys)];rectangles=[]
    for i,(x,kind) in enumerate(zip([5.,9.,13.,17.],kinds)):
        side=mirror*(-1 if i%2 else 1);y=side*float(rng.uniform(.65,1.2))
        pieces=[(0,0,.35,2.)]
        if kind=='L':pieces+=[(.7,side*.85,1.4,.3)]
        elif kind=='N':pieces+=[(-.7,side*.85,1.4,.3)]
        elif kind=='U':pieces+=[(-.6,.85,1.2,.3),(-.6,-.85,1.2,.3)]
        elif kind=='zigzag':pieces+=[(.65,.85,1.3,.3),(-.65,-.85,1.3,.3)]
        elif kind=='T':pieces+=[(0,side*.85,1.8,.3)]
        else:pieces=[(-.65,.65,.30,1.4),(.65,-.65,.30,1.4)]
        angle=float(rng.choice([-.35,0.,.35,math.pi/2,-math.pi/2]));c,s=math.cos(angle),math.sin(angle)
        for j,(dx,dy,sx,sy) in enumerate(pieces):rectangles.append(dict(name=f'planned_obstacle_{i}_{kind}_{j}',kind=kind,x=x+c*dx-s*dy,y=y+s*dx+c*dy,sx=sx,sy=sy,heading=angle,height=OBSTACLE_HEIGHT_LIMIT))
    # Physical clutter on both sides, leaving a traversable central corridor.
    for i,x in enumerate((7.,11.,15.,19.)):
        for side in (-1,1):
            rectangles.append(dict(name=f'side_obstacle_{i}_{side}',kind='T',x=x+float(rng.uniform(-.35,.35)),y=side*float(rng.uniform(2.7,3.1)),sx=float(rng.uniform(.4,.75)),sy=float(rng.uniform(.5,.8)),heading=float(rng.uniform(-.25,.25)),height=float(rng.uniform(1.6,OBSTACLE_HEIGHT_LIMIT))))
    course=dict(version='mission_plan_wall_signs_v5',bounds=[-1.,145.,-4.,4.],body_radius=.12,planning_margin=.30,
        grid_resolution=.25,obstacle_rectangles=rectangles,route_marker_size=.5,marker_approach_distance=.85,
        route_markers=[dict(id=i+1,position=p,heading=0.) for i,p in enumerate(points)],
        route_choice_actions=['shortest','prefer_left','prefer_right','cautious','hold'],
        map_source='provided static simulator course map; explicitly available to global planner',
        arrow_source='floor visuals aligned to safe nominal A* route; navigation completion requires camera ArUco',
        route_style=style if mirror>0 else 'mirrored_'+style)
    from dynamic_drone.scene.geometry import clearance
    for marker in course['route_markers']:
        x,y,z=marker['position']
        candidates=[[x,float(other),z] for other in np.linspace(-2.8,2.8,57) if clearance([x,other],course)>=.55]
        if not candidates:raise ValueError('No safe sign approach position')
        marker['position']=min(candidates,key=lambda other:abs(other[1]-y))
    previous=cfg['start_position'][:2];previous[1]=0.;cfg['start_position']=[0.,0.,.022]
    
    for marker in course['route_markers']:
        target=marker['position'][:2];path=astar(previous,target,course)
        # Boards face incoming motion; their world coordinates are for rendering.
        direction=np.asarray(target)-np.asarray(path[-2]);theta=float(math.atan2(direction[1],direction[0]))
        from dynamic_drone.scene.geometry import clearance
        candidates=[theta,0.,math.pi/4,-math.pi/4,math.pi/2,-math.pi/2]
        valid=[angle for angle in candidates if clearance(np.asarray(target)+course['marker_approach_distance']*np.array([math.cos(angle),math.sin(angle)]),course)>.08]
        if not valid:raise ValueError('No visible/free marker board placement')
        marker['heading']=min(valid,key=lambda angle:abs((angle-theta+math.pi)%(2*math.pi)-math.pi))
        if marker['id']==len(course['route_markers']):marker['heading']=mirror*math.pi/2  # Exit-side sign cannot occlude the first ring.
        previous=target
    course['floor_arrows']=[]
    mode=os.getenv('DYNAMIC_ROUTE_SIGN_MODE',profile.get('route_sign_mode','alternating'))
    if mode not in ('alternating','random','aruco','arrow'):raise ValueError('Invalid route_sign_mode')
    first=int(rng.integers(0,2));kinds_sign=[]
    for i,marker in enumerate(course['route_markers']):
        kind=('aruco' if (i+first)%2==0 else 'arrow') if mode=='alternating' else str(rng.choice(['aruco','arrow'])) if mode=='random' else mode
        marker['kind']=kind;kinds_sign.append(kind)
        next_point=course['route_markers'][i+1]['position'] if i+1<len(points) else [26.,0.,1.5]
        bearing=math.atan2(next_point[1]-marker['position'][1],next_point[0]-marker['position'][0])
        turn=(bearing-marker['heading']+math.pi)%(2*math.pi)-math.pi
        from dynamic_drone.perception.wall_signs import TURN
        marker['arrow_label']=min(TURN,key=lambda label:abs((turn-TURN[label]+math.pi)%(2*math.pi)-math.pi))
    course.update(route_sign_mode=mode,wall_arrow_size=.8,route_sign_kinds=kinds_sign,
        arrow_source='YOLO class and OpenCV wall-board PnP; exclusive sign per route stop')
    for sign in course['route_markers']:
        theta=sign['heading'];center=np.asarray(sign['position'][:2])+.91*np.array([math.cos(theta),math.sin(theta)])
        course['obstacle_rectangles'].append(dict(name=f"route_sign_panel_{sign['id']}",kind='sign_panel',x=float(center[0]),y=float(center[1]),sx=.06,sy=1.05,heading=float(theta),height=1.05))
    cfg['mission_plan']=course;cfg['scene_variant']['name']='mission_plan_v3'
    cfg['scene_variant'].update(course_start=[0.,0.,.022],mission_obstacle_types=kinds,mission_route_style=course['route_style'],render_profile='closed_side_corridor',obstacle_distribution='alternating_left_right_with_bilateral_clutter',obstacle_height_limit_m=OBSTACLE_HEIGHT_LIMIT)
    cfg['visual_navigation']['markers']=[dict(id=1,position=[3.,0.,1.5],direction='left'),dict(id=2,position=[7.,mirror*1.6,1.5],direction='right')]
    cfg['gate']['centers']=[[26.+i*4.3,0. if i%2==0 else mirror*.5,1.5] for i in range(4)]
    cfg['truck_start']=[42.7,mirror*.5,.3];cfg['vehicle_speed']=float(rng.uniform(.30,.50))
    cfg['flag_follow'].update(duration=3.,tolerance=.22,max_speed=.75)
    cfg['relative_speed_tolerance']=.065;cfg['stage_timeout']=120.
    cfg['geofence']=[-1.,145.,-3.85,3.85,-.2,3.0]
    cfg['front_tof']['prediction_timeout']=16.;cfg['front_tof']['deck_confirmations_required']=5
    if os.getenv('DYNAMIC_ARROW_OBSTACLE_TRIAL')=='1':configure_obstacle_arrow_trial(cfg)
    if stage=='navigation':cfg['start_position'][2]=cfg['flight_height']
    elif stage=='gates':cfg['start_position']=[23.,0.,1.5]
    elif stage=='landing':
        cfg['start_position']=[42.7+cfg['flag_follow']['marker_offset'][0]-cfg['flag_follow']['distance']-.035,mirror*.5,1.5]
    return cfg


# 처리: ArUco 사전으로 마커판 렌더링 형상 생성
def board(world,identifier,point,heading,size):
    quiet=size/6;outer=size+quiet*2
    model=ET.SubElement(world,'model',name=f'route_marker_{identifier}');text(model,'static','true')
    text(model,'pose',' '.join(map(str,[*point,0,0,heading])))
    link=ET.SubElement(model,'link',name='tag')
    from dynamic_drone.perception.aruco_dictionary import get_dictionary
    dictionary=get_dictionary()
    image=cv2.aruco.generateImageMarker(dictionary,identifier,6) if hasattr(cv2.aruco,'generateImageMarker') else cv2.aruco.drawMarker(dictionary,identifier,6)
    # 처리: 마커판의 한 비트 셀을 렌더링 형상으로 생성
    def pixel(name,position,dimensions,color):
        visual=ET.SubElement(link,'visual',name=name);text(visual,'pose',' '.join(map(str,[*position,0,0,0])))
        geometry=ET.SubElement(visual,'geometry');text(ET.SubElement(geometry,'box'),'size',' '.join(map(str,dimensions)))
        material=ET.SubElement(visual,'material')
        for key in ('ambient','diffuse'):text(material,key,' '.join(map(str,[color]*3+[1.])))
    pixel('white_board',[0,0,0],[.005,outer,outer],1.)
    for row in range(6):
        for col in range(6):
            if image[row,col]==0:pixel(f'bit_{row}_{col}',[-.006,(2.5-col)*quiet,(2.5-row)*quiet],[.004,quiet*1.015,quiet*1.015],.005)


# 처리: 환경 설정의 표식·구조물을 월드에 반영
def apply(world,cfg,root):
    course=cfg['mission_plan']
    if cfg.get('sensor_only_controller'):
        # Nonsemantic floor texture: optical flow needs visible corners, not route arrows.
        texture=ET.SubElement(world,'model',name='sensor_flow_floor_texture');text(texture,'static','true');link=ET.SubElement(texture,'link',name='floor_pattern')
        from dynamic_drone.scene.ground_texture import floor_mesh
        visual=ET.SubElement(link,'visual',name='batched_floor_corners');text(visual,'pose','0 0 .025 0 0 0')
        geometry=ET.SubElement(visual,'geometry');mesh=ET.SubElement(geometry,'mesh');text(mesh,'uri',str(floor_mesh(root)))
        truck=world.find("model[@name='mission_vehicle']/link[@name='chassis']")
        if truck is not None:
            for i in range(7):
                for j in range(9):
                    patch=ET.SubElement(truck,'visual',name=f'flow_deck_{i}_{j}')
                    text(patch,'pose',f'{-.90+i*.12} {-.60+j*.15} .309 0 0 0')
                    geo=ET.SubElement(patch,'geometry');text(ET.SubElement(geo,'box'),'size','.045 .045 .001')
                    material=ET.SubElement(patch,'material')
                    for key in ('ambient','diffuse'):text(material,key,'.04 .04 .04 1')
    for model in list(world.findall('model')):
        name=model.get('name','')
        if name.startswith(('marker_','obstacle_wall_','room_side_','warehouse_','room_pillar','clutter_')):world.remove(model)
    for side in (-1,1):solid(world,f'uncrossable_side_{side}',[71.,side*4.15,2.25],[150.,.30,4.5],color=(.58,.58,.58))
    for rect in course['obstacle_rectangles']:solid(world,rect['name'],[rect['x'],rect['y'],rect['height']/2],[rect['sx'],rect['sy'],rect['height']],color=(.3,.3,.3),rotation=(0,0,rect.get('heading',0.)))
    for marker in course['route_markers']:
        theta=marker['heading'];offset=course['marker_approach_distance'];point=np.asarray(marker['position'])+np.array([offset*math.cos(theta),offset*math.sin(theta),0])
        # The sign is physically mounted on a short vertical wall panel.
        if not marker.get('attached_obstacle'):
            solid(world,f"route_sign_wall_{marker['id']}",(point+np.array([.06*math.cos(theta),.06*math.sin(theta),0])).tolist(),[.06,1.05,1.0],color=(.22,.22,.22),rotation=(0,0,theta))
        if marker['kind']=='aruco':board(world,marker['id'],point.tolist(),theta,course['route_marker_size'])
        else:wall_arrow_board(world,marker['id'],point.tolist(),theta,course['wall_arrow_size'],marker['arrow_label'])


# 처리: 환경 생성 보조용 화살표판 형상 구성
def wall_arrow_board(world,identifier,point,heading,size,label):
    from dynamic_drone.perception.wall_signs import arrow_image
    image=arrow_image(label,32);cell=size/32
    model=ET.SubElement(world,'model',name=f'wall_arrow_{identifier}');text(model,'static','true');text(model,'pose',' '.join(map(str,[*point,0,0,heading])))
    link=ET.SubElement(model,'link',name='sign')
    # 처리: 화살표 표식의 한 구성 요소를 렌더링 형상으로 생성
    def box(name,x,y,z,sy,sz,color):
        visual=ET.SubElement(link,'visual',name=name);text(visual,'pose',f'{x} {y} {z} 0 0 0')
        text(ET.SubElement(ET.SubElement(visual,'geometry'),'box'),'size',f'.004 {sy} {sz}')
        material=ET.SubElement(visual,'material')
        for key in ('ambient','diffuse'):text(material,key,f'{color} {color} {color} 1')
    box('black_frame',.005,0,0,size+.06,size+.06,.01)
    box('white_square',0,0,0,size,size,.98)
    for row in range(32):
        for col in range(32):
            if image[row,col]==0:box(f'arrow_{row}_{col}',-.006,(15.5-col)*cell,(15.5-row)*cell,cell*1.01,cell*1.01,.005)


# 처리: 환경 생성 보조용 화살표 장애물 배치 구성
def configure_obstacle_arrow_trial(cfg):
    """Controlled test: arrows on actual L walls, then the corridor side wall."""
    course=cfg['mission_plan'];offset=course['marker_approach_distance']
    course['obstacle_rectangles']=[
        dict(name='arrow_trial_L_vertical',kind='L',x=4.,y=0.,sx=.25,sy=2.6,height=OBSTACLE_HEIGHT_LIMIT,heading=0.),
        dict(name='arrow_trial_L_foot',kind='L',x=4.65,y=-1.15,sx=1.30,sy=.30,height=OBSTACLE_HEIGHT_LIMIT,heading=0.),
        dict(name='arrow_trial_N_vertical',kind='N',x=9.,y=1.9,sx=.25,sy=2.5,height=OBSTACLE_HEIGHT_LIMIT,heading=0.),
        dict(name='arrow_trial_N_foot',kind='N',x=9.65,y=3.0,sx=1.30,sy=.30,height=OBSTACLE_HEIGHT_LIMIT,heading=0.)]
    signs=[]
    for i,(point,heading,label,attached) in enumerate([
        ([3.86,0.,1.5],0.,'left','arrow_trial_L_vertical'),
        ([8.86,1.9,1.5],0.,'right','arrow_trial_N_vertical'),
        ([12.,-3.985,1.5],-math.pi/2,'left','uncrossable_side_-1')]):
        approach=np.asarray(point)-offset*np.array([math.cos(heading),math.sin(heading),0.])
        signs.append(dict(id=i+1,position=approach.tolist(),heading=heading,kind='arrow',arrow_label=label,attached_obstacle=attached))
    course.update(route_markers=signs,route_sign_kinds=['arrow']*3,route_sign_mode='arrow',floor_arrows=[],route_style='obstacle_mounted_arrow_trial',arrow_source='YOLO/OpenCV signs mounted on actual obstacle faces; provided map safety planner')
    cfg['start_position']=[0.,0.,cfg['flight_height']]
    cfg['scene_variant'].update(mission_obstacle_types=['L','N'],mission_route_style=course['route_style'])
