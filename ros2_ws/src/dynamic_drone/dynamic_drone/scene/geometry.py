# 파일 역할: 환경 배치용 거리·경로 계산; 기체에 제공하는 지도가 아님
"""Known static course map: inflated A*, clearance and route-choice costs.

This map is explicitly provided to the planner, not inferred from camera depth.
Dynamic truck state is never part of this map or policy observation.
"""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import heapq
import math
import numpy as np

# 파트: 공통 경로·좌표계·설정값
ACTIONS=('shortest','prefer_left','prefer_right','cautious','hold')


# 처리: 환경 배치용 사각 장애물까지의 거리 계산
def rectangle_distance(point,rectangle):
    angle=rectangle.get('heading',0.);c,s=math.cos(angle),math.sin(angle)
    x,y=point[0]-rectangle['x'],point[1]-rectangle['y']
    dx=max(abs(c*x+s*y)-rectangle['sx']/2,0.)
    dy=max(abs(-s*x+c*y)-rectangle['sy']/2,0.)
    return math.hypot(dx,dy)


# 처리: 환경의 장애물·경계에 대한 배치 여유 계산
def clearance(point,course):
    distances=[rectangle_distance(point,r) for r in course['obstacle_rectangles']]
    xmin,xmax,ymin,ymax=course['bounds']
    distances.extend([point[0]-xmin,xmax-point[0],point[1]-ymin,ymax-point[1]])
    return max(0.,min(distances)-course['body_radius'])


# 처리: 환경 생성용 선분이 장애물을 통과하는지 검사
def line_free(start,end,course,margin=None):
    margin=course['planning_margin'] if margin is None else margin
    length=math.dist(start[:2],end[:2]);samples=max(2,int(length/.08)+1)
    return all(clearance(p,course)>=margin for p in np.linspace(start[:2],end[:2],samples))


# 처리: 환경 배치용 경로 생성; 실시간 기체 항법에 제공하지 않음
def astar(start,goal,course,choice=0):
    """8-connected graph; diagonal corner cutting disallowed by segment checks."""
    xmin,xmax,ymin,ymax=course['bounds'];resolution=course.get('grid_resolution',.25)
    nx=int(round((xmax-xmin)/resolution))+1;ny=int(round((ymax-ymin)/resolution))+1
    # 처리: 환경 생성용 격자 셀을 좌표로 변환
    def point(cell):return (xmin+cell[0]*resolution,ymin+cell[1]*resolution)
    # 처리: 환경 생성용 좌표를 격자 셀로 변환
    def cell(p):return (int(round((p[0]-xmin)/resolution)),int(round((p[1]-ymin)/resolution)))
    start_cell=cell(start);goal_cell=cell(goal)
    free={}
    # 처리: 환경 배치용 격자 셀의 여유 검사
    def allowed(c):
        if c not in free:free[c]=(0<=c[0]<nx and 0<=c[1]<ny and clearance(point(c),course)>=course['planning_margin'])
        return free[c]
    # 처리: 환경 생성용 가까운 유효 셀 탐색
    def nearest_free(p,c):
        if allowed(c):return c
        candidates=[(c[0]+dx,c[1]+dy) for dx in range(-2,3) for dy in range(-2,3) if allowed((c[0]+dx,c[1]+dy))]
        candidates=[other for other in candidates if line_free(p,point(other),course,max(.15,min(course['planning_margin'],clearance(p,course)-.01)))]
        if not candidates:raise ValueError('Start/goal violates map clearance')
        return min(candidates,key=lambda other:math.dist(p[:2],point(other)))
    start_cell=nearest_free(start,start_cell);goal_cell=nearest_free(goal,goal_cell)
    queue=[(0.,0.,start_cell)];cost={start_cell:0.};previous={};visited=set()
    while queue:
        _,g,current=heapq.heappop(queue)
        if current in visited:continue
        visited.add(current)
        if current==goal_cell:
            path=[current]
            while path[-1]!=start_cell:path.append(previous[path[-1]])
            path=[point(c) for c in path[::-1]];path[0]=tuple(start[:2]);path[-1]=tuple(goal[:2])
            smooth=[path[0]];index=0
            while index<len(path)-1:
                far=index+1
                for j in range(index+2,len(path)):
                    if line_free(path[index],path[j],course):far=j
                    else:break
                smooth.append(path[far]);index=far
            if not all(line_free(a,b,course) for a,b in zip(smooth,smooth[1:])):raise ValueError('Unsafe smoothed path')
            return smooth
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            other=(current[0]+dx,current[1]+dy)
            if not allowed(other):continue
            if not line_free(point(current),point(other),course):continue
            if dx and dy and (not allowed((current[0]+dx,current[1])) or not allowed((current[0],current[1]+dy))):continue
            x,y=point(other);distance=clearance((x,y),course)
            # Route preference is soft: the wall/obstacle mask always wins.
            bias=(.16*(ymax-y)/(ymax-ymin) if choice==1 else .16*(y-ymin)/(ymax-ymin) if choice==2 else 0.)
            if choice in (1,2):
                for obstacle in course['obstacle_rectangles']:
                    if abs(x-obstacle['x'])<1.25 and ((choice==1 and y<obstacle['y']) or (choice==2 and y>obstacle['y'])):
                        bias+=1.2
            risk=(.22 if choice==3 else .025)/max(.10,distance)
            candidate=g+math.hypot(dx,dy)*resolution*(1+bias+risk)
            if candidate<cost.get(other,float('inf')):
                cost[other]=candidate;previous[other]=current
                heapq.heappush(queue,(candidate+math.dist(point(other),point(goal_cell)),candidate,other))
    raise ValueError('No route inside boundary walls')


# 처리: 점들을 잇는 경로 길이 계산
def path_length(path):return sum(math.dist(a,b) for a,b in zip(path,path[1:]))


# 처리: 경로상 앞쪽 목표점 선택
def lookahead(position,path,distance=.55):
    if not path:return None
    # Use first still-unreached segment, retaining corners before turning.
    while len(path)>1 and math.dist(position[:2],path[0])<.18:path.pop(0)
    target=np.asarray(path[0],float);delta=target-np.asarray(position[:2],float);length=np.linalg.norm(delta)
    return target if length<=distance else np.asarray(position[:2])+delta/length*distance
