# 파일 역할: 실제로 인식한 마커 영상과 증빙 기록 저장
"""Append-only marker evidence for offline simulation preparation."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import datetime,hashlib,json,time,uuid
from pathlib import Path
import cv2

# 파트: 실제 마커 인식 증빙 저장
class MarkerEvidence:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self,directory,dictionary,*,source='Gazebo_static_front_camera_preview',action='none_preview_only'):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.dictionary=dictionary;self.saved={};self.total=0
        self.source=source;self.action=action
    # 처리: 새 관측 영상과 실제 인식 결과 저장
    def record(self,gray,overlay,corners,ids,capture_seconds):
        now=time.monotonic()
        for box,identifier in zip(corners,[] if ids is None else ids.ravel()):
            identifier=int(identifier)
            if now-self.saved.get(identifier,-1e9)<3.:continue
            points=box.reshape(4,2);x0,y0=points.min(axis=0);x1,y1=points.max(axis=0)
            crop=gray[max(0,int(y0)):min(gray.shape[0],int(y1)+1),max(0,int(x0)):min(gray.shape[1],int(x1)+1)]
            if not crop.size:continue
            key=uuid.uuid4().hex;paths={name:self.directory/f'{key}_{name}.png' for name in ('raw','recognized','marker')}
            for name,image in (('raw',gray),('recognized',overlay),('marker',crop)):
                if not cv2.imwrite(str(paths[name]),image):raise OSError('Could not save '+str(paths[name]))
            row=dict(event_id=key,recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                capture_simulation_seconds=capture_seconds,marker_id=identifier,dictionary=self.dictionary,
                corners_px=points.tolist(),confidence=None,accuracy=None,
                detector='OpenCV_ArUco',source=self.source,onboard_execution=False,
                action=self.action,images={k:p.name for k,p in paths.items()},
                raw_sha256=hashlib.sha256(paths['raw'].read_bytes()).hexdigest())
            with (self.directory/'events.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            self.saved[identifier]=now;self.total+=1
