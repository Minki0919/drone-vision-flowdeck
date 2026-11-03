# 파일 역할: 환경 렌더링용 바닥 텍스처 메시 생성
"""Batched, nonsemantic corner texture for the full simulator road."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
from pathlib import Path
import os

# 처리: 바닥 시각 패턴의 메시 생성 또는 기존 파일 사용
def floor_mesh(root):
    path=Path(root)/'assets/flow_ground_150m_v2.dae';path.parent.mkdir(exist_ok=True)
    if path.exists():return path
    positions=[];indices=[]
    for i in range(256):
        for j in range(13):
            x=i*.6-.3;y=j*.6-3.6;n=len(positions)//3
            positions.extend([x-.06,y-.05,0.,x+.06,y-.05,0.,x+.06,y+.05,0.,x-.06,y+.05,0.])
            indices.extend([n,0,n+1,0,n+2,0,n,0,n+2,0,n+3,0])
    data=f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit meter="1" name="meter"/><up_axis>Z_UP</up_axis></asset>
<library_effects><effect id="ink"><profile_COMMON><technique sid="common"><lambert><ambient><color>.10 .10 .10 1</color></ambient><diffuse><color>.10 .10 .10 1</color></diffuse></lambert></technique></profile_COMMON></effect></library_effects>
<library_materials><material id="mat"><instance_effect url="#ink"/></material></library_materials>
<library_geometries><geometry id="patches"><mesh>
<source id="pos"><float_array id="pa" count="{len(positions)}">{' '.join(map(str,positions))}</float_array><technique_common><accessor source="#pa" count="{len(positions)//3}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="normal"><float_array id="na" count="3">0 0 1</float_array><technique_common><accessor source="#na" count="1" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<vertices id="v"><input semantic="POSITION" source="#pos"/></vertices><triangles count="{len(indices)//6}" material="ink"><input semantic="VERTEX" source="#v" offset="0"/><input semantic="NORMAL" source="#normal" offset="1"/><p>{' '.join(map(str,indices))}</p></triangles>
</mesh></geometry></library_geometries><library_visual_scenes><visual_scene id="scene"><node id="floor"><instance_geometry url="#patches"><bind_material><technique_common><instance_material symbol="ink" target="#mat"/></technique_common></bind_material></instance_geometry></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#scene"/></scene></COLLADA>'''
    temporary=path.with_suffix(f'.{os.getpid()}.tmp');temporary.write_text(data);temporary.replace(path);return path
