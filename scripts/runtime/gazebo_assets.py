#!/usr/bin/env python3
# 파일 역할: 렌더러 선택과 이 PC에 맞는 메시 경로 연결
"""Relocate Gazebo assets and select this PC's renderer without editing policy sources."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

# 파트: 공통 경로·좌표계·설정값
MESH_SUFFIX='/vendor/crazyflie-simulation/meshes/'
GPU_KEYS=('DRI_PRIME','MESA_LOADER_DRIVER_OVERRIDE','LIBGL_ALWAYS_SOFTWARE',
          '__NV_PRIME_RENDER_OFFLOAD','__GLX_VENDOR_LIBRARY_NAME','__VK_LAYER_NV_optimus',
          '__EGL_VENDOR_LIBRARY_FILENAMES')


# 처리: GPU·소프트웨어 렌더러 환경 선택
def renderer_environment(env, mode='auto'):
    """Do not inherit the original computer's hard-coded AMD PCI address."""
    if mode not in ('auto','nvidia','mesa','software','inherit'):
        raise ValueError('GPU must be auto, nvidia, mesa, software or inherit')
    result=dict(env)
    for key in GPU_KEYS:
        result.pop(key,None)
    if mode=='inherit':
        for key in GPU_KEYS:
            if key in os.environ:
                result[key]=os.environ[key]
        return result
    if mode=='auto':
        mode='mesa'
        if shutil.which('nvidia-smi'):
            try:
                if subprocess.run(['nvidia-smi','-L'],stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,timeout=3).returncode==0:
                    mode='nvidia'
            except subprocess.TimeoutExpired:
                pass
    if mode=='nvidia':
        result.update(__NV_PRIME_RENDER_OFFLOAD='1',__GLX_VENDOR_LIBRARY_NAME='nvidia',
                      __VK_LAYER_NV_optimus='NVIDIA_only')
        vendor=Path('/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
        if vendor.is_file():
            result['__EGL_VENDOR_LIBRARY_FILENAMES']=str(vendor)
    elif mode=='software':
        result.update(LIBGL_ALWAYS_SOFTWARE='1',__GLX_VENDOR_LIBRARY_NAME='mesa')
    return result


# 처리: 메시 파일 경로를 현재 프로젝트 위치로 변경
def relocate_meshes(world, root):
    tree=ET.parse(world)
    for uri in tree.findall('.//uri'):
        if uri.text and MESH_SUFFIX in uri.text:
            suffix=uri.text.split(MESH_SUFFIX,1)[1]
            base=(Path(root)/'vendor/crazyflie-simulation/meshes').resolve()
            target=(base/suffix).resolve()
            if not target.is_relative_to(base) or not target.is_file():
                raise FileNotFoundError(f'Missing local drone mesh: {target}')
            uri.text=str(target)
    tree.write(world,encoding='utf-8',xml_declaration=True)




