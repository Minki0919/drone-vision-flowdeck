# 파일 역할: 흑백 영상 입력으로 학습된 깊이 모델 추론
"""Small grayscale encoder-decoder, inspired by the paper, not its published model.

Ground-truth metric depth is used for supervised training/evaluation only.
The deployed predictor accepts ONE grayscale image, never simulator depth.
"""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import hashlib
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# 파트: 공통 경로·좌표계·설정값
SIZE=128
SCHEMA='paper_gray_metric_depth_v2'
INPUT_HW=(244,324)
OUTPUT_HW=(62,82)
LABEL_POLICY='gazebo_far_clip_20m_v1'




# 처리: 깊이별·점별 합성곱 신경망 블록 구성
def block(a,b,stride=1):
    return nn.Sequential(nn.Conv2d(a,a,3,stride,1,groups=a),nn.ReLU(),nn.Conv2d(a,b,1),nn.ReLU())




# 파트: 현재 깊이 모델의 인코더·디코더 구조
class PaperDepth(nn.Module):
    """Five DW/PW encoder and decoder stages with nearest-neighbor skips.

    A documented adaptation: the PDF does not specify every stride/channel;
    weights are fresh, metric Gazebo depth replaces its indoor depth labels.
    """
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self):
        super().__init__()
        channels=[128,192,256,384,512]
        self.stem=nn.Sequential(nn.Conv2d(1,128,3,2,1),nn.ReLU(),block(128,128))
        self.encoder=nn.ModuleList([block(a,b,2) for a,b in zip(channels,channels[1:])])
        self.decoder=nn.ModuleList([block(a+b,b) for a,b in zip(channels[:0:-1],channels[-2::-1])])
        self.final=block(128,128);self.head=nn.Conv2d(128,1,1)
    # 처리: 인코더·디코더로 입력 영상의 깊이 지도 계산
    def forward(self,x):
        features=[self.stem(x)]
        for layer in self.encoder:features.append(layer(features[-1]))
        out=features[-1]
        for layer,skip in zip(self.decoder,features[-2::-1]):
            out=layer(torch.cat([F.interpolate(out,size=skip.shape[-2:],mode='nearest'),skip],1))
        out=self.final(F.interpolate(out,size=OUTPUT_HW,mode='nearest'))
        return .05+19.95*torch.sigmoid(self.head(out))


# 처리: 요청한 CPU/GPU의 실제 사용 가능 여부 확인
def device_for(name):
    device='cuda' if name=='auto' and torch.cuda.is_available() else ('cpu' if name=='auto' else name)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    return device








# 파트: 저장된 깊이 모델 로딩과 흑백 영상 추론
class Predictor:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self,path,device='cpu'):
        self.path=Path(path);self.device=device_for(device)
        # 모델 로딩: 저장 형식 확인 후 가중치 연결
        saved=torch.load(self.path,map_location='cpu',weights_only=True)
        if saved.get('schema')!=SCHEMA: raise ValueError('Incompatible depth schema')
        self.model=PaperDepth().to(self.device);self.model.load_state_dict(saved['state_dict']);self.model.eval()
        self.hash=hashlib.sha256(self.path.read_bytes()).hexdigest();self.latencies=[]
        self.predict(np.zeros((SIZE,SIZE),np.uint8));self.latencies.clear() # warmup excluded
    # 처리: 새 흑백 영상으로 추론하고 처리 시간·관측 시점 기록
    def predict(self,gray):
        gray=cv2.resize(gray,(INPUT_HW[1],INPUT_HW[0]),interpolation=cv2.INTER_AREA)
        if gray.ndim!=2: raise ValueError('Depth input must be grayscale only')
        x=torch.from_numpy(gray.copy()).float()[None,None].to(self.device)/255
        if self.device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        # 깊이 추론: 현재 흑백 영상만 입력; Gazebo 정답 깊이 사용 안 함
        with torch.inference_mode():y=self.model(x)
        if self.device=='cuda':torch.cuda.synchronize()
        ms=(time.perf_counter()-start)*1000
        self.latencies.append(ms)
        return y[0,0].cpu().numpy(),ms




