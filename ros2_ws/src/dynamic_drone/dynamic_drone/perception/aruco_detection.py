# 파일 역할: 흑백 영상에서 ArUco ID와 정밀 모서리 검출
"""사용자가 제공한 ArUco_marker_detection.py의 영상 탐지 부분.

공통 ArUco 사전 + 모서리 정밀 보정 + 흑백 영상 입력을 사용합니다.
ROS 구독·표시·거리 추정은 기존 미션이 맡습니다. 원본의 고정 토픽,
해상도, ID별 크기를 가져오지 않아 CameraInfo 보정과 미션 ID를 유지합니다.
"""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import cv2
import numpy as np
from dynamic_drone.perception.aruco_dictionary import get_dictionary


# 파트: 공통 사전과 모서리 정밀 보정을 사용하는 탐지기
class ArucoImageDetector:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self):
        aruco = cv2.aruco
        self.dictionary = get_dictionary()
        self.parameters = (aruco.DetectorParameters() if hasattr(aruco, 'DetectorParameters')
                           else aruco.DetectorParameters_create())
        # Refine measured corners before metric PnP; IDs still come from ArUco.
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = (aruco.ArucoDetector(self.dictionary, self.parameters)
                         if hasattr(aruco, 'ArucoDetector') else None)

    # 처리: 흑백 변환 후 공통 사전으로 ID·모서리 검출
    def detectMarkers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError('ArUco input must be an 8-bit grayscale or BGR image')
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
