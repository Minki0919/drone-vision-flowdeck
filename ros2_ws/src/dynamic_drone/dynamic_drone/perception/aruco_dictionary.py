# 파일 역할: 마커 생성과 탐지에 사용하는 공통 ArUco 사전
"""Gazebo 마커 생성과 카메라 인식이 함께 사용하는 사전 설정."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import cv2

# 파트: 공통 경로·좌표계·설정값
DICTIONARY_NAME = 'DICT_4X4_50'


# 처리: 환경 생성과 탐지가 공유하는 ArUco 사전 반환
def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))
