# 파일 역할: ROS 영상 메시지 해석과 흑백 변환
"""OpenCV panel proposals and supervised arrow classification; no flight commands."""


# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import cv2
import numpy as np



# 파트: 공통 경로·좌표계·설정값
LABELS = ("none", "left", "right", "up", "down")
FEATURE_VERSION = "hog64_gray_v1"


# 처리: 유효한 영상만 받아 흑백으로 변환
def gray_image(image):
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty uint8 image")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected gray or BGR image")
















# 처리: ROS 영상 인코딩·행 간격에 맞춰 픽셀 배열 복원
def decode_image(message):
    """Decode ROS Image with row padding; no cv_bridge/NumPy ABI dependency."""
    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
    if message.encoding not in channels:
        raise ValueError(f"Unsupported encoding: {message.encoding}")
    count = channels[message.encoding]
    if message.width <= 0 or message.height <= 0 or message.step < message.width * count:
        raise ValueError("Invalid image dimensions/step")
    if len(message.data) != message.height * message.step:
        raise ValueError("Invalid image byte count")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    pixels = rows[:, :message.width * count].copy()
    if count == 1:
        return pixels.reshape(message.height, message.width)
    pixels = pixels.reshape(message.height, message.width, count)
    conversions = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR,
                   "bgra8": cv2.COLOR_BGRA2BGR}
    return cv2.cvtColor(pixels, conversions[message.encoding]) if message.encoding in conversions else pixels

