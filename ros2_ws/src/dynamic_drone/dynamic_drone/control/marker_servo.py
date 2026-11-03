# 파일 역할: 마커 영상 관측 자료형과 영상 기반 제어 보조
"""Size-free image servo for a cleared simulation trial, not metric localization.

The only observations are decoded marker corners and capture time. Neither a
marker size, fixture pose, camera extrinsic nor an assumed depth is accepted.
"""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
from dataclasses import dataclass
import math

import cv2
import numpy as np


# 파트: 마커의 영상 중심 오차·면적·모서리 자료
@dataclass
class ImageTarget:
    stamp: float
    center_error: np.ndarray
    area_fraction: float
    corners: np.ndarray


# 파트: 알려지지 않은 크기의 마커 영상 관측 보조
class UnknownMarkerObserver:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, marker_id, image_size=(320, 320)):
        if not 0 <= marker_id < 50:
            raise ValueError('DICT_4X4_50 marker ID required')
        self.marker_id = marker_id
        self.width, self.height = image_size
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
        self.latest = None
        self.last_stamp = None
        self.previous = None
        self.count = 0
        self.reason = 'waiting_for_marker_image'

    # 처리: 현재 관측을 무효 처리하고 원인을 기록
    def reject(self, reason):
        self.latest = None
        self.count = 0
        self.reason = reason
        return None

    # 처리: 새 영상의 형식·시간을 검사하고 인식 결과 갱신
    def process(self, gray, stamp):
        if gray.shape != (self.height, self.width) or gray.dtype != np.uint8:
            return self.reject('unexpected_image_format')
        corners, ids, _ = self.detector.detectMarkers(gray)
        indices = [] if ids is None else np.flatnonzero(ids.ravel() == self.marker_id)
        return self.observe(corners[int(indices[0])] if len(indices) == 1 else None, stamp)

    # 처리: 현재 영상 관측을 제어기 상태에 연결
    def observe(self, corners, stamp):
        if not math.isfinite(stamp) or (self.last_stamp is not None and stamp <= self.last_stamp):
            return self.reject('nonmonotonic_camera_timestamp')
        if self.last_stamp is not None and stamp-self.last_stamp > .25:
            self.count = 0
        self.last_stamp = stamp
        if corners is None:
            return self.reject('marker_missing_or_duplicated')
        points = np.asarray(corners, dtype=float).reshape(4, 2)
        if (not np.isfinite(points).all() or np.min(points) < 3
                or np.max(points[:, 0]) > self.width-4 or np.max(points[:, 1]) > self.height-4):
            return self.reject('marker_clipped')
        sides = np.linalg.norm(points-np.roll(points, 1, axis=0), axis=1)
        contour = points.astype(np.float32)
        if min(sides) < 16 or min(sides)/max(sides) < .55 or not cv2.isContourConvex(contour):
            return self.reject('marker_small_or_oblique')
        area = abs(float(cv2.contourArea(contour)))/(self.width*self.height)
        center = (points.mean(axis=0)-[self.width/2., self.height/2.])/[self.width/2., self.height/2.]
        if self.previous is not None:
            previous_stamp, previous_center, previous_area = self.previous
            if stamp-previous_stamp <= .25 and (np.linalg.norm(center-previous_center) > .25
                                               or not .65 < area/previous_area < 1.55):
                return self.reject('image_target_jump')
        self.previous = (stamp, center.copy(), area)
        self.count += 1
        if self.count < 5:
            self.latest = None
            self.reason = 'confirming_same_marker'
            return None
        self.latest = ImageTarget(stamp, center, area, points)
        self.reason = 'image_target_valid_no_metric_pose'
        return self.latest

    # 처리: 추정 상태·유효 여부·근거를 기록용 자료로 반환
    def telemetry(self):
        target = self.latest
        return dict(valid=target is not None, reason=self.reason, marker_id=self.marker_id,
                    source='front_image_only', marker_size_m=None, position_m=None,
                    ground_range_m=None, marker_distance_m=None, map_used=False,
                    center_error=None if target is None else target.center_error.tolist(),
                    area_fraction=None if target is None else target.area_fraction)


# 파트: 영상 중심과 면적 기반 제어 보조
class UnknownMarkerFlight:
    """Align vertically/laterally, approach by image area, hold then end trial.

    Metric command integrals below are exposure budgets, NOT estimated motion
    or altitude. This controller cannot establish a physical altitude limit.
    """
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, target_area=.06):
        if not math.isfinite(target_area) or not .04 <= target_area <= .08:
            raise ValueError('Target marker image area must be .04..08')
        self.target_area = target_area
        self.phase = 'ACQUIRE'
        self.entered = None
        self.started = None
        self.stable_since = None
        self.last_time = None
        self.previous_command = np.zeros(4)
        self.upward_command_integral = 0.
        self.downward_command_integral = 0.
        self.forward_command_integral = 0.
        self.lateral_command_integral = 0.

    # 처리: 현재 관측 오차에 맞춘 접근 명령 계산
    def approach(self, target, now):
        ready = target.area_fraction >= self.target_area*.94
        return ready, min(.18, .9*max(0., 1.-target.area_fraction/self.target_area))

    # 처리: 현재 단계와 유효 관측으로 이동 명령 결정
    def command(self, target, now):
        if not math.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            self.previous_command[:] = 0
            return np.zeros(4), 'invalid_control_time'
        elapsed = 0. if self.last_time is None else now-self.last_time
        self.last_time = now
        # Account for the actual previously held command before stopping it.
        self.upward_command_integral += max(0., self.previous_command[2])*elapsed
        self.downward_command_integral += max(0., -self.previous_command[2])*elapsed
        self.forward_command_integral += max(0., self.previous_command[0])*elapsed
        self.lateral_command_integral += abs(self.previous_command[1])*elapsed
        dt = min(max(elapsed, .001), .1)
        if target is None or not 0 <= now-target.stamp <= .25:
            self.stable_since = None
            self.previous_command[:] = 0
            return np.zeros(4), 'image_unavailable_stop'
        if self.phase == 'ACQUIRE':
            self.started = self.entered = now
            self.phase = 'ALIGN'
        if (target.area_fraction > .12 or self.upward_command_integral > .95
                or self.downward_command_integral > .35 or self.forward_command_integral > 3.
                or self.lateral_command_integral > 1.5 or now-self.started > 75):
            self.phase = 'ABORT'
        if self.phase in ('ABORT', 'COMPLETE'):
            self.previous_command[:] = 0
            return np.zeros(4), 'image_or_command_budget_stop'
        ex, ey = target.center_error
        centered = abs(ex) < .07 and abs(ey) < .07
        area_ready, approach_speed = self.approach(target, now)
        ready = centered and (self.phase == 'ALIGN' or area_ready)
        if ready:
            if self.stable_since is None:
                self.stable_since = now
            dwell = 3. if self.phase == 'HOLD' else .7
            if now-self.stable_since >= dwell:
                self.phase = {'ALIGN': 'APPROACH', 'APPROACH': 'HOLD', 'HOLD': 'COMPLETE'}[self.phase]
                self.entered = now
                self.stable_since = None
        else:
            self.stable_since = None
        if now-self.entered > 35:
            self.phase = 'ABORT'
            self.previous_command[:] = 0
            return np.zeros(4), 'phase_timeout'
        # Body +y is image-left and body +z is image-up. No PnP or depth.
        lateral = np.clip(-.55*ex, -.16, .16) if abs(ex) > .025 else 0.
        vertical = np.clip(-.55*ey, -.08, .15) if abs(ey) > .025 else 0.
        # Do not descend from the initial ground-supported acquisition pose.
        if vertical < 0 and self.upward_command_integral < .2:
            vertical = 0.
        forward = 0.
        if self.phase == 'APPROACH' and centered and not area_ready:
            forward = approach_speed
        command = np.array([forward, lateral, vertical, 0.])
        limit = np.array([.3, .35, .3, .4])*dt
        command = self.previous_command+np.clip(command-self.previous_command, -limit, limit)
        # Forward inhibition must take effect immediately after alignment is lost.
        if not centered or self.phase in ('ALIGN', 'HOLD', 'COMPLETE') or area_ready:
            command[0] = 0.
        if self.phase == 'COMPLETE':
            command[:] = 0.
        self.previous_command = command
        return command, self.phase.lower()

    # 처리: 추정 상태·유효 여부·근거를 기록용 자료로 반환
    def telemetry(self):
        return dict(target_area_fraction=self.target_area,
                    upward_command_integral_m=self.upward_command_integral,
                    downward_command_integral_m=self.downward_command_integral,
                    forward_command_integral_m=self.forward_command_integral,
                    command_integrals_are_position_estimates=False,
                    physical_height_m=None, physical_standoff_m=None)
