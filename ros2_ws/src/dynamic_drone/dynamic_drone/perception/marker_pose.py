# 파일 역할: 알려진 마커 크기와 카메라 보정값으로 상대 자세 계산
"""Metric relative pose from ONE stationary, known-size front-camera marker.

No scene coordinates, IMU integration, depth, or downward sensors. The origin
is the body pose observed before takeoff; z means displacement, NOT ground range.
This Python reference implementation is not GAP8 firmware.
"""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
from dataclasses import dataclass
import math

import cv2
import numpy as np


# Optical right/down/forward <- aircraft forward/left/up.
# 파트: 공통 경로·좌표계·설정값
CAMERA_FROM_BODY = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


# 처리: 두 회전행렬 사이의 각도 차이 계산
def rotation_distance(a, b):
    return float(np.arccos(np.clip((np.trace(a.T @ b)-1.)/2., -1., 1.)))


def square_pose_candidates(object_points, pixels, matrix, distortion):
    """Solve both planar branches plus a homography-initialized PnP fit.

    IPPE alone can return a numerically broken Rodrigues vector near a
    fronto-parallel tag. A separate iterative initialization recovers that
    solution without borrowing a pose from the simulator or previous frame.
    Every candidate must still explain the measured corners and face forward.
    """
    objects = np.asarray(object_points, dtype=np.float64).reshape(4, 3)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(pixels).all():
        return []
    seeds = []
    for method in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE):
        try:
            solved = cv2.solvePnPGeneric(objects, pixels, matrix, distortion, flags=method)
            seeds.extend(zip(solved[1], solved[2]))
        except cv2.error:
            continue
    candidates = []
    for rvec, tvec in seeds:
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            continue
        try:
            rvec, tvec = cv2.solvePnPRefineLM(objects, pixels, matrix, distortion,
                                             rvec.copy(), tvec.copy())
            rotation = cv2.Rodrigues(rvec)[0]
            translation = tvec.reshape(3)
            points = (rotation @ objects.T).T+translation
            projected = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)[0].reshape(4, 2)
            error = float(np.sqrt(np.mean(np.sum((projected-pixels)**2, axis=1))))
            if (not np.isfinite(points).all() or points[:, 2].min() <= .15
                    or not math.isfinite(error) or error > 1.5
                    or np.linalg.norm(translation) > 18.):
                continue
            candidates.append(dict(rotation_camera=rotation, translation_camera=translation,
                                   reprojection_px=error))
        except cv2.error:
            continue
    unique = []
    for row in sorted(candidates, key=lambda row: row['reprojection_px']):
        if any(rotation_distance(row['rotation_camera'], old['rotation_camera']) < 1e-3
               and np.linalg.norm(row['translation_camera']-old['translation_camera']) < 1e-4
               for old in unique):
            continue
        unique.append(row)
    return unique


# 파트: 마커로 계산한 위치·회전·오차 자료
@dataclass
class MarkerPose:
    stamp: float
    position: np.ndarray
    rotation: np.ndarray
    marker_body: np.ndarray
    reprojection_px: float
    corners: np.ndarray


# 파트: 고정 마커 관측으로 상대 자세 계산
class FrontMarkerLocalizer:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, camera_matrix, distortion, image_size, marker_id, marker_size_m,
                 camera_offset_body, *, required_frames=8, max_gap=.3):
        self.k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        self.distortion = np.asarray(distortion, dtype=float).reshape(-1)
        self.offset = np.asarray(camera_offset_body, dtype=float).reshape(3)
        self.width, self.height = map(int, image_size)
        if (not np.isfinite(self.k).all() or not np.isfinite(self.distortion).all()
                or not np.isfinite(self.offset).all() or min(self.width, self.height) < 32
                or self.k[0, 0] <= 0 or self.k[1, 1] <= 0
                or not math.isfinite(marker_size_m) or marker_size_m <= 0
                or not 0 <= marker_id < 50 or required_frames < 3 or max_gap <= 0):
            raise ValueError('Valid calibrated camera, fixed marker ID and metric size required')
        half = marker_size_m/2.
        # ArUco's decoded corner order, NOT a screen-relative corner reordering.
        self.object_points = np.array([[-half, half, 0], [half, half, 0],
                                       [half, -half, 0], [-half, -half, 0]], dtype=float)
        self.marker_id = int(marker_id)
        self.required_frames = required_frames
        self.max_gap = max_gap
        from dynamic_drone.perception.aruco_detection import ArucoImageDetector
        self.detector = ArucoImageDetector()
        self.reference = None
        self.bootstrap = []
        self.previous = None
        self.last_input = None
        self.latest = None
        self.reason = 'waiting_for_fixed_marker'
        self.consecutive = 0

    # 처리: 현재 관측을 무효 처리하고 원인을 기록
    def reject(self, reason):
        self.latest = None
        self.reason = reason
        self.consecutive = 0
        self.bootstrap.clear()
        return None

    # 처리: 새 영상의 형식·시간을 검사하고 인식 결과 갱신
    def process(self, gray, stamp):
        if gray.shape != (self.height, self.width) or gray.dtype != np.uint8:
            return self.reject('camera_calibration_size_mismatch')
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return self.observe(None, stamp)
        indices = np.flatnonzero(ids.reshape(-1) == self.marker_id)
        if len(indices) != 1:
            return self.observe(None, stamp, 'fixed_marker_missing_or_duplicated')
        return self.observe(corners[int(indices[0])], stamp)

    # 처리: 현재 영상 관측을 제어기 상태에 연결
    def observe(self, corners, stamp, missing_reason='fixed_marker_not_visible'):
        if not math.isfinite(stamp) or (self.last_input is not None and stamp <= self.last_input):
            return self.reject('nonmonotonic_camera_timestamp')
        if self.last_input is not None and stamp-self.last_input > self.max_gap:
            self.consecutive = 0
            self.bootstrap.clear()
        self.last_input = stamp
        if corners is None:
            return self.reject(missing_reason)
        pixels = np.asarray(corners, dtype=float).reshape(4, 2)
        if (not np.isfinite(pixels).all() or np.min(pixels) < 3
                or np.max(pixels[:, 0]) > self.width-4
                or np.max(pixels[:, 1]) > self.height-4
                or min(np.linalg.norm(pixels-np.roll(pixels, 1, axis=0), axis=1)) < 16):
            return self.reject('marker_too_small_or_clipped')
        candidates = []
        for row in square_pose_candidates(self.object_points, pixels, self.k, self.distortion):
            r_cm, t_cm = row['rotation_camera'], row['translation_camera']
            error = row['reprojection_px']
            r_mb = r_cm.T @ CAMERA_FROM_BODY
            p_mb = -r_cm.T @ t_cm-r_mb @ self.offset
            if error <= 1.5:
                candidates.append((error, p_mb, r_mb, CAMERA_FROM_BODY.T @ t_cm+self.offset))
        if not candidates:
            return self.reject('pose_reprojection_failed')
        candidates.sort(key=lambda item: item[0])
        best = candidates[0]
        if len(candidates) > 1 and candidates[1][0]-best[0] < .15:
            if rotation_distance(best[2], candidates[1][2]) > math.radians(15):
                return self.reject('ambiguous_planar_pose')
        error, position, rotation, marker_body = best
        if self.previous is not None:
            old_stamp, old_position, old_rotation = self.previous
            dt = stamp-old_stamp
            if dt <= self.max_gap:
                if (np.linalg.norm(position-old_position) > .06+1.5*dt
                        or rotation_distance(rotation, old_rotation) > .08+2.*dt):
                    return self.reject('pose_jump_rejected')
        self.previous = (stamp, position.copy(), rotation.copy())
        self.consecutive += 1
        if self.reference is None:
            self.bootstrap.append((position, rotation))
            if len(self.bootstrap) < self.required_frames:
                self.reason = 'initializing_stationary_origin'
                return None
            first_position, first_rotation = self.bootstrap[0]
            if any(np.linalg.norm(p-first_position) > .08 or
                   rotation_distance(r, first_rotation) > math.radians(5)
                   for p, r in self.bootstrap):
                return self.reject('origin_not_stable')
            self.reference = (np.mean([p for p, _ in self.bootstrap], axis=0), rotation.copy())
            self.bootstrap.clear()
        if self.consecutive < 3:
            self.reason = 'reconfirming_fixed_marker'
            return None
        p_origin, r_origin = self.reference
        self.latest = MarkerPose(stamp, r_origin.T @ (position-p_origin), r_origin.T @ rotation,
                                 marker_body, error, pixels)
        self.reason = 'fixed_marker_pose_valid'
        return self.latest

    # 처리: 추정 상태·유효 여부·근거를 기록용 자료로 반환
    def telemetry(self):
        pose = self.latest
        return dict(valid=pose is not None, reason=self.reason, marker_id=self.marker_id,
                    source='front_camera_fixed_marker_pnp', frame='initial_body_pose',
                    position_m=None if pose is None else pose.position.tolist(),
                    marker_body_m=None if pose is None else pose.marker_body.tolist(),
                    reprojection_px=None if pose is None else pose.reprojection_px,
                    ground_range_m=None, absolute_world_position=None,
                    measured_accuracy_m=None, map_used=False)


# 파트: 독립 마커 비행 제어 보조
class MarkerFlightTrial:
    """Isolated simulation trial: acquire -> rise -> approach -> visual hold.

    Requires a stationary marker visible throughout a cleared test volume.
    No truck landing, obstacle course, or stale-pose dead reckoning is implied.
    """
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, rise_m=.65, stand_off_m=2.0):
        if not .25 <= rise_m <= 1.0 or not 1.5 <= stand_off_m <= 3.:
            raise ValueError('Trial rise must be .25..1m; stand-off 1.5..3m')
        self.rise = rise_m
        self.stand_off = stand_off_m
        self.phase = 'ACQUIRE'
        self.goal = np.array([0., 0., rise_m])
        self.entered = None
        self.stable_since = None
        self.last_time = None
        self.previous_command = np.zeros(4)

    # 처리: 현재 단계와 유효 관측으로 이동 명령 결정
    def command(self, pose, now):
        if not math.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            self.previous_command[:] = 0
            return np.zeros(4), 'invalid_control_time'
        dt = .05 if self.last_time is None else min(now-self.last_time, .1)
        self.last_time = now
        if pose is None or not 0 <= now-pose.stamp <= .25:
            self.stable_since = None
            self.previous_command[:] = 0
            return np.zeros(4), 'vision_unavailable_stop'
        if self.phase == 'ACQUIRE':
            self.phase = 'TAKEOFF'
            self.entered = now
        yaw = math.atan2(pose.rotation[1, 0], pose.rotation[0, 0])
        if (pose.position[2] > self.rise+.25 or pose.position[2] < -.15
                or abs(pose.position[1]) > .7 or pose.marker_body[0] < 1.1
                or rotation_distance(np.eye(3), pose.rotation) > math.radians(25)):
            self.phase = 'ABORT'
        if self.phase == 'ABORT':
            self.previous_command[:] = 0
            return np.zeros(4), 'visual_trial_bounds_exceeded'
        if self.phase == 'TAKEOFF':
            error = self.goal-pose.position
        else:
            # Current observed target, not a marker position taken from the world.
            marker_origin = pose.position+pose.rotation @ pose.marker_body
            self.goal = np.array([marker_origin[0]-self.stand_off,
                                  marker_origin[1], self.rise])
            error = self.goal-pose.position
        aligned = np.linalg.norm(error) < .08 and abs(yaw) < .06
        if aligned:
            if self.stable_since is None:
                self.stable_since = now
            if now-self.stable_since >= (3. if self.phase == 'HOLD' else 1.):
                transitions = {'TAKEOFF': 'APPROACH', 'APPROACH': 'HOLD', 'HOLD': 'COMPLETE'}
                self.phase = transitions.get(self.phase, self.phase)
                self.entered = now
                self.stable_since = None
        else:
            self.stable_since = None
        if now-self.entered > 45:
            self.phase = 'ABORT'
            self.previous_command[:] = 0
            return np.zeros(4), 'phase_timeout'
        velocity_origin = np.clip(.65*error, [-.25, -.2, -.12], [.25, .2, .18])
        command = np.r_[pose.rotation.T @ velocity_origin, np.clip(-.7*yaw, -.2, .2)]
        # Acceleration bound is bypassed on loss/abort so stale motion stops now.
        limit = np.array([.4, .4, .3, .5])*dt
        command = self.previous_command+np.clip(command-self.previous_command, -limit, limit)
        self.previous_command = command
        return command, self.phase.lower()
