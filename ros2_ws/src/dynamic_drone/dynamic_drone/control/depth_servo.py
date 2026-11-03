# 파일 역할: 깊이 ROI 기반 제어 보조; 현재 1단계 위치 보완에는 사용하지 않음
"""Learned depth for the size-free marker trial. No simulator truth input."""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import math
import cv2
import numpy as np
from dynamic_drone.control.marker_servo import UnknownMarkerFlight


# 파트: 마커 영상 영역의 추정 깊이 유효성 검사
class MarkerDepthEstimate:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self):
        self.latest = None
        self.previous = None
        self.last_input = None
        self.count = 0
        self.reason = 'waiting_for_predicted_depth'

    # 처리: 현재 관측을 무효 처리하고 원인을 기록
    def reject(self, reason):
        self.latest = None
        self.count = 0
        self.reason = reason

    # 처리: 새 입력의 시간·품질을 확인해 추정 상태 갱신
    def update(self, depth, target, image_shape):
        self.latest = None
        if target is None:
            self.reject('marker_unavailable_for_depth_roi')
            return
        stamp = target.stamp
        if not math.isfinite(stamp) or (self.last_input is not None and stamp <= self.last_input):
            self.reject('nonmonotonic_depth_frame')
            return
        if self.last_input is not None and stamp-self.last_input > .25:
            self.count = 0
        self.last_input = stamp
        depth = np.asarray(depth, dtype=float)
        if depth.ndim != 2 or min(depth.shape) < 16:
            self.reject('invalid_depth_shape')
            return
        h, w = depth.shape
        points = target.corners
        # Erode the polygon geometrically to avoid including background pixels.
        inner = points.mean(axis=0)+.65*(points-points.mean(axis=0))
        inner *= [w/image_shape[1], h/image_shape[0]]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, np.rint(inner).astype(np.int32), 1)
        valid = np.isfinite(depth) & (depth >= .15) & (depth < 19.5)
        values = depth[(mask > 0) & valid]
        if values.size < 8 or values.size < .85*np.count_nonzero(mask):
            self.reject('marker_depth_invalid_or_saturated')
            return
        distance = float(np.median(values))
        spread = float(np.percentile(values, 75)-np.percentile(values, 25))
        if spread > max(.7, distance*.4):
            self.reject('marker_depth_spatially_inconsistent')
            return
        band = depth[int(.35*h):int(.65*h), int(.35*w):int(.65*w)]
        usable = band[np.isfinite(band) & (band >= .15) & (band < 19.5)]
        if usable.size < band.size*.8:
            self.reject('front_depth_invalid_or_saturated')
            return
        clearance = float(np.percentile(usable, 20))
        if self.previous is not None:
            old_stamp, old_distance = self.previous
            if stamp-old_stamp <= .25 and abs(distance-old_distance) > max(.7, 4.*(stamp-old_stamp)):
                self.reject('marker_depth_temporal_jump')
                return
        self.previous = (stamp, distance)
        self.count += 1
        if self.count < 3:
            self.reason = 'confirming_predicted_depth'
            return
        self.latest = dict(stamp=stamp, estimated_marker_depth_m=distance,
                           estimated_front_clearance_m=clearance, roi_iqr_m=spread,
                           roi_valid_pixels=int(values.size))
        self.reason = 'predicted_depth_valid_not_ground_truth'

    # 처리: 추정 상태·유효 여부·근거를 기록용 자료로 반환
    def telemetry(self):
        return dict(valid=self.latest is not None, reason=self.reason,
                    source='learned_monocular_depth', **(self.latest or {}),
                    measured_accuracy_m=None, ground_range_m=None)


# 파트: 깊이 ROI 기반 제어 보조
class DepthMarkerFlight(UnknownMarkerFlight):
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, stop_distance=2.):
        super().__init__()
        if not math.isfinite(stop_distance) or not 1.5 <= stop_distance <= 2.5:
            raise ValueError('Predicted stop distance must be 1.5..2.5m')
        self.stop_distance = stop_distance
        self.depth = MarkerDepthEstimate()

    # 처리: 현재 관측 오차에 맞춘 접근 명령 계산
    def approach(self, target, now):
        distance = self.depth.latest['estimated_marker_depth_m']
        return distance <= self.stop_distance+.15, min(.18, .35*max(0., distance-self.stop_distance))

    # 처리: 현재 단계와 유효 관측으로 이동 명령 결정
    def command(self, target, now):
        depth = self.depth.latest
        if (target is None or depth is None or depth['stamp'] != target.stamp
                or not 0 <= now-depth['stamp'] <= .25):
            command, _ = super().command(None, now)
            return command, 'depth_unavailable_stop'
        if (depth['estimated_marker_depth_m'] < self.stop_distance-.35
                or (self.phase in ('APPROACH', 'HOLD') and depth['estimated_front_clearance_m'] < .8)):
            # Do not report a successful standoff when already too close.
            super().command(None, now)
            self.phase = 'ABORT'
            return np.zeros(4), 'predicted_depth_too_close'
        if self.phase == 'HOLD' and depth['estimated_marker_depth_m'] > self.stop_distance+.3:
            self.phase = 'APPROACH'
            self.entered = now
            self.stable_since = None
        return super().command(target, now)

    # 처리: 추정 상태·유효 여부·근거를 기록용 자료로 반환
    def telemetry(self):
        record = super().telemetry()
        record['target_area_fraction'] = None
        record['predicted_stop_distance_m'] = self.stop_distance
        record['depth'] = self.depth.telemetry()
        return record
