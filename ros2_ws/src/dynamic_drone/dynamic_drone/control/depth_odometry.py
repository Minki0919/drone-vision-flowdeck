# 파일 역할: 영상 특징점과 추정 깊이로 이동량 계산; 링·차량 단계에서 사용
"""Frame-to-frame front visual odometry with learned depth scale.

Short-lived features only; no supplied map, world pose, ToF or IMU input.
This estimate can drift. Invalid tracking is never replaced by dead reckoning.
"""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import cv2
import numpy as np
from dynamic_drone.perception.marker_pose import CAMERA_FROM_BODY


# 파트: 전방 영상 특징점과 추정 깊이 기반 상대 이동 계산
class FrontDepthOdometry:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, matrix, offset=(.035, 0., .042425)):
        self.k = np.asarray(matrix, float).reshape(3, 3)
        self.offset = np.asarray(offset, float)
        self.previous = None
        self.rotation_camera = np.eye(3)
        self.position_camera = np.zeros(3)
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.velocity = np.zeros(3)
        self.valid = False
        self.reason = 'waiting_for_front_features'
        self.inliers = 0
        self.error_px = None
        self.stamp = None
        self.failures = 0
        self.tracking_lost = False
        self.last_failure_reason = None
        self.feature_count = 0
        self.quality_checks = True

    # 처리: 새 입력의 시간·품질을 확인해 추정 상태 갱신
    def update(self, gray, depth, stamp, moving_boxes=()):
        self.valid = False
        if self.tracking_lost and self.quality_checks:
            self.reason = 'tracking_lost_restart_required'
            return
        # Retry against the last accepted frame, preserving its pose and depth.
        # Never reset the origin or integrate commands across a tracking gap.
        if self.stamp is not None and (stamp <= self.stamp or (self.quality_checks and stamp-self.stamp > .5)):
            self.failures += 1
            self.tracking_lost = self.quality_checks
            self.reason = 'odometry_time_gap'
            if self.last_failure_reason is None:
                self.last_failure_reason = self.reason
            return
        if self.previous is None:
            self.previous = (gray.copy(), np.asarray(depth).copy(), list(moving_boxes))
            self.stamp = stamp
            self.reason = 'initializing_front_features'
            return
        old_gray, old_depth, old_boxes = self.previous
        mask = np.full(gray.shape, 255, np.uint8)
        for box in old_boxes:
            x1, y1, x2, y2 = np.asarray(box, int)
            cv2.rectangle(mask, (max(0, x1-15), max(0, y1-15)),
                          (min(gray.shape[1]-1, x2+15), min(gray.shape[0]-1, y2+15)), 0, -1)
        try:
            points = cv2.goodFeaturesToTrack(old_gray, 280, .015, 7, mask=mask)
            if points is None or len(points) < 60:
                # Sparse indoor views need denser candidates. Acceptance still
                # requires forward/backward tracking, valid depth and PnP checks.
                points = cv2.goodFeaturesToTrack(old_gray, 600, .003, 3, mask=mask)
            self.feature_count = 0 if points is None else len(points)
            if points is None or len(points) < (30 if self.quality_checks else 4):
                raise ValueError('insufficient_static_features')
            moved, ok, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, points, None, winSize=(21, 21), maxLevel=3)
            if moved is None or ok is None:
                raise ValueError('front_feature_tracking_failed')
            old_pixels, new_pixels = points.reshape(-1, 2), moved.reshape(-1, 2)
            good = (ok.ravel() > 0) & np.isfinite(new_pixels).all(axis=1)
            if self.quality_checks:
                reverse, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, old_gray, moved, None, winSize=(21, 21), maxLevel=3)
                if reverse is None or back_ok is None:
                    raise ValueError('front_feature_reverse_check_failed')
                good &= (back_ok.ravel() > 0) & (np.linalg.norm(reverse.reshape(-1, 2)-old_pixels, axis=1) < 1.)
            for box in moving_boxes:
                x1, y1, x2, y2 = box
                good &= ~((new_pixels[:, 0] >= x1-15) & (new_pixels[:, 0] <= x2+15) &
                          (new_pixels[:, 1] >= y1-15) & (new_pixels[:, 1] <= y2+15))
            scaled = old_pixels * [old_depth.shape[1]/gray.shape[1], old_depth.shape[0]/gray.shape[0]]
            px = np.clip(np.rint(scaled[:, 0]).astype(int), 0, old_depth.shape[1]-1)
            py = np.clip(np.rint(scaled[:, 1]).astype(int), 0, old_depth.shape[0]-1)
            z = old_depth[py, px]
            good &= np.isfinite(z) & ((z > .3) & (z < 12.) if self.quality_checks else z > 0)
            xyz = np.c_[(old_pixels[:, 0]-self.k[0, 2])*z/self.k[0, 0],
                        (old_pixels[:, 1]-self.k[1, 2])*z/self.k[1, 1], z]
            xyz, pixels = xyz[good], new_pixels[good]
            if len(xyz) < (25 if self.quality_checks else 4):
                raise ValueError('insufficient_valid_depth_features')
            # Ring flight bypasses stop policies, not measurement integrity.
            # Outlier tracks must never move the saved path or count a passage.
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                xyz.astype(np.float32), pixels.astype(np.float32), self.k, np.zeros(5),
                iterationsCount=100, reprojectionError=2., confidence=.995, flags=cv2.SOLVEPNP_EPNP)
            minimum_inliers = max(20, len(xyz)*.55) if self.quality_checks else 4
            if not success or inliers is None or len(inliers) < minimum_inliers:
                raise ValueError('visual_motion_inliers_insufficient')
            selected = inliers.ravel()
            projected = cv2.projectPoints(xyz[selected], rvec, tvec, self.k, np.zeros(5))[0].reshape(-1, 2)
            error = float(np.median(np.linalg.norm(projected-pixels[selected], axis=1)))
            r_new_old = cv2.Rodrigues(rvec)[0]
            dt = stamp-self.stamp
            if (not np.isfinite(tvec).all() or not np.isfinite(rvec).all()
                    or np.linalg.norm(tvec) > .04+1.5*min(dt, .5)
                    or np.linalg.norm(rvec) > .04+1.5*min(dt, .5) or error > 1.5):
                raise ValueError('visual_motion_jump_or_residual')
            r_world_new = self.rotation_camera @ r_new_old.T
            p_world_new = self.position_camera-r_world_new @ tvec.ravel()
            new_position = CAMERA_FROM_BODY.T @ (
                p_world_new-r_world_new @ CAMERA_FROM_BODY @ self.offset+CAMERA_FROM_BODY @ self.offset)
            self.velocity = .65*self.velocity+.35*(new_position-self.position)/dt
            self.position = new_position
            self.position_camera, self.rotation_camera = p_world_new, r_world_new
            self.rotation = CAMERA_FROM_BODY.T @ r_world_new @ CAMERA_FROM_BODY
            self.valid = True
            self.tracking_lost = False
            self.failures = 0
            self.inliers, self.error_px = len(inliers), error
            self.reason = 'learned_depth_visual_odometry_valid'
            self.stamp = stamp
            self.previous = (gray.copy(), np.asarray(depth).copy(), list(moving_boxes))
        except (ValueError, cv2.error) as error:
            self.reason = str(error)
            self.last_failure_reason = self.reason
            self.failures += 1

    # 처리: 추정 상태·유효 여부·근거를 기록용 자료로 반환
    def telemetry(self):
        return dict(valid=self.valid, reason=self.reason, inliers=self.inliers,
                    quality_checks_enabled=self.quality_checks,
                    feature_count=self.feature_count, consecutive_failures=self.failures,
                    tracking_lost=self.tracking_lost, last_failure_reason=self.last_failure_reason,
                    reprojection_px=self.error_px, stamp=self.stamp,
                    estimated_position_m=self.position.tolist() if self.valid else None,
                    estimated_velocity_m_s=self.velocity.tolist() if self.valid else None,
                    scale_source='learned_depth_not_measured_range', ground_range_m=None,
                    ground_truth_used=False, provided_map_used=False)
