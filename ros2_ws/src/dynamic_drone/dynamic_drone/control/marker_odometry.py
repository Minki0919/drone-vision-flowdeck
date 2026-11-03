# 파일 역할: 마커 PnP 위치·고도 추정과 마지막 유효 관측 보관
"""Stage-1 PnP localisation in an observed, flight-local landmark frame.

Only marker dimensions, calibration and observed corners are inputs. No fixture
positions, simulator pose, IMU or downward sensors. Metric accuracy is unverified.
Lost marker poses are held as historical references, never depth-extrapolated.
"""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
from collections import deque
import math

import cv2
import numpy as np

from dynamic_drone.control.depth_odometry import FrontDepthOdometry
from dynamic_drone.perception.marker_pose import CAMERA_FROM_BODY, rotation_distance, square_pose_candidates


# 처리: 여러 회전행렬에서 평균 회전 계산
def mean_rotation(rotations):
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


# 파트: 1단계 마커 기반 위치 추정과 마지막 관측 저장
class Stage1MarkerOdometry:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, matrix, distortion, marker_size, offset):
        self.k = np.asarray(matrix, float).reshape(3, 3)
        self.distortion = np.asarray(distortion, float).ravel()
        self.offset = np.asarray(offset, float).reshape(3)
        self.marker_size = float(marker_size)
        if (not np.isfinite(self.k).all() or not np.isfinite(self.distortion).all()
                or not np.isfinite(self.offset).all() or not .05 <= self.marker_size <= 1.5
                or self.k[0, 0] <= 0 or self.k[1, 1] <= 0
                or not np.allclose(self.k[2], [0., 0., 1.])
                or self.distortion.size not in (0, 4, 5, 8, 12, 14)):
            raise ValueError('Valid camera calibration and measured marker size required')
        half = self.marker_size/2.
        self.object_points = np.array([[-half, half, 0.], [half, half, 0.],
                                       [half, -half, 0.], [-half, -half, 0.]])
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.velocity = np.zeros(3)
        self.valid = False
        self.stamp = None
        self.last_input_stamp = None
        self.reason = 'waiting_for_stationary_marker_origin'
        self.source = 'unavailable'
        self.observations = {}
        self.rejections = {}
        self.landmarks = {}
        self.samples = {}
        self.corner_samples = {}
        self.pose_method = 'unavailable'
        self.pose_marker_count = 0
        self.orientation_samples = {}
        self.supported_ids = []
        self.altitude_reference = None
        self.control_height = None
        self.control_height_stamp = None
        self.altitude_control_status = {}
        self.marker_motion_references = {}
        self.relative_pose_ids = set()
        self.pose_innovation = {}
        # 마지막 유효 위치·고도 저장; 추정 상실 시 과거 기록으로만 사용
        self.last_valid_marker_snapshot = None

    def pose_anchor(self, identifier):
        """One fixed observed anchor for both single and joint PnP."""
        landmark = self.landmarks[identifier]
        reference = self.marker_motion_references.get(identifier)
        if not landmark.get('orientation_reliable', True) and reference is not None:
            return dict(position=reference['position']+reference['rotation'] @ reference['body'],
                        rotation=reference['rotation'] @ CAMERA_FROM_BODY.T @ reference['rotation_camera']), True
        return landmark, False

    # 처리: 마지막 동일 마커 관측 또는 기억한 기준으로 위치 계산
    def observation_pose(self, identifier, observation):
        anchor, relative = self.pose_anchor(identifier)
        position, rotation = self.world_pose(anchor, observation)
        return position, rotation, relative

    # 처리: 현재 유효 위치와 같은 시점의 마커 관측 저장
    def remember_marker_motion(self, identifier, observation, stamp):
        self.marker_motion_references[identifier] = dict(
            stamp=float(stamp), position=self.position.copy(), rotation=self.rotation.copy(),
            body=observation['body'].copy(), rotation_camera=observation['rotation_camera'].copy())

    # 처리: 재투영 오차와 기존 자세·위치 차이로 PnP 후보 비교
    def candidate_cost(self, identifier, observation):
        position, rotation, _ = self.observation_pose(identifier, observation)
        weight = 2. if self.landmarks[identifier].get('orientation_reliable', True) else .2
        return (observation['reprojection_px']+weight*rotation_distance(rotation, self.rotation)
                +np.linalg.norm(position-self.position))

    # 처리: 유효 마커 고도를 완만하게 반영; 오래된 기준은 거부
    def control_altitude(self, now):
        """Smooth control height from accepted marker PnP observations only.

        Keep raw localisation and its safety bounds separate. A held reference
        is not a new measurement and expires after three simulation seconds.
        """
        anchor = self.altitude_reference
        age = None if anchor is None else now-anchor['stamp']
        if not self.valid or age is None or not 0 <= age <= 3.:
            self.altitude_control_status = dict(valid=False, reference_age_seconds=age,
                                                reason='marker_height_reference_expired_or_pose_invalid')
            return None
        target = anchor['height_m']
        previous = self.control_height
        dt = (0. if self.control_height_stamp is None else
              min(max(now-self.control_height_stamp, 0.), .1))
        if previous is None:
            self.control_height = target
        elif dt > 0:
            # Limit changes in the controller's estimate, not raw observations
            # or physical flight height. Preserve raw out-of-bounds shutdown.
            self.control_height += float(np.clip(target-previous, -.30*dt, .30*dt))
        speed = 0. if previous is None or dt <= 0 else (self.control_height-previous)/dt
        self.control_height_stamp = now
        self.altitude_control_status = dict(valid=True,
            source='last_accepted_marker_PnP_height',
            marker_ids=anchor['marker_ids'], reference_stamp=anchor['stamp'],
            reference_age_seconds=float(age), reference_height_m=anchor['height_m'],
            depth_position_used=False, raw_estimated_height_m=float(self.position[2]),
            filtered_control_height_m=float(self.control_height),
            filtered_vertical_speed_m_s=float(speed), estimate_change_limit_m_s=.30,
            measured_ground_altitude_available=False)
        return float(self.control_height), float(speed)

    # 처리: 알려진 마커 크기로 PnP 후보 계산과 품질 검사
    def poses(self, identifier, target):
        pixels = np.asarray(target.corners, float).reshape(4, 2)
        if not np.isfinite(pixels).all():
            self.rejections[identifier] = 'nonfinite_marker_corners'
            return []
        if min(np.linalg.norm(pixels-np.roll(pixels, 1, axis=0), axis=1)) < 12.:
            self.rejections[identifier] = 'too_few_pixels_for_metric_PnP'
            return []
        # Smooth altitude input without requiring extra confirmation frames.
        samples = self.corner_samples.setdefault(identifier, deque(maxlen=5))
        if samples and (not 0 < target.stamp-samples[-1][0] <= .15
                        or np.max(np.linalg.norm(pixels-samples[-1][1], axis=1)) > 8.):
            samples.clear()
        samples.append((target.stamp, pixels.copy()))
        averaged = np.mean([row[1] for row in samples], axis=0)
        candidates = square_pose_candidates(self.object_points, averaged, self.k, self.distortion)
        current = []
        for row in candidates:
            projected = cv2.projectPoints(self.object_points,
                cv2.Rodrigues(row['rotation_camera'])[0], row['translation_camera'],
                self.k, self.distortion)[0].reshape(4, 2)
            if np.sqrt(np.mean(np.sum((projected-pixels)**2, axis=1))) <= 1.5:
                current.append(row)
        candidates = current or square_pose_candidates(self.object_points, pixels, self.k, self.distortion)
        for row in candidates:
            row['body'] = CAMERA_FROM_BODY.T @ row['translation_camera']+self.offset
            row['source'] = ('OpenCV_refined_square_PnP_tracked_corners'
                             if getattr(target, 'measurement_source', '') == 'verified_optical_flow_corners'
                             else 'OpenCV_refined_square_PnP_known_size')
        return candidates

    # 처리: 관측으로 만든 비행 기준 좌표계로 자세 변환
    def world_pose(self, landmark, observation):
        rotation = landmark['rotation'] @ observation['rotation_camera'].T @ CAMERA_FROM_BODY
        position = landmark['position']-rotation @ observation['body']
        return position, rotation

    # 처리: 여러 관측 마커를 이용해 공통 카메라 자세 계산
    def joint_pose(self, frame):
        """Fit one camera pose to several observed landmarks, rejecting outliers."""
        # The joint fit resolves weak single-tag normals from the spatial
        # baseline between confirmed tags; RANSAC still verifies their corners.
        ids = [i for i in self.observations if i in self.landmarks]
        if len(ids) < 2:
            return None
        anchors = [self.pose_anchor(i)[0] for i in ids]
        objects = np.concatenate([(anchor['rotation'] @ self.object_points.T).T+
                                  anchor['position'] for anchor in anchors])
        pixels = np.concatenate([frame.markers[i].corners for i in ids]).astype(np.float64)
        labels = np.repeat(ids, 4)
        r_cw = CAMERA_FROM_BODY @ self.rotation.T
        rvec = cv2.Rodrigues(r_cw)[0]
        tvec = (-r_cw @ (self.position+self.rotation @ self.offset)).reshape(3, 1)
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                objects, pixels, self.k, self.distortion, rvec=rvec, tvec=tvec,
                useExtrinsicGuess=True, iterationsCount=100, reprojectionError=2.,
                confidence=.995, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or inliers is None:
                return None
            chosen = inliers.ravel()
            supported = [i for i in ids if np.count_nonzero(labels[chosen] == i) >= 3]
            if len(chosen) < max(6, math.ceil(.75*len(objects))) or len(supported) < 2:
                return None
            rvec, tvec = cv2.solvePnPRefineLM(objects[chosen], pixels[chosen], self.k,
                                            self.distortion, rvec, tvec)
            r_cw = cv2.Rodrigues(rvec)[0]
            camera_points = (r_cw @ objects[chosen].T).T+tvec.reshape(3)
            projected = cv2.projectPoints(objects[chosen], rvec, tvec, self.k,
                                         self.distortion)[0].reshape(-1, 2)
            error = float(np.median(np.linalg.norm(projected-pixels[chosen], axis=1)))
            if not np.isfinite(camera_points).all() or camera_points[:, 2].min() <= .15 or error > 1.5:
                return None
            rotation = r_cw.T @ CAMERA_FROM_BODY
            position = -r_cw.T @ tvec.reshape(3)-rotation @ self.offset
            return position, rotation, supported
        except cv2.error:
            return None

    # 처리: 마커 탐지·기준 설정·후보 검증·위치 갱신
    def update_markers(self, frame):
        self.valid = False
        self.observations, self.rejections = {}, {}
        self.relative_pose_ids = set()
        self.pose_innovation = {}
        if frame is None:
            self.reason = 'camera_unavailable'
            return False
        if (not math.isfinite(frame.stamp) or
                (self.last_input_stamp is not None and frame.stamp <= self.last_input_stamp)):
            self.reason = 'nonmonotonic_marker_timestamp'
            return False
        self.last_input_stamp = frame.stamp
        for identifier, target in frame.markers.items():
            if identifier not in range(1, 7):
                continue
            if identifier in frame.tracked_marker_ids and identifier not in self.landmarks:
                self.rejections[identifier] = 'tracking_cannot_bootstrap_unknown_landmark'
                continue
            candidates = self.poses(identifier, target)
            if not candidates:
                self.rejections.setdefault(identifier, 'PnP_reprojection_or_range_rejected')
                continue
            if identifier in self.landmarks and self.stamp is not None:
                # Temporal orientation disambiguates a planar tag; a large
                # flip remains rejected instead of rotating the entire map.
                candidates.sort(key=lambda row: self.candidate_cost(identifier, row))
            self.observations[identifier] = candidates[0]

        # 기준 설정: 이륙 전 정지 관측으로 비행 좌표계 원점 설정
        if not self.landmarks:
            # Establish the origin before arming; never bootstrap a new origin
            # in the middle of a flight following a tracking failure.
            observation = self.observations.get(1)
            if observation is None or 1 not in frame.decoded_marker_ids:
                self.reason = 'waiting_for_marker_1_metric_origin'
                return False
            self.landmarks[1] = dict(position=observation['body'].copy(),
                rotation=CAMERA_FROM_BODY.T @ observation['rotation_camera'],
                first_seen=frame.stamp, last_seen=frame.stamp, frames=1, spread=0.,
                anchor_pose_source='initial_body_pose', orientation_reliable=True,
                orientation_pixels=0.)
            self.position, self.rotation = np.zeros(3), np.eye(3)
            self.velocity[:] = 0.
            self.pose_method, self.pose_marker_count = 'stationary_marker_origin', 1
            self.supported_ids = [1]
        else:
            candidates = []
            for identifier, observation in self.observations.items():
                if identifier in self.landmarks:
                    p, r, relative = self.observation_pose(identifier, observation)
                    if relative:
                        self.relative_pose_ids.add(identifier)
                    candidates.append((observation['reprojection_px'], identifier, p, r))
            if not candidates:
                self.reason = 'no_visible_confirmed_landmark_for_PnP_localisation'
                return False
            trusted = [row for row in candidates if self.landmarks[row[1]].get('orientation_reliable', True)]
            if trusted:
                candidates = trusted
            # Prefer a coherent, temporally close cluster instead of changing
            # reference marker whenever a distant tag has tiny pixel residuals.
            candidate_snapshot = tuple(candidates)
            candidates.sort(key=lambda row: (
                -sum(np.linalg.norm(other[2]-row[2]) <= .35 and
                     rotation_distance(other[3], row[3]) <= math.radians(12) for other in candidate_snapshot),
                np.linalg.norm(row[2]-self.position)+rotation_distance(row[3], self.rotation), row[0]))
            _, _, reference_p, reference_r = candidates[0]
            consistent = [row for row in candidates if np.linalg.norm(row[2]-reference_p) <= .35
                          and rotation_distance(row[3], reference_r) <= math.radians(12)]
            p = np.median([row[2] for row in consistent], axis=0)
            r = mean_rotation([row[3] for row in consistent])
            supported = [row[1] for row in consistent]
            # 현재 자세 계산: 여러 유효 마커가 있으면 공통 자세 사용
            joint = self.joint_pose(frame)
            self.pose_method = 'temporally_consistent_single_marker_cluster'
            if joint is not None:
                p, r, supported = joint
                self.pose_method = 'multi_marker_RANSAC_refined_PnP'
            elif any(i in self.relative_pose_ids for i in supported):
                self.pose_method = 'same_marker_relative_PnP_in_accepted_flight_frame'
            self.pose_marker_count = len(supported)
            self.supported_ids = supported
            dt = frame.stamp-self.stamp
            jump = float(np.linalg.norm(p-self.position))
            self.pose_innovation = dict(position_jump_m=jump,
                vertical_jump_m=float(p[2]-self.position[2]),
                rotation_jump_deg=math.degrees(rotation_distance(r, self.rotation)),
                supporting_ids=list(supported))
            self.velocity = .8*self.velocity+.2*(p-self.position)/dt
            self.position, self.rotation = p, r
            for identifier in supported:
                self.landmarks[identifier]['last_seen'] = frame.stamp

        self.stamp = frame.stamp
        self.valid = True
        self.source, self.reason = 'OpenCV_marker_PnP', 'observed_landmark_PnP_valid'
        # Keep next tags' relative references fresh while the current tag
        # localises this image. They need not determine the current pose, but
        # must be decoded now and have passed relative PnP quality checks.
        # This avoids using a far tag's initial normal after the old tag exits.
        independent_anchor = any(self.landmarks[i].get('orientation_reliable', True)
                                 for i in self.supported_ids)
        for identifier, observation in self.observations.items():
            # Never re-anchor a tag using a pose derived from that same tag.
            # Feeding a filtered pose back into its own reference attenuates
            # real motion and accumulates orientation/height error at handoff.
            if (independent_anchor and identifier in frame.decoded_marker_ids
                    and identifier not in self.supported_ids):
                self.remember_marker_motion(identifier, observation, frame.stamp)
        self.altitude_reference = dict(height_m=float(self.position[2]),
            stamp=float(frame.stamp), marker_ids=list(self.supported_ids))
        self.last_valid_marker_snapshot = dict(stamp=float(frame.stamp),
            position_m=self.position.tolist(), height_m=float(self.position[2]),
            rotation=self.rotation.tolist(), marker_ids=list(self.supported_ids),
            source='accepted_marker_PnP', historical_only_when_pose_invalid=True)
        self.learn(frame)
        self.refine_orientations(frame)
        return True

    # 처리: 현재 비행에서 관측한 마커 기준을 기억; 신경망 학습이 아님
    def learn(self, frame):
        # Learn ALL visible route IDs, not just the active target. Freeze
        # established anchors: bad depth updates must not erase a good marker.
        for identifier, observation in self.observations.items():
            if identifier in self.landmarks or identifier in frame.tracked_marker_ids:
                continue
            p = self.position+self.rotation @ observation['body']
            r = self.rotation @ CAMERA_FROM_BODY.T @ observation['rotation_camera']
            samples = self.samples.setdefault(identifier, deque(maxlen=5))
            if samples and not 0 < frame.stamp-samples[-1][0] <= .3:
                samples.clear()
            samples.append((frame.stamp, p, r))
            if len(samples) < 3:
                continue
            center = np.median([row[1] for row in samples], axis=0)
            orientation = mean_rotation([row[2] for row in samples])
            spread = max(np.linalg.norm(row[1]-center) for row in samples)
            if spread > .25 or max(rotation_distance(row[2], orientation) for row in samples) > math.radians(10):
                self.rejections[identifier] = 'new_landmark_needs_consistent_PnP_frames'
                continue
            self.landmarks[identifier] = dict(position=center, rotation=orientation,
                first_seen=samples[0][0], last_seen=frame.stamp, frames=len(samples), spread=float(spread),
                anchor_pose_source=self.source, orientation_reliable=False,
                orientation_pixels=float(np.min(np.linalg.norm(frame.markers[identifier].corners-
                    np.roll(frame.markers[identifier].corners, 1, axis=0), axis=1))))
            # A new tag's relative reference belongs to this accepted pose and
            # image, not to its distant, weakly estimated surface normal.
            self.remember_marker_motion(identifier, observation, frame.stamp)

    # 처리: 독립 마커 기준으로 불확실한 표식 자세 보정
    def refine_orientations(self, frame):
        """Upgrade a far tag's normal from independently localised close views.

        The stored centre and flight origin are not reset. A tag cannot certify
        its own orientation by localising the camera from that same orientation.
        """
        for identifier, landmark in self.landmarks.items():
            if (landmark.get('orientation_reliable', True) or identifier not in self.observations
                    or identifier not in frame.decoded_marker_ids):
                continue
            quad = frame.markers[identifier].corners
            side = float(np.min(np.linalg.norm(quad-np.roll(quad, 1, axis=0), axis=1)))
            if side < 18. or side < 1.15*landmark.get('orientation_pixels', 0.):
                continue
            references = [(self.observations[i]['reprojection_px'], i) for i in self.observations
                          if i != identifier and i in self.landmarks
                          and self.landmarks[i].get('orientation_reliable', True)
                          and i in frame.decoded_marker_ids]
            anchor_source = None
            if references:
                _, reference_id = min(references)
                p_ref, r_ref = self.world_pose(self.landmarks[reference_id], self.observations[reference_id])
                anchor_source = 'independent_confirmed_marker'
            else:
                continue
            observation = self.observations[identifier]
            # Check the remembered centre, but do not move it to make a bad
            # orientation fit. Large inconsistencies still stop localisation.
            if np.linalg.norm(p_ref+r_ref @ observation['body']-landmark['position']) > .35:
                continue
            normal = r_ref @ CAMERA_FROM_BODY.T @ observation['rotation_camera']
            samples = self.orientation_samples.setdefault(identifier, deque(maxlen=5))
            if samples and not 0 < frame.stamp-samples[-1][0] <= .3:
                samples.clear()
            samples.append((frame.stamp, normal))
            if len(samples) < 5:
                continue
            average = mean_rotation([row[1] for row in samples])
            if max(rotation_distance(row[1], average) for row in samples) > math.radians(4):
                continue
            landmark['rotation'] = average
            landmark['orientation_pixels'] = side
            landmark['orientation_reliable'] = side >= 28.
            landmark['orientation_refinement_source'] = anchor_source
            samples.clear()

    # 처리: 단계 전환 시 기존 좌표계를 링·차량 위치 추정기에 전달
    def seed_depth_odometry(self):
        position, rotation, velocity = self.position, self.rotation, self.velocity
        bridge = FrontDepthOdometry(self.k, self.offset)
        bridge.position, bridge.rotation = position.copy(), rotation.copy()
        bridge.velocity = velocity.copy()
        bridge.rotation_camera = CAMERA_FROM_BODY @ rotation @ CAMERA_FROM_BODY.T
        bridge.position_camera = (CAMERA_FROM_BODY @ position+
            bridge.rotation_camera @ CAMERA_FROM_BODY @ self.offset-CAMERA_FROM_BODY @ self.offset)
        return bridge

    # 처리: 현재 비행에서 관측한 마커 기억을 기록용으로 반환
    def memory(self):
        return {identifier: dict(position_m=self.pose_anchor(identifier)[0]['position'].tolist(),
            first_seen_seconds=float(row['first_seen']), last_seen_seconds=float(row['last_seen']),
            confirming_frames=row['frames'], sample_spread_m=row['spread'], valid=True,
            inconsistent_updates=0, source='observed_OpenCV_PnP_landmark',
            position_anchor_source=row['anchor_pose_source'],
            orientation_reliable=row.get('orientation_reliable', True),
            orientation_pixels=row.get('orientation_pixels'),
            orientation_refinement_source=row.get('orientation_refinement_source'),
            coordinate_frame='initial_body_pose_observed_marker_map', metric_accuracy_verified=False)
            for identifier, row in self.landmarks.items()}

    # 처리: 추정 상태·유효 여부·근거를 기록용 자료로 반환
    def telemetry(self):
        return dict(valid=self.valid, reason=self.reason, source=self.source, stamp=self.stamp,
            estimated_position_m=self.position.tolist() if self.valid else None,
            estimated_velocity_m_s=self.velocity.tolist() if self.valid else None,
            scale_source='configured_marker_side_length_m', marker_size_m=self.marker_size,
            remembered_ids=sorted(self.landmarks),
            PnP_rejections={str(i): reason for i, reason in self.rejections.items()},
            pose_method=self.pose_method, pose_marker_count=self.pose_marker_count,
            altitude_reference_control=self.altitude_control_status.copy(),
            last_valid_marker_snapshot=self.last_valid_marker_snapshot,
            depth_position_fallback_enabled=False,
            marker_pose_innovation=self.pose_innovation.copy(),
            relative_PnP_marker_ids=sorted(self.relative_pose_ids),
            ground_range_m=None, ground_truth_used=False, provided_map_used=False,
            map_source='observed_markers_only_this_flight', metric_accuracy_verified=False)
