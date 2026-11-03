# 파일 역할: 단계 전환·마커 방문·링 통과·차량 추종 제어
"""Experimental front-camera mission controller; no legacy Flow/ToF adapter.

Mission IDs and ring count are development task rules, not scene coordinates.
Stage 1 prefers calibrated known-size marker PnP and an observed landmark map.
Rings use image alignment; downward ToF verifies ring passage and flag-guided truck landing.
"""
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
import math
import cv2
import numpy as np
from dynamic_drone.control.marker_servo import ImageTarget
from dynamic_drone.control.depth_odometry import FrontDepthOdometry
from dynamic_drone.control.marker_odometry import Stage1MarkerOdometry
from dynamic_drone.control.image_pid import ImageCenterPID
from dynamic_drone.perception.marker_pose import CAMERA_FROM_BODY, square_pose_candidates


# 처리: 각도를 -π~π 범위로 정규화
def angle(value):
    return (value+math.pi)%(2*math.pi)-math.pi


# 파트: 한 시점의 영상·마커·링 관측 묶음
class MissionFrame(ImageTarget):
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, gray, stamp, markers, ring, decoded_ids=None, tracked_ids=(), visible_rings=None):
        super().__init__(stamp, np.zeros(2), 0., np.zeros((4, 2)))
        self.gray = gray
        self.markers = markers
        self.ring = ring
        self.visible_rings = list(visible_rings or [])
        self.decoded_marker_ids = set(markers) if decoded_ids is None else set(decoded_ids)
        self.tracked_marker_ids = set(tracked_ids)


# 파트: ArUco·링 영상 관측 구성
class MissionObserver:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, yolo):
        self.yolo = yolo
        self.detect_rings = True
        self.ring_target_locked = False
        from dynamic_drone.perception.aruco_detection import ArucoImageDetector
        self.detector = ArucoImageDetector()
        self.marker_diagnostics = {}
        self.ring_diagnostics = {}
        self.latest = None
        self.reason = 'waiting_for_mission_image'
        self.last_stamp = None
        self.previous_gray = None
        self.previous_image_stamp = None
        self.corner_tracks = {}
        self.ring_track = None
        self.ring_gray = None

    # 처리: 현재 관측을 무효 처리하고 원인을 기록
    def reject(self, reason):
        self.latest = None
        self.reason = reason
        self.ring_diagnostics.clear()
        self.previous_gray = None
        self.previous_image_stamp = None
        self.corner_tracks.clear()
        self.ring_track = self.ring_gray = None

    # 처리: 전방 영상에서 짧은 모서리 추적; 새 ID 인식으로 계산하지 않음
    def track_missing_corners(self, gray, stamp, markers, filtered):
        """Brief measured optical-flow bridge, never a newly decoded identity."""
        tracked_ids = set()
        if (self.previous_gray is None or self.previous_image_stamp is None
                or not 0 < stamp-self.previous_image_stamp <= .15):
            return tracked_ids
        for identifier, old in self.corner_tracks.items():
            if (identifier not in range(1, 7) or identifier in markers or identifier in filtered
                    or not 0 < stamp-old.decoded_stamp <= .25):
                continue
            points = np.asarray(old.corners, np.float32).reshape(4, 1, 2)
            try:
                moved, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None,
                                                      winSize=(21, 21), maxLevel=3)
                if moved is None or ok is None or not np.all(ok):
                    continue
                reverse, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, moved, None,
                                                             winSize=(21, 21), maxLevel=3)
                if (reverse is None or back_ok is None or not np.all(back_ok)
                        or np.max(np.linalg.norm(reverse-points, axis=2)) > .75):
                    continue
                quad = moved.reshape(4, 2)
                if (not np.isfinite(quad).all() or np.min(quad) < 2 or np.max(quad) > 317
                        or not cv2.isContourConvex(quad.astype(np.float32))):
                    continue
                area = float(cv2.contourArea(quad))
                old_area = float(cv2.contourArea(points.reshape(4, 2)))
                if (area < 64 or not .7 <= area/max(old_area, 1.) <= 1.4
                        or np.min(np.linalg.norm(quad-np.roll(quad, 1, axis=0), axis=1)) < 8):
                    continue
                lo, hi = quad.min(axis=0), quad.max(axis=0)
                # Never let an old identity latch onto another decoded tag.
                conflict = False
                for other_id, target in markers.items():
                    if other_id == identifier:
                        continue
                    overlap = np.maximum(0., np.minimum(hi, target.corners.max(axis=0))-
                                          np.maximum(lo, target.corners.min(axis=0)))
                    if float(np.prod(overlap)) > .2*area:
                        conflict = True
                        break
                if conflict:
                    continue
                target = ImageTarget(stamp, (quad.mean(axis=0)-160)/160, area/(320*320), quad)
                target.decoded_stamp = old.decoded_stamp
                target.measurement_source = 'verified_optical_flow_corners'
                markers[identifier] = target
                tracked_ids.add(identifier)
            except cv2.error:
                continue
        return tracked_ids

    @staticmethod
    def same_ring(first, second):
        """Match size and overlap so a small rear ring cannot take over."""
        a, b = np.asarray(first['box']), np.asarray(second['box'])
        area_a, area_b = np.prod(a[2:]-a[:2]), np.prod(b[2:]-b[:2])
        overlap = np.prod(np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])))
        return .80 <= area_b/max(area_a, 1.) <= 1.25 and overlap/max(area_a+area_b-overlap, 1.) >= .6

    def track_ring(self, gray, stamp):
        """Measure rim motion on each new image; expire 0.6 s after YOLO confirmation."""
        old = self.ring_track
        if (old is None or self.ring_gray is None
                or not 0 < stamp-old['box_stamp'] <= .15
                or not 0 <= stamp-old['detected_stamp'] <= .60):
            return None
        box = np.asarray(old['box'], float)
        center, size = (box[:2]+box[2:])/2, box[2:]-box[:2]
        yy, xx = np.indices(gray.shape)
        radius = ((xx-center[0])/(size[0]/2))**2+((yy-center[1])/(size[1]/2))**2
        # Use the rim, not the background/next ring visible through its opening.
        mask = np.uint8((radius >= .65**2) & (radius <= 1.15**2))*255
        points = cv2.goodFeaturesToTrack(self.ring_gray, 80, .01, 3, mask=mask, blockSize=3)
        if points is None or len(points) < 8:
            return None
        moved, ok, error = cv2.calcOpticalFlowPyrLK(self.ring_gray, gray, points, None,
                                                   winSize=(21, 21), maxLevel=3)
        if moved is None or ok is None or error is None:
            return None
        reverse, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.ring_gray, moved, None,
                                                      winSize=(21, 21), maxLevel=3)
        if reverse is None or back_ok is None:
            return None
        good = (ok.ravel().astype(bool) & back_ok.ravel().astype(bool)
                & (error.ravel() < 25.) & (np.linalg.norm(reverse-points, axis=2).ravel() < .75)
                & np.isfinite(moved.reshape(-1, 2)).all(axis=1)
                & (moved.reshape(-1, 2).min(axis=1) >= 2.)
                & (moved.reshape(-1, 2).max(axis=1) <= 317.))
        if np.count_nonzero(good) < 8:
            return None
        before, after = points[good].reshape(-1, 2), moved[good].reshape(-1, 2)
        transform, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC,
            ransacReprojThreshold=1., maxIters=200, confidence=.99)
        if transform is None or inliers is None or not np.isfinite(transform).all():
            return None
        kept = inliers.ravel().astype(bool)
        if (np.count_nonzero(kept) < 8 or np.mean(kept) < .70
                or np.any(np.ptp(before[kept], axis=0) < .40*size)):
            return None
        scale = np.linalg.norm(transform[:, 0])
        rotation = math.atan2(transform[1, 0], transform[0, 0])
        new_center = transform[:, :2] @ center+transform[:, 2]
        if (not .9 <= scale <= 1.12 or abs(rotation) > .12
                or np.linalg.norm(new_center-center) > max(8., .18*min(size))):
            return None
        # Require current image edges: vanished rings/blank frames are not observations.
        edge_distance = cv2.distanceTransform(255-cv2.Canny(gray, 40, 100), cv2.DIST_L2, 3)
        samples = np.rint(after[kept]).astype(int)
        if np.mean(edge_distance[samples[:, 1], samples[:, 0]] <= 2.) < .70:
            return None
        new_size = size*scale
        projected_box = np.r_[new_center-new_size/2, new_center+new_size/2]
        new_box = np.clip(projected_box, 0., 320.)
        visible_size = new_box[2:]-new_box[:2]
        if np.prod(visible_size) < .80*np.prod(new_size):
            return None
        # Match YOLO's visible bounding-box convention at image edges.
        new_center = (new_box[:2]+new_box[2:])/2
        return dict(old, center=new_center, box_center=new_center, box=new_box,
                    box_stamp=stamp, axes=tuple(visible_size), contour=None,
                    source='verified_ring_optical_flow', tracked_points=int(np.count_nonzero(kept)))

    # 처리: 새 영상의 형식·시간을 검사하고 인식 결과 갱신
    def process(self, gray, stamp):
        if self.last_stamp is not None and stamp <= self.last_stamp:
            self.reject('nonmonotonic_mission_frame')
            return
        self.last_stamp = stamp
        if gray.shape != (320, 320) or gray.dtype != np.uint8:
            self.reject('unexpected_mission_image_format')
            return
        # User-provided detector with subpixel corners for calibrated PnP.
        decoded = {}
        ambiguous = set()
        corners, identifiers, rejected = self.detector.detectMarkers(gray)
        rejected_candidates = len(rejected)
        if identifiers is not None:
            for box, identifier in zip(corners, identifiers.ravel()):
                identifier = int(identifier)
                points = box.reshape(4, 2)
                if identifier in decoded:
                    if np.max(np.linalg.norm(points-decoded[identifier], axis=1)) > 5.:
                        ambiguous.add(identifier)
                else:
                    decoded[identifier] = points
        markers = {}
        filtered = {}
        for identifier, points in decoded.items():
            area = cv2.contourArea(points.astype(np.float32))
            sides = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
            if identifier in ambiguous:
                filtered[identifier] = 'duplicate_id_or_conflicting_corners'
            elif np.min(points) < 2 or np.max(points) > 317:
                filtered[identifier] = 'clipped_at_image_edge'
            elif area < 64 or np.min(sides) < 8:
                filtered[identifier] = 'insufficient_image_pixels'
            else:
                markers[identifier] = ImageTarget(stamp, (points.mean(axis=0)-160)/160, area/(320*320), points)
                markers[identifier].decoded_stamp = stamp
                markers[identifier].measurement_source = 'fresh_ArUco_decode'
        decoded_ids = set(markers)
        tracked_ids = set()
        if not self.detect_rings:
            tracked_ids = self.track_missing_corners(gray, stamp, markers, filtered)
            self.previous_gray, self.previous_image_stamp = gray.copy(), stamp
            self.corner_tracks = {i: row for i, row in markers.items() if i in range(1, 7)}
        else:
            self.previous_gray, self.previous_image_stamp = None, None
            self.corner_tracks.clear()
        self.marker_diagnostics = dict(decoded_ids=sorted(decoded), accepted_ids=sorted(markers),
                                       fresh_decoded_ids=sorted(decoded_ids), tracked_ids=sorted(tracked_ids),
                                       filtered_ids=filtered, rejected_quad_candidates=rejected_candidates,
                                       preprocessing='raw_grayscale_user_ArUco_subpixel_corners',
                                       detector_source='ArUco_marker_detection.py', accuracy=None)
        detections = ([d for d in self.yolo.predict(gray, stamp) if d['label'] == 'ring']
                      if self.detect_rings else [])
        # Acquisition uses fresh boxes or fresh measured rim motion. Cached YOLO
        # coordinates are display-only while the already locked ring is crossed.
        if not self.detect_rings or self.ring_target_locked:
            self.ring_track = self.ring_gray = None
        tracked = self.track_ring(gray, stamp) if self.detect_rings and not self.ring_target_locked else None
        fresh = [d for d in detections if d['capture_seconds'] == stamp]
        ring = None
        visible_rings = []
        for detection in detections if self.ring_target_locked else fresh:
            box = np.asarray(detection['xyxy'], float)
            if not np.isfinite(box).all() or min(box[2:]-box[:2]) <= 0:
                continue
            box_center = (box[:2]+box[2:])/2
            candidate = dict(center=box_center, box_center=box_center, box=box,
                             box_stamp=detection['capture_seconds'],
                             axes=tuple(box[2:]-box[:2]), rotation=0., contour=None,
                             confidence=detection['confidence'], detected_stamp=detection['capture_seconds'],
                             source='yolo')
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(1, x1-4), max(1, y1-4)
            x2, y2 = min(319, x2+4), min(319, y2+4)
            roi = gray[y1:y2, x1:x2]
            contours = (cv2.findContours(cv2.Canny(roi, 40, 100), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
                        if min(roi.shape, default=0) > 0 else [])
            candidates = []
            for contour in contours:
                if len(contour) < 5:
                    continue
                (cx, cy), (a, b), rotation = cv2.fitEllipse(contour)
                if not np.isfinite([cx, cy, a, b, rotation]).all() or min(a, b) <= 0:
                    continue
                candidates.append((a*b, (cx+x1, cy+y1), (a, b), rotation,
                                   contour+np.array([[[x1, y1]]])))
            if candidates:
                _, center, axes, rotation, contour = max(candidates, key=lambda row: row[0])
                candidate.update(center=np.array(center), axes=axes, rotation=rotation, contour=contour)
            visible_rings.append(candidate)
            if ring is None or np.prod(box[2:]-box[:2]) > np.prod(ring['box'][2:]-ring['box'][:2]):
                ring = candidate
        larger_visible = (tracked is not None and ring is not None
            and np.prod(ring['box'][2:]-ring['box'][:2]) > 1.5*np.prod(tracked['box'][2:]-tracked['box'][:2]))
        if tracked is not None and not larger_visible:
            matches = [candidate for candidate in visible_rings if self.same_ring(tracked, candidate)]
            if matches:
                matched = max(matches, key=lambda candidate: candidate['confidence'])
                # Optical flow predicts the new location; blend only detector jitter.
                box = .65*tracked['box']+.35*matched['box']
                matched.update(box=box, box_center=(box[:2]+box[2:])/2,
                               source='yolo_and_verified_ring_flow', tracked_points=tracked['tracked_points'])
                ring = matched
            else:
                ring = tracked
                visible_rings.append(tracked)
        elif (not larger_visible and ring is not None and self.ring_track is not None
                and 0 <= stamp-self.ring_track['detected_stamp'] <= .60
                and not self.same_ring(self.ring_track, ring)):
            # Briefly wait for the near target instead of turning toward a rear ring.
            ring = None
        if not self.ring_target_locked and ring is not None:
            self.ring_track, self.ring_gray = ring.copy(), gray.copy()
        elif self.ring_track is not None and stamp-self.ring_track['detected_stamp'] > .60:
            self.ring_track = self.ring_gray = None
        self.ring_diagnostics = dict(active=self.detect_rings, yolo_candidates=len(detections),
            fresh_yolo_candidates=len(fresh),
            target_source=None if ring is None else ring['source'],
            target_yolo_age_s=None if ring is None else stamp-ring['detected_stamp'],
            tracked_points=0 if ring is None else ring.get('tracked_points', 0),
            rejected_non_ring_shapes=getattr(self.yolo, 'rejected_rings', 0),
            target_locked=self.ring_target_locked,
            target_acquisition_enabled=self.detect_rings and not self.ring_target_locked,
            box_detected=ring is not None, rim_geometry_available=ring is not None and ring['contour'] is not None,
            confidence=None if ring is None else ring['confidence'])
        self.latest = MissionFrame(gray, stamp, markers,
            None if self.ring_target_locked else ring, decoded_ids, tracked_ids, visible_rings)
        self.reason = 'front_mission_image_valid'

    # 처리: 추정 상태·유효 여부·근거를 기록용 자료로 반환
    def telemetry(self):
        return dict(valid=self.latest is not None, reason=self.reason,
                    capture_stamp=None if self.latest is None else self.latest.stamp,
                    marker_ids=[] if self.latest is None else list(self.latest.markers),
                    ring_visible=self.latest is not None and bool(self.latest.visible_rings),
                    control_ring_visible=self.latest is not None and self.latest.ring is not None,
                    ring_detection=self.ring_diagnostics,
                    marker_detection=self.marker_diagnostics,
                    marker_size_m=None, position_m=None, ground_range_m=None, map_used=False)


# 파트: 링 단계의 깊이·위치 관측 연결; 마커·차량 단계는 표시 전용
class MissionDepth:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, pilot):
        self.pilot = pilot
        self.latest = None
        self.reason = 'waiting_for_mission_depth'

    # 처리: 현재 관측을 무효 처리하고 원인을 기록
    def reject(self, reason):
        self.latest = None
        self.reason = reason

    # 처리: 새 입력의 시간·품질을 확인해 추정 상태 갱신
    def update(self, depth, frame, image_shape):
        self.latest = None
        if self.pilot.truck_stage_active():
            self.reason = 'truck_relative_flag_PnP_depth_display_only'
            return
        if frame is None or self.pilot.odometry is None:
            self.reject('waiting_for_image_and_camera_calibration')
            return
        if self.pilot.stage1_active():
            self.reason = 'stage1_marker_only_depth_control_disabled'
            return
        if self.pilot.odometry is self.pilot.stage1_odometry:
            # Preserve the observed coordinate frame at the stage boundary.
            # Do not change ring/landing policies or reset the aircraft origin.
            self.pilot.odometry = self.pilot.stage1_odometry.seed_depth_odometry()
        ring_stage = self.pilot.ring_stage_active()
        self.pilot.odometry.quality_checks = not ring_stage
        depth = np.asarray(depth, float)
        h, w = depth.shape
        boxes = []
        # Exclude a generous vehicle region, not just its moving flag pixels.
        for identifier, marker in frame.markers.items():
            if identifier >= 23:
                lo, hi = marker.corners.min(axis=0), marker.corners.max(axis=0)
                boxes.append([lo[0]-55, lo[1]-25, hi[0]+55, 319])
        self.pilot.odometry.update(frame.gray, depth, frame.stamp, boxes)
        if not self.pilot.odometry.valid and not ring_stage:
            self.reason = self.pilot.odometry.reason
            return
        bands = {}
        for name, x1, x2 in [('front', .4, .6), ('left', .08, .32), ('right', .68, .92)]:
            block = depth[int(.38*h):int(.62*h), int(x1*w):int(x2*w)]
            valid = block[np.isfinite(block) & (block > 0)] if ring_stage else block[np.isfinite(block) & (block >= .15) & (block < 19.5)]
            if not ring_stage and len(valid) < block.size*.75:
                self.reject('mission_depth_band_invalid_'+name)
                return
            bands[name] = float(np.percentile(valid, 15)) if len(valid) else float('nan')
        self.pilot.observe(frame, depth, bands)
        self.latest = dict(stamp=frame.stamp, estimated_front_clearance_m=bands['front'], **bands)
        self.reason = ('mission_depth_and_visual_odometry_valid' if self.pilot.odometry.valid
                       else 'mission_depth_available_odometry_unavailable')

    # 처리: 추정 상태·유효 여부·근거를 기록용 자료로 반환
    def telemetry(self):
        geometry = {key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
                    for key, value in (self.latest or {}).items()}
        return dict(valid=self.latest is not None, reason=self.reason,
                    source=(self.latest or {}).get('geometry_source', 'learned_monocular_depth'),
                    **geometry, measured_accuracy_m=None, ground_range_m=None)


# 파트: 1·2·3단계의 비행 명령과 안전 종료 결정
class FrontVisionMission:
    # 처리: 이 클래스의 상태·설정·관측 이력 초기화
    def __init__(self, marker_size=.6):
        self.phase = 'ACQUIRE'
        self.odometry = None
        self.stage1_odometry = None
        self.route_marker_size = float(marker_size)
        self.depth = MissionDepth(self)
        self.matrix = None
        self.distortion = np.zeros(5)
        self.camera_offset = np.array([.035, 0., .042425])
        self.frame = None
        self.markers = {}
        self.ring = None
        self.visited = set()
        self.active_marker = None
        self.crossed = 0
        self.ring_goal = None
        self.ring_obstacle_avoidance = {}
        self.tof_sensor = None
        self.tof_transport_fresh = False
        self.ring_tof = {}
        self.truck_tof = dict(over_deck=False, last_stamp=None, last_height=None,
                              near_samples=0, last_near_stamp=None)
        self.entered = None
        self.started = None
        self.last_time = None
        self.previous_command = np.zeros(4)
        self.stage1_forward_ramp_time = 0.
        self.image_pid = ImageCenterPID()
        # Ring-only image feedback; marker/vehicle following keeps its gains.
        self.ring_image_pid = ImageCenterPID()
        self.ring_image_pid.kp = np.array([.90, .85])
        self.ring_image_pid.ki = np.array([.010, .015])
        self.ring_image_pid.kd = np.array([.16, .20])
        self.cruise = 1.2
        self.stage1_height = 1.5
        self.stage1_altitude_control = {}
        self.marker_pose_lost_since = None
        self.marker_search_status = {}
        self.takeoff_in_progress = False
        self.reason = 'waiting_for_front_vision'
        self.events = []
        self.last_progress = None
        self.progress_stamp = None
        self.truck_integral = np.zeros(2)
        self.truck_observation = None
        self.truck_relative_velocity = None
        self.truck_status = {}
        self.landing_attempted = False
        self.landing_confirmed = False
        self.marker_depth_rejections = {}
        self.stage1_action = 'waiting'
        self.route_marker_memory = {}
        self.marker_memory_rejections = {}
        self.marker_memory_navigation = None

        self.route_id_votes = {}
        self.confirmed_route_ids = {}
        self.last_route_id_frame = None
        self.target_marker_id = 1
        self.target_marker_activated_stamp = -1.
        self.target_visit_status = {}
        self.route_completion_order = []
        self.stage1_last_target_heading = {}
        self.stage1_search_id = None
        self.stage1_search_started = None
        self.stage1_forward_observation = {}
        self.stage1_stand_off = 3.5
        self.stage1_search_yaw = None

    # 처리: 현재 목표 ID를 여러 새 영상에서 확인
    def confirm_current_target(self, frame):
        """Use the current decoded marker immediately."""
        return self.target_marker_id in frame.decoded_marker_ids

    # 처리: 마커 ID·거리·정렬 조건으로 도착 판단
    def confirm_target_vicinity(self, frame, target_confirmed):
        """Visit in order once the current marker is centered and in range."""
        identifier = self.target_marker_id
        if identifier is None:
            return False
        area = None
        bearing = None
        distance = None
        target = frame.markers.get(identifier)
        observation = self.markers.get(identifier)
        if target is not None and observation is not None:
            area = float(target.area_fraction)
            body = observation['body']
            distance = float(np.linalg.norm(body[:2]))
            bearing = math.atan2(body[1], body[0])
        centered = target is not None and np.max(np.abs(target.corners.mean(axis=0)-self.matrix[:2, 2])) <= 10.
        arrived = (self.target_visit_status.get('marker_id') == identifier
                   and self.target_visit_status.get('reached_standoff', False))
        arrived = bool(arrived or distance is not None and distance <= self.stage1_stand_off)
        nearby = bool(target_confirmed and centered and frame.stamp > self.target_marker_activated_stamp
                      and distance is not None and arrived)
        self.target_visit_status = dict(marker_id=identifier,
            estimated_horizontal_distance_m=distance, bearing_error_rad=bearing,
            image_area_fraction=area, preferred_distance_m=self.stage1_stand_off,
            arrival_distance_m=self.stage1_stand_off, reached_standoff=arrived,
            image_centered=bool(centered), center_tolerance_px=10.,
            nearby_frames=int(nearby),
            source=None if observation is None else observation['source'],
            metric_distance_required=True,
            metric_accuracy_verified=False, fixed_dwell_seconds=0.)
        return nearby

    # 처리: 새로 읽힌 미션 마커 ID와 확인 이력 저장
    def record_route_ids(self, frame, now):
        """Record decoded IDs independently of learned metric-depth quality."""
        if (frame is None or not 0 <= now-frame.stamp <= .3
                or self.last_route_id_frame == frame.stamp):
            return
        if self.last_route_id_frame is not None and frame.stamp <= self.last_route_id_frame:
            return
        self.last_route_id_frame = frame.stamp
        for identifier in range(1, 7):
            old_stamp, count = self.route_id_votes.get(identifier, (-1., 0))
            if identifier in frame.tracked_marker_ids:
                continue
            if identifier not in frame.decoded_marker_ids:
                self.route_id_votes[identifier] = (frame.stamp, 0)
                continue
            count = count+1 if 0 < frame.stamp-old_stamp <= .3 else 1
            self.route_id_votes[identifier] = (frame.stamp, count)
            if count >= 3:
                record = self.confirmed_route_ids.setdefault(identifier, dict(
                    first_confirmed_seconds=float(frame.stamp),
                    source='decoded_front_ArUco_ID_three_distinct_frames'))
                record.update(last_confirmed_seconds=float(frame.stamp), confirming_frames=count)
                # Remember directions of ALL reliably seen IDs, not just the
                # current one, so the next target can be brought back into view.
                if (self.odometry is not None and self.odometry.valid
                        and self.odometry.stamp == frame.stamp):
                    ray = self.odometry.rotation @ self.point(frame.markers[identifier].corners.mean(axis=0), 1.)
                    self.stage1_last_target_heading[identifier] = (float(frame.stamp), math.atan2(ray[1], ray[0]))

    # 처리: 현재 단계가 마커 비행 구간인지 확인
    def stage1_active(self):
        return self.phase in ('ACQUIRE', 'S1_TAKEOFF', 'S1_MARKERS')

    # 처리: 깊이 보완 없이 1단계 마커 위치 계산 수행
    def update_stage1_primary(self, frame):
        if not self.stage1_active():
            return False
        self.depth.latest = None
        self.frame, self.markers, self.ring = frame, {}, None
        if self.stage1_odometry is None or frame is None:
            if self.stage1_odometry is not None:
                self.stage1_odometry.valid = False
            self.depth.reason = 'waiting_for_stage1_image_and_calibration'
            return False
        self.odometry = self.stage1_odometry
        primary_valid = self.odometry.update_markers(frame)
        previous_ids = set(self.route_marker_memory)
        self.route_marker_memory = self.odometry.memory()
        self.marker_memory_rejections = self.odometry.rejections.copy()
        for identifier in set(self.route_marker_memory)-previous_ids:
            self.events.append(dict(stamp=frame.stamp, event='route_marker_remembered',
                                    marker_id=identifier, source='observed_OpenCV_PnP_landmark'))
        if primary_valid:
            self.observe_stage1(frame)
        else:
            self.depth.reason = self.odometry.reason
        return primary_valid

    # 처리: PnP가 성립한 마커의 상대 위치·거리만 제어에 연결
    def observe_stage1(self, frame):
        """Only measured marker corners and known-size PnP supply geometry."""
        self.frame, self.markers, self.ring = frame, {}, None
        self.marker_depth_rejections = self.stage1_odometry.rejections.copy()
        for identifier, target in frame.markers.items():
            if identifier not in range(1, 7):
                continue
            observation = self.stage1_odometry.observations.get(identifier)
            if observation is not None:
                body, source = observation['body'].copy(), observation['source']
            else:
                continue
            self.markers[identifier] = dict(target=target, body=body,
                position=self.odometry.position+self.odometry.rotation @ body, source=source)
        bands = dict(front=float('nan'), left=float('nan'), right=float('nan'))
        # Marker ranges do not establish obstacle-free space.
        central = [row['body'][0] for row in self.markers.values()
                   if abs(math.atan2(row['body'][1], row['body'][0])) < .35]
        if central:
            bands['front'] = float(min(central))
        self.depth.latest = dict(stamp=frame.stamp, estimated_front_clearance_m=bands['front'],
            geometry_source=self.odometry.source,
            clearance_source='observed_marker_PnP_ranges_only', **bands)
        self.depth.reason = self.odometry.reason

    # 처리: 현재 목표 마커를 향한 정렬·전진 명령 계산
    def observe_markers_while_forward(self, cmd, identifier, frame, bands, yaw, now):
        """Follow current PnP, remember the next ID's position across occlusion."""
        cmd[:2] = 0.
        self.marker_memory_navigation = None
        if identifier is None:
            self.stage1_action = 'all_route_IDs_recorded'
            return cmd
        heading = yaw
        source = 'bounded_stationary_image_search'
        area = None
        distance = None
        visible = identifier in frame.markers
        if visible:
            target = frame.markers[identifier]
            ray = self.point(target.corners.mean(axis=0), 1.)
            heading = yaw+math.atan2(ray[1], ray[0])
            self.stage1_last_target_heading[identifier] = (float(now), heading)
            source = 'current_target_image_bearing'
            area = float(target.area_fraction)
            observation = self.markers.get(identifier)
            if observation is not None:
                distance = float(np.linalg.norm(observation['body'][:2]))
                source = observation['source']
            if self.stage1_search_id is not None and self.confirm_current_target(frame):
                self.last_progress, self.progress_stamp = self.odometry.position.copy(), now
                self.events.append(dict(stamp=now, event='route_marker_reacquired', marker_id=identifier))
                self.stage1_search_id = None
        else:
            if self.stage1_search_id != identifier:
                self.stage1_search_id = identifier
                self.stage1_search_started = now
                self.stage1_search_yaw = yaw
            elapsed = now-self.stage1_search_started
            memory = self.route_marker_memory.get(identifier)
            if memory is not None and memory['valid']:
                delta = np.asarray(memory['position_m'])-self.odometry.position
                heading = math.atan2(delta[1], delta[0])
                distance = float(np.linalg.norm(delta[:2]))
                source = 'remembered_PnP_landmark_reacquisition'
                self.marker_memory_navigation = dict(marker_id=identifier,
                    goal_m=memory['position_m'], mode='face_remembered_position_reacquire_before_translation')
            else:
                # No measured position yet: the decoded ID's past image bearing
                # is only a bounded search hint. No arbitrary 15/60 s expiry.
                recent = self.stage1_last_target_heading.get(identifier)
                hint = 0. if recent is None else float(np.clip(angle(recent[1]-self.stage1_search_yaw),
                    -math.radians(50), math.radians(50)))
                direction = -1. if hint < 0. else 1.
                offset = hint+direction*math.radians(50)*math.sin(elapsed*.2)
                heading = self.stage1_search_yaw+offset
        heading_error = angle(heading-yaw)
        yaw_limit = .30 if source == 'remembered_PnP_landmark_reacquisition' else .25
        cmd[3] = float(np.clip(heading_error*.8, -yaw_limit, yaw_limit))
        if visible:
            speed = 0. if self.target_visit_status.get('reached_standoff', False) else .50
            cmd = self.center_on_image(target.corners.mean(axis=0), speed, frame.stamp,
                                       ('marker', identifier), align_before_forward=False)
        else:
            self.image_pid.reset()
        speed = float(cmd[0])
        self.stage1_action = ('stage1_follow_visible_target' if speed > 0. else
                              'stage1_align_marker_with_camera_center') if visible else 'stage1_stationary_search_next_ID'
        self.stage1_forward_observation = dict(marker_id=identifier, target_visible=visible,
            bearing_source=source, forward_speed_m_s=float(speed), reverse_enabled=False,
            inplace_scan_enabled=not visible, scan_half_angle_deg=50.,
            image_area_fraction=area, preferred_distance_m=self.stage1_stand_off,
            estimated_marker_distance_m=distance, depth_waypoint_navigation_used=False,
            yaw_rate_limit_rad_s=yaw_limit, heading_error_rad=heading_error,
            source=source, metric_accuracy_verified=False)
        return cmd

    # 처리: CameraInfo와 카메라 장착 오프셋 연결
    def configure_camera(self, info, offset=(.035, 0., .042425)):
        self.matrix = np.asarray(info.k).reshape(3, 3)
        self.distortion = np.asarray(info.d, dtype=float)
        self.camera_offset = np.asarray(offset, dtype=float)
        self.odometry = FrontDepthOdometry(self.matrix)
        self.stage1_odometry = Stage1MarkerOdometry(self.matrix, info.d, self.route_marker_size, offset)

    # 처리: 영상 픽셀과 깊이로 기체 좌표계의 광선 위치 계산
    def point(self, pixels, distance):
        u, v = pixels
        return np.array([distance, -(u-self.matrix[0, 2])*distance/self.matrix[0, 0],
                         -(v-self.matrix[1, 2])*distance/self.matrix[1, 1]])

    # 처리: 현재 영상 관측을 제어기 상태에 연결
    def observe(self, frame, depth, bands):
        self.frame = frame
        self.markers = {}
        self.marker_depth_rejections = {}
        h, w = depth.shape
        for identifier, marker in frame.markers.items():
            points = marker.corners.mean(axis=0)+.65*(marker.corners-marker.corners.mean(axis=0))
            points = np.rint(points*[w/320, h/320]).astype(np.int32)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, points, 1)
            values = depth[(mask > 0) & np.isfinite(depth) & (depth > .15) & (depth < 19.5)]
            if values.size < 6:
                self.marker_depth_rejections[identifier] = 'insufficient_predicted_depth_pixels'
                continue
            distance = float(np.median(values))
            body = self.point(marker.corners.mean(axis=0), distance)
            self.markers[identifier] = dict(target=marker, body=body,
                position=self.odometry.position+self.odometry.rotation @ body)
        self.ring = None
        if self.phase == 'S2_CROSS_RING':
            goal = self.ring_goal
            if goal.get('rim_pixels') is None:
                return
            # Refresh only an overlapping, non-shrinking box of the locked
            # ring. A smaller rear ring cannot replace it or steer the aircraft.
            old_box = goal.get('box')
            matched = None
            if old_box is not None:
                old_area = float(np.prod(old_box[2:]-old_box[:2]))
                best_overlap = .30
                for candidate in frame.visible_rings:
                    if candidate['box_stamp'] != frame.stamp or candidate['box_stamp'] <= goal['box_stamp']:
                        continue
                    box = candidate['box']
                    area = float(np.prod(box[2:]-box[:2]))
                    intersection = float(np.prod(np.maximum(0., np.minimum(box[2:], old_box[2:])-np.maximum(box[:2], old_box[:2]))))
                    overlap = intersection/(area+old_area-intersection)
                    if area >= .80*old_area and overlap > best_overlap:
                        matched, best_overlap = candidate, overlap
                # Prefer the full-image rim. A YOLO crop can cut the
                # ellipse and make its diameter jump during approach.
                best_score = float('inf')
                contours = cv2.findContours(cv2.Canny(frame.gray, 40, 100),
                    cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
                size = old_box[2:]-old_box[:2]
                for contour in contours:
                    if len(contour) < 20:
                        continue
                    (cx, cy), (a, b), rotation = cv2.fitEllipse(contour)
                    if not np.isfinite([cx,cy,a,b,rotation]).all() or min(a,b) <= 0 or min(a,b)/max(a,b) < .75:
                        continue
                    points = contour.reshape(-1,2).astype(float)-[cx,cy]
                    theta = math.radians(rotation)
                    local = points @ np.array([[math.cos(theta),-math.sin(theta)], [math.sin(theta),math.cos(theta)]])
                    if np.median(np.abs(np.linalg.norm(local/[a/2,b/2],axis=1)-1.)) > .05:
                        continue
                    # Recover the diameter from the observed arc even when
                    # part of the same ring extends beyond the image.
                    extent = np.sqrt([(a/2*math.cos(theta))**2+(b/2*math.sin(theta))**2,
                                      (a/2*math.sin(theta))**2+(b/2*math.cos(theta))**2])
                    lo, hi = np.array([cx,cy])-extent, np.array([cx,cy])+extent
                    scale = (hi-lo)/size
                    offset = np.linalg.norm((np.array([cx,cy])-goal['pixel_center'])/size)
                    if min(scale) < .75 or max(scale) > 1.6 or offset > .30:
                        continue
                    score = offset+float(np.linalg.norm(np.log(scale)))
                    if score < best_score:
                        best_score = score
                        matched = dict(box=np.r_[lo,hi].astype(float),box_stamp=frame.stamp,
                            box_center=np.array([cx,cy]),contour=contour)
            if matched is not None:
                pixels = matched['contour']
                if pixels is None:
                    x1, y1, x2, y2 = matched['box']
                    pixels = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])
                pixels = np.asarray(pixels, np.float32).reshape(-1, 2)
                goal.update(box=matched['box'].copy(), box_stamp=matched['box_stamp'],
                    pixel_center=matched['box_center'].copy(), gray=frame.gray.copy(),
                    rim_pixels=pixels[::3] if len(pixels) > 9 else pixels.copy())
                self.observe_ring_expansion(goal, pixels, frame.stamp)
            if goal['completion_ready']:
                return
            # Follow the captured rim until it fills most of the image. A
            # distant, noisy depth plane must not unlock the next ring early.
            # These pixels only confirm completion; they never steer the path.
            pixels = goal['rim_pixels']
            try:
                moved, ok, _ = cv2.calcOpticalFlowPyrLK(goal['gray'], frame.gray,
                    pixels.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
                if moved is None or ok is None:
                    return
                reverse, back_ok, _ = cv2.calcOpticalFlowPyrLK(frame.gray, goal['gray'],
                    moved, None, winSize=(21, 21), maxLevel=3)
            except cv2.error:
                return
            if reverse is None or back_ok is None:
                return
            moved = moved.reshape(-1, 2)
            good = ok.ravel().astype(bool) & np.isfinite(moved).all(axis=1)
            good &= back_ok.ravel().astype(bool) & (np.linalg.norm(reverse.reshape(-1, 2)-pixels, axis=1) < 1.)
            if np.count_nonzero(good) < 3:
                return
            goal['pixel_center'] += np.mean(moved[good]-pixels[good], axis=0)
            goal['rim_pixels'], goal['gray'] = moved[good], frame.gray.copy()
            span = (np.min(goal['box'][2:]-goal['box'][:2]) if goal.get('box') is not None
                    else np.min(np.ptp(moved[good], axis=0)))
            if (span < .7*min(frame.gray.shape)
                    or not self.odometry.valid):
                return
            measured = self.ring_plane(depth, dict(contour=moved[good],
                box_center=goal['pixel_center'], box_stamp=goal['box_stamp']))
            if measured is not None:
                goal['center'] = measured['position']
                goal['normal'] = goal['rotation'][:, 0].copy()
                goal['completion_ready'] = True
                goal['previous_position'] = self.odometry.position.copy()
            return
        if frame.ring is not None:
            self.ring = self.ring_plane(depth, frame.ring)

    def observe_ring_expansion(self, goal, pixels, stamp):
        """Predict passage from measured expansion; never replace missing pose."""
        if len(pixels) < 20:
            return
        (cx, cy), (a, b), rotation = cv2.fitEllipse(pixels.reshape(-1, 1, 2))
        if not np.isfinite([cx,cy,a,b,rotation]).all() or min(a,b) <= 0 or min(a,b)/max(a,b) < .75:
            return
        theta = math.radians(rotation)
        local = (pixels-[cx,cy]) @ np.array([[math.cos(theta),-math.sin(theta)], [math.sin(theta),math.cos(theta)]])
        if np.median(np.abs(np.linalg.norm(local/[a/2,b/2],axis=1)-1.)) > .05:
            return
        diameter = min(a,b)
        goal['last_rim_observation'] = stamp
        goal['visual_filled'] = goal.get('visual_filled', False) or diameter >= .95*320
        # Near the ring the command is constant. Inverse diameter is then
        # proportional to remaining distance; its zero gives camera passage.
        if diameter < .70*320 or not .19 <= self.previous_command[0] <= .21:
            return
        samples = [(t,d) for t,d in goal.get('expansion_samples', []) if 0 < stamp-t <= 4.]
        samples.append((stamp, 1./diameter))
        goal['expansion_samples'] = samples
        if len(samples) < 6 or stamp-samples[0][0] < .8:
            return
        values = np.asarray(samples)
        times = values[:,0]-stamp
        # Clipped arcs can produce an occasional wrong diameter. Fit the
        # consistent expansion sequence instead of averaging those jumps.
        i,j = np.triu_indices(len(times),1)
        separated = times[j]-times[i] >= .4
        i,j = i[separated],j[separated]
        slopes = (values[j,1]-values[i,1])/(times[j]-times[i])
        growing = slopes < 0
        slopes,i = slopes[growing],i[growing]
        if not len(slopes):
            return
        intercepts = values[i,1]-slopes*times[i]
        fits = slopes[:,None]*times+intercepts[:,None]
        inliers = np.abs(fits-values[:,1]) <= .015*values[:,1]
        good = inliers[np.argmax(inliers.sum(axis=1))]
        if np.count_nonzero(good) < 6 or np.ptp(times[good]) < .8:
            return
        times,values = times[good],values[good]
        slope, intercept = np.linalg.lstsq(np.c_[times, np.ones(len(times))], values[:,1], rcond=None)[0]
        total = float(np.sum((values[:,1]-values[:,1].mean())**2))
        residual = float(np.sum((values[:,1]-(slope*times+intercept))**2))
        if slope < 0 and total > 0 and residual/total <= .10:
            remaining = -intercept/slope
            if 0 < remaining <= 10.:
                prediction = float(stamp+remaining)
                previous = goal.get('visual_pass_time')
                if previous is None or abs(prediction-previous) <= .6:
                    goal['visual_pass_time'] = prediction

    def ring_plane(self, depth, target):
        """Completion geometry from observed rim pixels, never a start permission."""
        h, w = depth.shape
        # Only numerical prerequisites remain. Shape confidence, aperture
        # margins, range limits, residual limits and repeated votes are removed.
        contour = target['contour']
        if contour is None:
            x1, y1, x2, y2 = target['box']
            contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        else:
            contour = contour.reshape(-1, 2)
            if len(contour) > 9:
                contour = contour[::3]
        ix = np.clip(np.rint(contour[:, 0]*w/320).astype(int), 0, w-1)
        iy = np.clip(np.rint(contour[:, 1]*h/320).astype(int), 0, h-1)
        distances = depth[iy, ix]
        good = np.isfinite(distances) & (distances > 0)
        if np.count_nonzero(good) < 3:
            return  # A plane requires three measured points.
        points = np.array([self.point(pixel, d) for pixel, d in zip(contour[good], distances[good])])
        center = points.mean(axis=0)
        _, spread, vh = np.linalg.svd(points-center, full_matrices=False)
        if spread[1] <= np.finfo(float).eps:
            return  # Collinear points cannot define a plane.
        normal = vh[-1]
        if normal[0] < 0:
            normal = -normal
        ray = self.point(target['box_center'], 1.)
        denominator = float(ray @ normal)
        if abs(denominator) <= np.finfo(float).eps:
            return  # Do not divide by zero for a parallel viewing ray.
        center_body = ray*float(center @ normal)/denominator
        if not np.isfinite(center_body).all() or center_body[0] <= 0:
            return  # Only an observed center in front of the camera is a target.
        position = self.odometry.position+self.odometry.rotation @ center_body
        return dict(position=position, normal=self.odometry.rotation @ normal,
                         box_stamp=target['box_stamp'])

    # 처리: 미션 단계·전환 시간·전환 원인 기록
    def transition(self, phase, now, reason):
        self.phase, self.entered = phase, now
        self.reason = reason
        self.events.append(dict(phase=phase, stamp=now, reason=reason))

    # 처리: 목표와의 기체 기준 오차로 제한된 속도 계산
    def velocity_to(self, position, speed=.3):
        error = self.odometry.rotation.T @ (position-self.odometry.position)
        return np.clip(.65*error, [-speed, -speed, -.15], [speed, speed, .2])

    def center_on_image(self, center, forward_speed, stamp, key, *, align_before_forward=True, pid=None):
        """PID alignment; marker following advances while correcting its image."""
        error = np.asarray(center)-self.matrix[:2, 2]
        controller = self.image_pid if pid is None else pid
        limits = np.array([.35, .18]) if controller is self.ring_image_pid else np.array([.25, .22])
        lower = -limits if controller is self.ring_image_pid else np.array([-.25, -.10])
        correction = controller.update(center, self.matrix, stamp, key, lower, limits)
        if controller is self.ring_image_pid:
            correction *= np.clip(np.abs(error)/16., .60, 1.)
        speed = forward_speed
        if align_before_forward:
            speed *= float(np.clip((10.-np.max(np.abs(error)))/6., 0., 1.))
        return np.array([speed, 0., correction[1], correction[0]])

    def stage1_output(self, cmd, dt):
        """Ramp to constant marker speed with a gradual acceleration onset."""
        previous_speed = max(0., self.previous_command[0])
        if cmd[0] > previous_speed:
            # A 3-second S-curve reaches 0.50 m/s: .5 s increasing
            # acceleration, 2 s at .20 m/s², .5 s easing to cruise.
            self.stage1_forward_ramp_time = min(3., self.stage1_forward_ramp_time+dt)
            t = self.stage1_forward_ramp_time
            speed = (.20*t*t if t <= .5 else
                     .05+.20*(t-.5) if t <= 2.5 else .50-.20*(3.-t)**2)
            cmd[0] = min(cmd[0], max(previous_speed, speed))
        else:
            self.stage1_forward_ramp_time = 0.
        cmd[2:] = self.previous_command[2:]+np.clip(cmd[2:]-self.previous_command[2:],
                                                 -np.array([.25, .5])*dt, np.array([.25, .5])*dt)
        self.previous_command = cmd.copy()
        return cmd

    def search_without_marker_pose(self, frame, now, dt):
        """Image centering remains available without a metric position."""
        if frame is None or not 0 <= now-frame.stamp <= .3:
            self.image_pid.reset()
            self.previous_command[:] = 0.
            self.stage1_forward_ramp_time = 0.
            return np.zeros(4), 'waiting_for_current_marker_image'
        if self.marker_pose_lost_since is None:
            self.marker_pose_lost_since = now
        age = now-self.marker_pose_lost_since
        target = frame.markers.get(self.target_marker_id)
        if target is None:
            self.image_pid.reset()
        speed = 0. if self.target_visit_status.get('reached_standoff', False) else .50
        command = (self.center_on_image(target.corners.mean(axis=0), speed, frame.stamp,
                   ('marker', self.target_marker_id), align_before_forward=False) if target is not None
                   else np.array([0., 0., 0., .25 if age % 16. < 8. else -.25]))
        command = self.stage1_output(command, dt)
        self.marker_search_status = dict(active=True, target_marker_id=self.target_marker_id,
            target_visible=target is not None, pose_valid=False, altitude_available=False)
        self.stage1_altitude_control = dict(valid=False, target_height_m=None,
            commanded_vertical_speed_m_s=float(command[2]),
            source='marker_image_center' if target is not None else 'altitude_unavailable')
        self.stage1_action = 'marker_image_centering_without_metric_pose' if target is not None else 'marker_image_search'
        return command, self.stage1_action

    def ring_stage_active(self):
        return self.phase in ('S2_FIND_RING', 'S2_ALIGN_RING', 'S2_CROSS_RING')

    def tof_reading(self, now):
        """Fresh measured surface height, tilt-corrected with sensor attitude."""
        if self.tof_sensor is None or not self.tof_transport_fresh:
            return None
        data = self.tof_sensor.range_telemetry(now)
        surface = self.tof_sensor.surface(now)
        if (not data['valid'] or surface is None
                or surface['stamp'] != data['capture_stamp']):
            return None
        return dict(stamp=data['capture_stamp'], height=float(surface['height']), range=float(surface['range']))

    def verify_ring_tof(self, now):
        """Ground -> shorter lower-rim range -> restored ground range."""
        status = self.ring_tof
        if not status:
            status.update(state='WAIT_BASELINE', baseline_m=None, last_stamp=None,
                          last_height_m=None, short_samples=0, edge_seen=False, clear_since=None)
        measured = self.tof_reading(now)
        status['sample_valid'] = measured is not None
        if status['state'] == 'PASSED':
            return False
        if measured is None:
            status.update(short_samples=0, clear_since=None)
            if status['edge_seen']:
                status['state'] = 'OVER_LOWER_EDGE'
            return False
        stamp, height = measured['stamp'], measured['height']
        if status['last_stamp'] is not None and stamp <= status['last_stamp']:
            return False  # A repeated ToF sample cannot complete a passage.
        previous, previous_stamp = status['last_height_m'], status['last_stamp']
        status.update(last_stamp=stamp, last_height_m=height, surface_m=height)
        if status['baseline_m'] is None:
            status.update(baseline_m=height, state='SEEK_LOWER_EDGE')
            return False
        if previous_stamp is not None and stamp-previous_stamp > .20:
            status.update(short_samples=0, clear_since=None)
        drop = status['baseline_m']-height
        status['height_drop_m'] = drop
        if not status['edge_seen']:
            abrupt = previous is not None and previous-height >= .12 and stamp-previous_stamp <= .20
            if drop >= .15 and (abrupt or status['short_samples']):
                status['short_samples'] += 1
                if status['short_samples'] >= 2:
                    status.update(edge_seen=True, state='OVER_LOWER_EDGE')
            else:
                status['short_samples'] = 0
            return False
        if height >= status['baseline_m']-.06:
            if status['clear_since'] is None:
                status.update(clear_since=stamp, state='GROUND_RECOVERED')
            # Let the rear of the body clear the rim after the center beam exits.
            if stamp-status['clear_since'] >= .30:
                status.update(state='PASSED', passed_stamp=stamp)
                return True
        else:
            status.update(clear_since=None, state='OVER_LOWER_EDGE')
        return False

    def update_truck_tof(self, now):
        """A flag-aligned surface step establishes deck entry; close range backs touchdown."""
        status = self.truck_tof
        measured = self.tof_reading(now)
        status['sample_valid'] = measured is not None
        if measured is None:
            status['near_samples'] = 0
            return
        stamp, height = measured['stamp'], measured['height']
        if status['last_stamp'] is not None and stamp <= status['last_stamp']:
            return
        old_height, old_stamp = status['last_height'], status['last_stamp']
        status.update(last_stamp=stamp, last_height=height, surface_m=height)
        observed = self.truck_observation
        flag_aligned = (observed is not None and 0 <= now-observed['stamp'] <= .25
            and abs(observed['body'][0]-1.50) < .35 and abs(observed['body'][1]) < .35)
        if old_stamp is not None and 0 < stamp-old_stamp <= .20:
            jump = old_height-height
            status['surface_step_m'] = jump
            if jump >= .20 and flag_aligned:
                status.update(over_deck=True, deck_entry_stamp=stamp, near_samples=0,
                              last_near_stamp=None, state='OVER_VEHICLE')
            near_recent = (status['last_near_stamp'] is not None
                           and 0 <= stamp-status['last_near_stamp'] <= .70)
            # At contact the sensor can be below its minimum range. Keep the
            # recent near-surface evidence, never substitute the new long range.
            if jump <= -.20 and not near_recent:
                status.update(over_deck=False, near_samples=0, state='LEFT_VEHICLE')
        else:
            status['near_samples'] = 0
        if status['over_deck'] and measured['range'] <= .06:
            status['near_samples'] += 1
            if status['near_samples'] >= 2:
                status.update(last_near_stamp=stamp, state='NEAR_DECK')
        else:
            status['near_samples'] = 0

    def ring_command(self, frame, now, dt):
        """Image alignment alone starts a forced passage along the captured path."""
        if self.odometry is None or self.matrix is None:
            return self.previous_command.copy(), 'ring_waiting_for_coordinate_reference'
        p, r = self.odometry.position, self.odometry.rotation
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return self.previous_command.copy(), 'ring_position_not_computable'
        yaw = math.atan2(r[1, 0], r[0, 0])
        cmd = np.zeros(4)
        self.ring_obstacle_avoidance = {}
        if self.phase != 'S2_CROSS_RING':
            if self.crossed:
                bands = self.depth.latest
                fresh = (bands is not None and 0 <= now-bands['stamp'] <= .3
                         and np.isfinite([bands['front'], bands['left'], bands['right']]).all())
                if not fresh or bands['front'] < 1.35:
                    # Only between passages. Crossing never enters this branch.
                    side = 0
                    if fresh:
                        side = 1 if bands['left'] >= bands['right'] else -1
                        if bands['left' if side == 1 else 'right'] <= 1.2:
                            side = 0
                    self.ring_image_pid.reset()
                    self.ring_obstacle_avoidance = dict(active=True, depth_valid=bool(fresh),
                        front_m=None if not fresh else bands['front'], lateral_direction=side)
                    self.previous_command[[0, 2, 3]] = 0.
                    if side*self.previous_command[1] <= 0:
                        self.previous_command[1] = 0.
                    cmd[1] = side*.08
                    # Stop immediately if no measured escape direction exists.
                    return self.ring_output(cmd, dt), 'between_rings_obstacle_avoidance' if fresh else 'between_rings_waiting_for_depth'
            target = None if frame is None else frame.ring
            # A pre-passage detection belongs to the completed ring, not the next one.
            if (target is not None and self.phase == 'S2_FIND_RING' and self.crossed
                    and target['box_stamp'] <= self.entered):
                target = None
            if target is not None:
                if self.phase != 'S2_ALIGN_RING':
                    self.transition('S2_ALIGN_RING', now, 'align_ring_box_to_camera_center')
                cmd = self.center_on_image(target['box_center'], 0.,
                    target['box_stamp'], ('ring_yaw', self.crossed),
                    align_before_forward=False, pid=self.ring_image_pid)
                if np.max(np.abs(target['box_center']-self.matrix[:2, 2])) <= 2.:
                    measured = self.ring
                    if measured is not None and measured['box_stamp'] != target['box_stamp']:
                        measured = None
                    center = None if measured is None else measured['position'].copy()
                    normal = None if measured is None else r[:, 0].copy()
                    pixels = target.get('contour')
                    if pixels is None and 'box' in target:
                        x1, y1, x2, y2 = target['box']
                        pixels = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])
                    if pixels is not None:
                        pixels = np.asarray(pixels, np.float32).reshape(-1, 2)
                        if len(pixels) > 9:
                            pixels = pixels[::3]
                    self.ring_goal = dict(start_position=p.copy(), rotation=r.copy(),
                        center=center, normal=normal, previous_position=p.copy(), plane_crossed=False,
                        completion_ready=bool(measured is not None and self.odometry.valid and pixels is not None
                            and np.min(np.ptp(pixels, axis=0)) >= .7*min(frame.gray.shape)),
                        gray=frame.gray.copy(), pixel_center=np.asarray(target['box_center'], float).copy(),
                        rim_pixels=None if pixels is None else pixels.copy(),
                        box=None if 'box' not in target else target['box'].copy(),
                        box_stamp=target['box_stamp'])
                    self.ring_tof = {}
                    self.ring_image_pid.reset()
                    self.previous_command[1:] = 0.
                    self.transition('S2_CROSS_RING', now, 'image_centers_aligned_forced_passage')
            elif (self.phase == 'S2_ALIGN_RING' and self.ring_image_pid.stamp is not None
                    and 0 <= now-self.ring_image_pid.stamp <= .20):
                self.previous_command[:] = 0.
                return cmd, 'ring_brief_detection_gap_hold'
            else:
                self.ring_image_pid.reset()
                cmd[3] = np.clip(angle(.65*math.sin((now-self.entered)*.35)-yaw), -.12, .12)
        if self.phase == 'S2_CROSS_RING':
            goal = self.ring_goal
            # The motor controller accepts body-frame velocity. Keep the
            # aligned aircraft pointing forward; drifting visual coordinates
            # must not introduce sideways, vertical or turning commands.
            #링 통과속도
            cmd[0] = .40
            slow_speed = min(.20, cmd[0])
            if goal['completion_ready']:
                cmd[0] = slow_speed
            elif goal.get('rim_pixels') is not None:
                span = (np.min(goal['box'][2:]-goal['box'][:2]) if goal.get('box') is not None
                        else np.min(np.ptp(goal['rim_pixels'].reshape(-1, 2), axis=0)))/320.
                cmd[0] *= np.clip((.70-span)/.30, slow_speed/cmd[0], 1.)
            self.previous_command[1:] = 0.
            # Plane/visual estimates still shape near-ring speed and diagnostics;
            # only the measured ToF height profile unlocks the next target.
            passed = self.verify_ring_tof(now)
            if passed:
                self.crossed += 1
                self.ring_goal = None
                self.ring = None
                self.ring_image_pid.reset()
                cmd[:] = 0.
                self.previous_command[:] = 0.
                self.transition('S3_FIND_TRUCK' if self.crossed == 4 else 'S2_FIND_RING', now,
                    'ToF_lower_edge_then_ground_recovered_select_next')
            if self.odometry.valid:
                goal['previous_position'] = p.copy()
        return self.ring_output(cmd, dt), self.phase.lower()

    def ring_output(self, cmd, dt):
        """Gentle ring commands; a requested forward stop is immediate."""
        limits = np.array([.10, .10, .18, .40] if self.phase == 'S2_ALIGN_RING' else [.10, .10, .08, .15])*dt
        if self.phase == 'S2_CROSS_RING' and cmd[0] < self.previous_command[0]:
            limits[0] = .25*dt
        out = self.previous_command+np.clip(cmd-self.previous_command, -limits, limits)
        if cmd[0] == 0.:
            out[0] = 0.
        self.previous_command = out
        return out

    def truck_stage_active(self):
        return self.phase in ('S3_FIND_TRUCK', 'S3_FOLLOW_TRUCK', 'S3_DESCEND')

    def observe_truck(self, frame, now):
        """Moving flag pose, never a stationary world-odometry landmark."""
        if (frame is None or self.matrix is None or not 0 <= now-frame.stamp <= .25
                or 23 not in frame.decoded_marker_ids):
            return None
        old = self.truck_observation
        if old is not None and frame.stamp <= old['stamp']:
            return old if frame.stamp == old['stamp'] else None
        # Existing ID 23: 0.40 m black ArUco square, excluding its white border.
        objects = np.array([[-.2, .2, 0.], [.2, .2, 0.], [.2, -.2, 0.], [-.2, -.2, 0.]])
        poses = square_pose_candidates(objects, frame.markers[23].corners, self.matrix, self.distortion)
        candidates = []
        for pose in poses:
            rotation = pose['rotation_camera'].T @ CAMERA_FROM_BODY
            position = -pose['rotation_camera'].T @ pose['translation_camera']-rotation @ self.camera_offset
            body = CAMERA_FROM_BODY.T @ pose['translation_camera']+self.camera_offset
            candidates.append(dict(stamp=frame.stamp, position=position, rotation=rotation,
                                   body=body, reprojection_px=pose['reprojection_px']))
        if not candidates:
            return None
        fresh_pair = old is not None and 0 < frame.stamp-old['stamp'] <= .25
        observed = (min(candidates, key=lambda row: np.linalg.norm(row['position']-old['position']))
                    if fresh_pair else candidates[0])
        if fresh_pair:
            elapsed = frame.stamp-old['stamp']
            velocity = (observed['position']-old['position'])/elapsed
            alpha = elapsed/(.20+elapsed)
            self.truck_relative_velocity = (velocity if self.truck_relative_velocity is None else
                self.truck_relative_velocity+alpha*(velocity-self.truck_relative_velocity))
        else:
            self.truck_relative_velocity = None
        self.truck_observation = observed
        return observed

    def truck_command(self, frame, now, dt):
        """Keep flag standoff in both directions and descend as soon as speeds match."""
        self.frame, self.markers, self.ring = frame, {}, None
        self.depth.reject('truck_relative_flag_PnP_depth_display_only')
        observed = self.observe_truck(frame, now)
        self.update_truck_tof(now)
        self.truck_status = dict(source='existing_flag_23_relative_PnP', standoff_m=1.50,
                                 speed_match_threshold_m_s=.05, depth_control=False,
                                 over_vehicle_by_tof=self.truck_tof['over_deck'],
                                 tof_surface_m=self.truck_tof.get('surface_m'))
        if observed is None:
            self.truck_relative_velocity = None
            old = self.truck_observation
            age = None if old is None else now-old['stamp']
            # The front flag can clip at the top just before deck contact.
            # Briefly retain the measured approach; never invent a new pose.
            final_descent = (self.phase == 'S3_DESCEND' and old is not None
                and self.truck_tof['over_deck']
                and 0 <= age <= 1.5 and old['body'][2] >= .70
                and abs(old['body'][0]-1.50) < .20 and abs(old['body'][1]) < .20)
            cmd = self.previous_command.copy() if final_descent else np.zeros(4)
            if final_descent:
                cmd[2], cmd[3] = -.08, 0.
            elif self.phase == 'S3_FIND_TRUCK':
                cmd[3] = .18*math.cos((now-(now if self.entered is None else self.entered))*.35)
            self.previous_command = cmd
            self.truck_status.update(observation_age_s=age, observation_valid=False,
                final_descent_using_last_command=final_descent)
            return cmd, 'truck_final_descent' if final_descent else 'truck_flag_missing_hold'
        if self.phase == 'S3_FIND_TRUCK':
            self.transition('S3_FOLLOW_TRUCK', now, 'existing_truck_flag_decoded')
        body = observed['body']
        velocity = (None if self.truck_relative_velocity is None else
                    observed['rotation'].T @ self.truck_relative_velocity)
        speed = None if velocity is None else float(np.linalg.norm(velocity[:2]))
        if self.phase == 'S3_FOLLOW_TRUCK' and speed is not None and speed <= .05:
            self.landing_attempted = True
            self.transition('S3_DESCEND', now, 'relative_speed_matched_start_descent_without_dwell')
        error = body[:2]-[1.50, 0.]
        self.truck_integral = np.clip(self.truck_integral+.35*error*dt, [-.45, -.20], [.45, .20])
        horizontal = .8*error+self.truck_integral
        if velocity is not None:
            horizontal -= .6*velocity[:2]
        vertical = np.clip(.5*body[2], -.15, .15)
        if self.phase == 'S3_DESCEND':
            # Start immediately; close a remaining horizontal offset while descending.
            vertical = -.10*float(np.clip(1.-np.linalg.norm(error)/1.5, .15, 1.))
        forward = -observed['rotation'].T[:, 2]
        yaw = np.clip(math.atan2(forward[1], forward[0]), -.20, .20)
        cmd = np.array([*np.clip(horizontal, [-.60, -.30], [.60, .30]), vertical, yaw])
        limit = np.array([.60, .45, .35, .50])*dt
        cmd = self.previous_command+np.clip(cmd-self.previous_command, -limit, limit)
        if self.phase == 'S3_DESCEND':
            cmd[2] = min(0., cmd[2])
        self.previous_command = cmd
        self.truck_status.update(observation_valid=True, observation_age_s=now-observed['stamp'],
            flag_body_m=body.tolist(), horizontal_error_m=error.tolist(),
            relative_velocity_body_m_s=None if velocity is None else velocity.tolist(),
            relative_horizontal_speed_m_s=speed, reprojection_px=observed['reprojection_px'],
            descent_speed_m_s=float(cmd[2]), final_descent_using_last_command=False)
        return cmd, self.phase.lower()

    def confirm_truck_contact(self, collision_names, normals, now):
        """Recent ToF deck/near-surface evidence plus simulated physical contact."""
        if self.landing_confirmed:
            return True
        if now is not None:
            self.update_truck_tof(now)
        observed = self.truck_observation
        if (self.phase != 'S3_DESCEND' or observed is None or now is None
                or not 0 <= now-observed['stamp'] <= 1.5):
            return False
        deck = any(name.endswith('mission_vehicle::chassis::collision') for name in collision_names)
        body = observed['body']
        horizontal_surface = any(abs(n[2]) > .8*np.linalg.norm(n) and np.linalg.norm(n) > .1 for n in normals)
        near_stamp = self.truck_tof['last_near_stamp']
        tof_touchdown = (self.truck_tof['over_deck'] and near_stamp is not None
                         and 0 <= now-near_stamp <= .70)
        if not (deck and horizontal_surface and tof_touchdown and body[2] >= .65
                and abs(body[0]-1.50) < .35 and abs(body[1]) < .35):
            return False
        self.landing_confirmed = True
        self.truck_tof['state'] = 'LANDED'
        self.previous_command[:] = 0.
        self.transition('COMPLETE', now, 'ToF_deck_entry_near_surface_and_contact_confirmed')
        return True

    # 처리: 현재 단계와 유효 관측으로 이동 명령 결정
    def command(self, frame, now):
        dt = .05 if self.last_time is None else min(max(0., now-self.last_time), .1)
        self.last_time = now
        if self.phase in ('ABORT', 'COMPLETE'):
            self.image_pid.reset()
            self.ring_image_pid.reset()
            self.previous_command[:] = 0
            self.stage1_forward_ramp_time = 0.
            return np.zeros(4), self.reason
        if self.ring_stage_active():
            return self.ring_command(frame, now, dt)
        if self.truck_stage_active():
            return self.truck_command(frame, now, dt)
        self.record_route_ids(frame, now)
        if (self.stage1_active() and self.stage1_odometry is not None
                and not self.stage1_odometry.valid and self.phase != 'ACQUIRE'):
            return self.search_without_marker_pose(frame, now, dt)
        if self.marker_pose_lost_since is not None and self.stage1_active() and self.odometry.valid:
            self.last_progress, self.progress_stamp = self.odometry.position.copy(), now
            self.events.append(dict(stamp=now, event='marker_pose_reacquired',
                marker_ids=list(self.stage1_odometry.supported_ids),
                depth_position_used=False))
        self.marker_pose_lost_since = None
        self.marker_search_status = {}
        if (frame is None or self.depth.latest is None or not self.odometry.valid
                or not 0 <= now-frame.stamp <= .3 or frame.stamp != self.depth.latest['stamp']):
            self.previous_command[:] = 0
            self.image_pid.reset()
            self.ring_image_pid.reset()
            self.stage1_forward_ramp_time = 0.
            return np.zeros(4), 'mission_vision_invalid_stop'
        p, r = self.odometry.position, self.odometry.rotation
        yaw = math.atan2(r[1, 0], r[0, 0])
        if self.started is None:
            self.started = now
        stage1_height_mode = self.phase in ('ACQUIRE', 'S1_TAKEOFF', 'S1_MARKERS')
        height_goal = self.stage1_height if stage1_height_mode else self.cruise
        height = float(p[2])
        vertical_velocity = float(self.odometry.velocity[2])
        if stage1_height_mode:
            altitude = self.stage1_odometry.control_altitude(now)
            if altitude is None:
                self.previous_command[:] = 0
                self.stage1_forward_ramp_time = 0.
                return np.zeros(4), 'stage1_marker_height_reference_unavailable_hold'
            height, vertical_velocity = altitude
        vertical = (np.clip(.8*(height_goal-height)-.20*vertical_velocity, -.10, .22)
                    if stage1_height_mode else np.clip(.6*(height_goal-p[2]), -.12, .18))
        cmd = np.array([0., 0., vertical, np.clip(-yaw, -.3, .3)])
        bands = self.depth.latest
        target_confirmed = self.confirm_current_target(frame)
        if self.phase == 'ACQUIRE':
            if self.target_marker_id == 1 and target_confirmed:
                self.takeoff_in_progress = True
                self.transition('S1_MARKERS', now, 'route_ID_seen_start_ascent_and_mission_together')
                self.events.append(dict(stamp=now, event='concurrent_takeoff_marker_mission_started',
                                        estimated_height_target_m=self.stage1_height,
                                        wait_for_cruise_before_mission=False))
            else:
                return np.zeros(4), 'waiting_for_route_aruco'
        if self.phase == 'S1_TAKEOFF':
            # Compatibility for a caller selecting the old phase: mission and
            # climb now run concurrently rather than waiting at cruise height.
            self.takeoff_in_progress = True
            self.transition('S1_MARKERS', now, 'concurrent_ascent_legacy_phase_entry')
        # 1단계: 현재 목표 마커의 거리·정렬·방문 순서 처리
        if self.phase == 'S1_MARKERS':
            # IDs describe mission order only; no scene position/obstacle map.
            next_id = next((i for i in range(1, 7) if i not in self.visited), None)
            # Current marker ID and PnP range decide arrival. Old recognition
            # records guide the route but cannot complete unvisited IDs.
            nearby_confirmed = self.confirm_target_vicinity(frame, target_confirmed)
            if next_id is not None and next_id == self.target_marker_id and nearby_confirmed:
                self.visited.add(next_id)
                self.route_completion_order.append(next_id)
                self.events.append(dict(stamp=now, event='aruco_metric_visit_completed', marker_id=next_id,
                                        completion='fresh_ID_center_and_marker_PnP_range_only',
                                        dwell_seconds=0., metric_distance_required=True,
                                        image_area_fraction=self.target_visit_status['image_area_fraction'],
                                        estimated_horizontal_distance_m=self.target_visit_status['estimated_horizontal_distance_m'],
                                        geometry_source=self.target_visit_status['source'],
                                        nearby_frames=1,
                                        metric_accuracy_verified=False,
                                        fresh_frames_after_target_activation=True,
                                        completion_order=self.route_completion_order.copy()))
                self.active_marker = None
                self.marker_memory_navigation = None
                next_id = next((i for i in range(1, 7) if i not in self.visited), None)
                self.target_marker_id = next_id
                self.target_marker_activated_stamp = frame.stamp
                self.target_visit_status = dict(marker_id=next_id, nearby_frames=0)
                self.stage1_search_id = None
                self.last_progress, self.progress_stamp = p.copy(), now
                if next_id is not None:
                    self.events.append(dict(stamp=now, event='next_route_marker_started',
                                            marker_id=next_id, wait_seconds=0.,
                                            ascent_in_progress=self.takeoff_in_progress))
            self.active_marker = next_id if next_id in frame.markers else None
            cmd = self.observe_markers_while_forward(cmd, next_id, frame, bands, yaw, now)
            if self.phase == 'ABORT':
                self.previous_command[:] = 0.
                return np.zeros(4), self.reason
            if len(self.visited) == 6:
                self.marker_memory_navigation = None
                self.previous_command[:2] = 0.
                cmd[:2] = 0.
                self.transition('S2_FIND_RING', now, 'all_six_markers_visited_in_order_PnP_primary')
        marker_centered = (self.phase == 'S1_MARKERS' and self.active_marker in frame.markers
                           and np.max(np.abs(frame.markers[self.active_marker].corners.mean(axis=0)-self.matrix[:2, 2])) <= 10.)
        if self.takeoff_in_progress and (height >= self.stage1_height-.10 or marker_centered):
            self.takeoff_in_progress = False
        if self.phase == 'S1_MARKERS':
            following_image = self.active_marker in frame.markers
            if not following_image and abs(self.stage1_height-height) < .05 and abs(vertical_velocity) < .08:
                cmd[2] = 0.
            cmd = self.stage1_output(cmd, dt)
            self.stage1_altitude_control = dict(target_height_m=None if following_image else self.stage1_height,
                estimated_height_m=float(height), estimated_vertical_speed_m_s=vertical_velocity,
                commanded_vertical_speed_m_s=float(cmd[2]),
                source='marker_image_center' if following_image else 'last_accepted_marker_PnP_height',
                reference=self.stage1_odometry.altitude_control_status.copy())
            self.stage1_forward_observation['forward_speed_m_s'] = float(cmd[0])
            return cmd, self.stage1_action
        if p[2] > 1.9:
            cmd[2] = min(0., cmd[2])
        if p[2] < .45:
            cmd[2] = max(0., cmd[2])
        if self.last_progress is None or np.linalg.norm(p-self.last_progress) > .12:
            self.last_progress, self.progress_stamp = p.copy(), now
        elif (now-self.progress_stamp > 25
              and not (self.phase == 'S1_MARKERS' and self.stage1_search_id is not None)):
            # Bounded stationary search has its own 45 s timeout.
            self.transition('ABORT', now, 'no_observed_progress')
            return np.zeros(4), self.reason
        # 출력 보정: 명령 변화율 제한으로 급격한 속도 변화 완화
        rates = np.array([.5, .5, .35, .5])
        limit = rates*dt
        out = self.previous_command+np.clip(cmd-self.previous_command, -limit, limit)
        if cmd[0] == 0.:
            out[0] = 0.
        self.previous_command = out
        return out, self.stage1_action if self.phase == 'S1_MARKERS' else self.phase.lower()

    # 처리: 추정 상태·유효 여부·근거를 기록용 자료로 반환
    def telemetry(self):
        return dict(phase=self.phase, rings_crossed_estimated=self.crossed,
                    ring_target_locked=self.phase == 'S2_CROSS_RING',
                    ring_flight_abort_checks_enabled=False,
                    ring_depth_stop_during_passage=False,
                    ring_between_passages_avoidance_enabled=self.crossed > 0 and self.phase in ('S2_FIND_RING', 'S2_ALIGN_RING'),
                    ring_obstacle_avoidance=self.ring_obstacle_avoidance.copy(),
                    ring_center_tolerance_px=2.,
                    ring_start_permission='image_center_alignment_only',
                    ring_crossing_motion='body_forward_only_no_depth_pose_steering',
                    ring_passage_confirmation='ToF_lower_edge_then_ground_recovery',
                    ring_tof=self.ring_tof.copy(), truck_tof=self.truck_tof.copy(),
                    ring_visual_pass_time=None if self.ring_goal is None else self.ring_goal.get('visual_pass_time'),
                    ring_saved_position_m=None if self.ring_goal is None else self.ring_goal['start_position'].tolist(),
                    ring_saved_rotation=None if self.ring_goal is None else self.ring_goal['rotation'].tolist(),
                    ring_completion_geometry_available=self.ring_goal is not None and self.ring_goal['center'] is not None,
                    ring_completion_ready=self.ring_goal is not None and self.ring_goal['completion_ready'],
                    aruco_completed_by_metric_visit=sorted(self.visited),
                    current_target_visit=self.target_visit_status,
                    route_completion_order=self.route_completion_order.copy(),
                    marker_pose_search=self.marker_search_status.copy(),
                    stage1_depth_position_fallback_enabled=False,
                    target_marker_confirmation_frames=self.route_id_votes.get(self.target_marker_id, (-1., 0))[1],
                    target_marker_activated_seconds=self.target_marker_activated_stamp,
                    early_seen_IDs_do_not_skip_sequence=True,
                    stage1_completion='sequential_fresh_ID_center_and_marker_PnP_range_only',
                    stage1_marker_size_m=self.route_marker_size,
                    stage1_current_geometry_sources={str(i): row.get('source', 'learned_depth')
                                                     for i, row in self.markers.items()},
                    takeoff_mode='concurrent_ascent_and_marker_mission',
                    takeoff_in_progress=self.takeoff_in_progress,
                    takeoff_estimated_height_target_m=self.stage1_height,
                    stage1_altitude_control=self.stage1_altitude_control,
                    confirmed_route_marker_ids=sorted(self.confirmed_route_ids),
                    route_ID_records={str(identifier): record for identifier, record in self.confirmed_route_ids.items()},
                    next_route_marker_id=next((i for i in range(1, 7) if i not in self.visited), None),
                    active_marker_id=self.active_marker, depth_usable_marker_ids=sorted(self.markers),
                    marker_depth_rejections=self.marker_depth_rejections,
                    remembered_marker_ids=sorted(self.route_marker_memory),
                    remembered_markers={str(identifier): memory
                        for identifier, memory in self.route_marker_memory.items()},
                    marker_memory_rejections=self.marker_memory_rejections,
                    marker_memory_navigation=self.marker_memory_navigation,
                    memory_persistence='current_flight_only_not_loaded_into_other_scenes',
                    stage1_action=self.stage1_action,
                    stage1_forward_observation=self.stage1_forward_observation,
                    stage1_search_mode='remembered_position_directed_reacquisition_otherwise_bounded_scan',
                    stage1_depth_usage='display_only_not_used_for_position_range_or_height',
                    stage1_policy='constant_forward_speed_with_marker_PID_alignment',
                    odometry=(self.odometry.telemetry()
                              if self.odometry and not self.truck_stage_active() and not self.landing_confirmed else None),
                    depth=self.depth.telemetry(), landing_confirmed=self.landing_confirmed,
                    landing_reference="existing_flag_and_downward_ToF", landing_attempted=self.landing_attempted,
                    landing_confirmation_source='ToF_deck_entry_near_surface_plus_simulation_contact',
                    truck=self.truck_status.copy(),
                    events=self.events[-15:], ground_truth_input=False, arrows_used=False)
