"""Optional Flow-deck assistance from measured images/range/IMU attitude.

The Gazebo camera/range pair is a sensor proxy, not a PMW3901 hardware driver.
No world pose, simulator odometry, learned depth or scene map is accepted.
"""
from collections import deque

import cv2
import numpy as np


# Downward optical right/down/forward <- body forward/left/up.
CAMERA_FROM_BODY = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
CAMERA_OFFSET = np.array([0., 0., -.020])  # Relative to the base_link/IMU.


class FlowDeckAssist:
    def __init__(self):
        self.matrix = None
        self.ranges = deque(maxlen=30)
        self.range_sample = None
        self.attitudes = deque(maxlen=60)
        self.previous = None
        self.velocity = None
        self.velocity_stamp = None
        self.reason = 'waiting_for_flow_sensors'
        self.hold_height = None
        self.hold_phase = None
        self.status = {}

    def configure(self, matrix):
        matrix = np.asarray(matrix, float).reshape(3, 3)
        if not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
            raise ValueError('Flow camera needs valid calibration')
        self.matrix = matrix

    def invalidate_flow(self, reason):
        self.previous, self.velocity, self.velocity_stamp = None, None, None
        self.reason = reason

    def update_range(self, stamp, distance):
        if not np.isfinite(stamp) or (self.ranges and stamp <= self.ranges[-1][0]):
            return
        # Conservative common range for original Flow deck and Flow deck v2.
        distance = float(distance)
        valid = np.isfinite(distance) and .02 <= distance < 2.
        self.ranges.append((stamp, distance if valid else None))
        self.range_sample = dict(stamp=stamp, distance_m=distance if np.isfinite(distance) else None,
                                 in_range=bool(valid))

    def range_telemetry(self, now, *, transport_fresh=True):
        """Raw downward ToF readout, independent of Flow texture/IMU quality."""
        sample = self.range_sample
        age = (None if sample is None or now is None or not np.isfinite(now)
               else float(now-sample['stamp']))
        state = ('WAITING' if sample is None else 'WAITING_CLOCK' if age is None else
                 'FUTURE' if age < 0 else 'STALE' if not transport_fresh or age > .15 else
                 'INVALID' if sample['distance_m'] is None else
                 'OUT_OF_RANGE' if not sample['in_range'] else 'LIVE')
        return dict(enabled=True, state=state, valid=state == 'LIVE',
                    distance_m=None if sample is None else sample['distance_m'],
                    capture_stamp=None if sample is None else sample['stamp'], age_s=age,
                    direction='downward_distance_to_surface', source='simulated_ToF_range_proxy',
                    control_min_m=.02, control_max_exclusive_m=2.)

    def update_attitude(self, stamp, quaternion):
        if not np.isfinite(stamp) or (self.attitudes and stamp <= self.attitudes[-1][0]):
            return
        q = np.asarray(quaternion, float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < .5:
            self.attitudes.append((stamp, None))
            return
        x, y, z, w = q/np.linalg.norm(q)
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        self.attitudes.append((stamp, rotation))

    def surface(self, stamp):
        if not self.ranges or not self.attitudes:
            return None
        distance = min(self.ranges, key=lambda row: abs(row[0]-stamp))
        attitude = min(self.attitudes, key=lambda row: abs(row[0]-stamp))
        if (max(abs(distance[0]-stamp), abs(attitude[0]-stamp)) > .075
                or distance[1] is None or attitude[1] is None):
            return None
        rotation = attitude[1]
        if rotation[2, 2] < np.cos(np.deg2rad(20.)):
            return None  # A steep/slanted surface is not reliable hover feedback.
        return dict(range=distance[1], rotation=rotation,
                    height=distance[1]*rotation[2, 2], stamp=distance[0])

    def update_image(self, gray, stamp):
        if self.previous is not None and stamp <= self.previous['stamp']:
            return  # Repeated frames never refresh velocity freshness.
        self.velocity_stamp = None
        measured = self.surface(stamp)
        if (self.matrix is None or not np.isfinite(stamp) or measured is None
                or gray.dtype != np.uint8 or gray.shape != (128, 128)):
            self.previous, self.velocity = None, None
            self.reason = 'flow_image_range_or_attitude_unavailable'
            return
        current = dict(measured, gray=gray.copy(), stamp=stamp)
        old, self.previous = self.previous, current
        if (old is None or not .015 <= stamp-old['stamp'] <= .20
                or abs(current['height']-old['height']) > .12 or measured['range'] < .10):
            self.velocity = None
            self.reason = 'flow_pair_or_surface_discontinuity'
            return
        points = cv2.goodFeaturesToTrack(old['gray'], 80, .02, 5, blockSize=5)
        if points is None or len(points) < 10:
            self.velocity = None
            self.reason = 'flow_insufficient_floor_texture'
            return
        moved, ok, _ = cv2.calcOpticalFlowPyrLK(old['gray'], gray, points, None,
                                             winSize=(15, 15), maxLevel=2)
        if moved is None or ok is None:
            self.velocity = None
            self.reason = 'flow_tracking_failed'
            return
        reverse, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, old['gray'], moved, None,
                                                     winSize=(15, 15), maxLevel=2)
        if reverse is None or back_ok is None:
            self.velocity = None
            self.reason = 'flow_reverse_tracking_failed'
            return
        good = (ok.ravel().astype(bool) & back_ok.ravel().astype(bool)
                & (np.linalg.norm(reverse-points, axis=2).ravel() < .75)
                & np.isfinite(moved.reshape(-1, 2)).all(axis=1)
                & (moved.reshape(-1, 2).min(axis=1) >= 2)
                & (moved.reshape(-1, 2).max(axis=1) <= 125))
        if np.count_nonzero(good) < 10:
            self.velocity = None
            self.reason = 'flow_insufficient_verified_tracks'
            return

        def surface_points(pixels, observation):
            rays = np.c_[pixels.reshape(-1, 2), np.ones(len(pixels))] @ np.linalg.inv(self.matrix).T
            normal = CAMERA_FROM_BODY @ -observation['rotation'][2]
            scale = observation['range']*normal[2]/(rays @ normal)
            return rays*scale[:, None]

        # Remove measured attitude change before interpreting pixel motion.
        old_to_new = current['rotation'].T @ old['rotation']
        camera_rotation = CAMERA_FROM_BODY @ old_to_new @ CAMERA_FROM_BODY.T
        before = surface_points(points[good], old)
        after = surface_points(moved[good], current)
        displacement = (before @ camera_rotation.T-after) @ CAMERA_FROM_BODY
        displacement += old_to_new @ CAMERA_OFFSET-CAMERA_OFFSET
        velocities = displacement/(stamp-old['stamp'])
        median = np.median(velocities, axis=0)
        inliers = np.linalg.norm(velocities-median, axis=1) < .12
        if (np.count_nonzero(inliers) < 10 or np.mean(inliers) < .65
                or not np.isfinite(median).all() or np.linalg.norm(median) > 2.):
            self.velocity = None
            self.reason = 'flow_inconsistent_surface_motion'
            return
        velocity = np.median(velocities[inliers], axis=0)
        alpha = (stamp-old['stamp'])/(.10+stamp-old['stamp'])
        self.velocity = velocity if self.velocity is None else self.velocity+alpha*(velocity-self.velocity)
        self.velocity_stamp = stamp
        self.reason = 'measured_downward_flow_range_and_attitude'

    def apply(self, command, phase, now, *, searching=False, landing_observed=False):
        """Only hover damping, search-height hold and near-surface soft descent."""
        out = np.asarray(command, float).copy()
        measured = self.surface(now)
        valid = (self.velocity_stamp is not None and 0 <= now-self.velocity_stamp <= .15
                 and measured is not None and self.previous is not None
                 and abs(measured['height']-self.previous['height']) <= .12)
        self.status = dict(enabled=True, velocity_valid=bool(valid), reason=self.reason,
            source='downward_image_and_range_with_IMU_rotation_compensation',
            velocity_body_m_s=self.velocity.tolist() if valid else None,
            surface_distance_m=None if measured is None else measured['range'],
            horizontal_assist=False, height_hold=False, touchdown_softening=False)
        stationary = (phase in ('S1_MARKERS', 'S2_FIND_RING', 'S2_ALIGN_RING')
                      and np.linalg.norm(out[:2]) < .015)
        if stationary and valid:
            out[:2] += np.clip(-.25*self.velocity[:2], -.05, .05)
            self.status['horizontal_assist'] = True
        hold = (searching and phase in ('S1_MARKERS', 'S2_FIND_RING')
                and abs(out[2]) < .02 and measured is not None)
        if hold:
            height = measured['height']
            if self.hold_phase != phase or self.hold_height is None or abs(height-self.hold_height) > .12:
                self.hold_height, self.hold_phase = height, phase
            # Hold only the locally observed surface distance, not world altitude.
            damping = .10*self.velocity[2] if valid else 0.
            out[2] = np.clip(.8*(self.hold_height-height)-damping, -.06, .06)
            self.status['height_hold'] = True
        else:
            self.hold_height, self.hold_phase = None, None
        # Ground optical flow is not truck-relative velocity; never brake its follow command.
        # Do not lengthen the existing bounded blind final descent after flag loss.
        if (phase == 'S3_DESCEND' and landing_observed and measured is not None
                and measured['range'] < .28 and out[2] < -.04):
            out[2] = -.04
            self.status['touchdown_softening'] = True
        self.status['correction'] = (out-np.asarray(command)).tolist()
        return out

    def telemetry(self):
        return self.status.copy()
