"""Offline ring regression cases; no simulator, ROS or motor output."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ros2_ws/src/dynamic_drone'))
from dynamic_drone.control.mission import FrontVisionMission, MissionFrame, MissionObserver
from dynamic_drone.control.depth_odometry import FrontDepthOdometry
from dynamic_drone.scene.ring_course import add_ring_corridor
from dynamic_drone.perception.yolo import MissionYOLO

K = np.array([[234., 0., 160.], [0., 234., 160.], [0., 0., 1.]])


def pilot_at(position=(0., 0., 1.2), phase='S2_FIND_RING', ring=None):
    pilot = FrontVisionMission()
    pilot.matrix = K.copy()
    pilot.phase, pilot.entered, pilot.started = phase, 9., 1.
    pilot.odometry = SimpleNamespace(position=np.array(position), rotation=np.eye(3),
                                    velocity=np.zeros(3), valid=True)
    pilot.depth.latest = dict(stamp=10., front=4., left=4., right=1.)
    frame = MissionFrame(np.zeros((320, 320), np.uint8), 10., {}, ring)
    return pilot, frame


def crossing_goal():
    return dict(center=np.array([0., 0., 1.2]), normal=np.array([1., 0., 0.]),
                start_position=np.array([-.55, 0., 1.2]), rotation=np.eye(3),
                previous_position=np.array([-.55, 0., 1.2]), plane_crossed=False,
                completion_ready=True)


def tof_sample(pilot, stamp, distance):
    from dynamic_drone.control.flow_deck import FlowDeckAssist
    if pilot.tof_sensor is None:
        pilot.tof_sensor = FlowDeckAssist()
    pilot.tof_transport_fresh = True
    pilot.tof_sensor.update_range(stamp, distance)
    pilot.tof_sensor.update_attitude(stamp, [0., 0., 0., 1.])


def cross_with_tof(pilot, frame, start):
    for offset, distance in [(0., 1.4), (.05, .7), (.10, .7), (.15, 1.4),
                             (.25, 1.4), (.35, 1.4), (.50, 1.4)]:
        tof_sample(pilot, start+offset, distance)
        pilot.command(frame, start+offset)


class RingPassageTests(unittest.TestCase):
    def test_tof_profile_counts_only_once_without_position_quality(self):
        pilot, frame = pilot_at(phase='S2_CROSS_RING')
        pilot.ring_goal = crossing_goal()
        pilot.odometry.valid = False
        cross_with_tof(pilot, frame, 10.)
        self.assertEqual(pilot.crossed, 1)
        self.assertEqual(pilot.ring_tof['state'], 'PASSED')
        self.assertFalse(pilot.verify_ring_tof(10.50))
        self.assertEqual(pilot.crossed, 1)

    def test_flat_ground_slow_drift_and_single_spike_are_not_passages(self):
        for distances in ([1.4]*20, [1.4-.02*i for i in range(20)],
                          [1.4, .7]+[1.4]*18):
            pilot, _ = pilot_at()
            for i, distance in enumerate(distances):
                stamp = 10.+i*.05
                tof_sample(pilot, stamp, distance)
                self.assertFalse(pilot.verify_ring_tof(stamp))

    def test_tof_recovery_requires_fresh_continuous_samples(self):
        pilot, _ = pilot_at()
        for stamp, distance in [(10., 1.4), (10.05, .7), (10.10, .7), (10.15, 1.4)]:
            tof_sample(pilot, stamp, distance)
            self.assertFalse(pilot.verify_ring_tof(stamp))
        # Duplicate, stale, invalid and missing transport cannot finish recovery.
        self.assertFalse(pilot.verify_ring_tof(10.16))
        self.assertFalse(pilot.verify_ring_tof(10.5))
        tof_sample(pilot, 10.55, float('nan'))
        self.assertFalse(pilot.verify_ring_tof(10.55))
        tof_sample(pilot, 10.60, 1.4)
        pilot.tof_transport_fresh = False
        self.assertFalse(pilot.verify_ring_tof(10.60))
        for stamp in (10.65, 10.75, 10.85):
            tof_sample(pilot, stamp, 1.4)
            self.assertFalse(pilot.verify_ring_tof(stamp))
        tof_sample(pilot, 11., 1.4)
        self.assertTrue(pilot.verify_ring_tof(11.))

    def test_near_opening_selected_over_more_confident_far_ring(self):
        gray = np.full((320, 320), 220, np.uint8)
        cv2.circle(gray, (90, 160), 45, 10, 7)
        cv2.circle(gray, (235, 160), 25, 10, 5)
        detections = [dict(label='ring', confidence=confidence, capture_seconds=10., xyxy=box)
                      for confidence, box in [(.60, [40, 110, 140, 210]), (.99, [205, 130, 265, 190])]]
        observer = MissionObserver(SimpleNamespace(predict=lambda *_: detections))
        observer.process(gray, 10.)
        self.assertIsNotNone(observer.latest.ring)
        np.testing.assert_allclose(observer.latest.ring['box_center'], [90., 160.], atol=2.)
        self.assertEqual(len(observer.latest.visible_rings), 2)
        self.assertEqual(observer.ring_diagnostics['yolo_candidates'], 2)
        # Detection order and higher confidence must not favor the smaller ring.
        detections.reverse()
        observer.process(gray, 10.05)
        np.testing.assert_equal(observer.latest.ring['box'], [40., 110., 140., 210.])

    def test_yolo_box_is_available_even_without_circular_rim_geometry(self):
        gray = np.full((320, 320), 220, np.uint8)
        cv2.rectangle(gray, (100, 100), (220, 220), 10, 8)
        detector = SimpleNamespace(predict=lambda *_: [dict(label='ring', confidence=.99,
                                   capture_seconds=10., xyxy=[90, 90, 230, 230])])
        observer = MissionObserver(detector)
        observer.process(gray, 10.)
        self.assertIsNotNone(observer.latest.ring)
        np.testing.assert_equal(observer.latest.ring['box_center'], [160., 160.])

    def test_locked_target_keeps_yolo_display_without_acquiring_next_ring(self):
        gray = np.full((320, 320), 220, np.uint8)
        cv2.circle(gray, (160, 160), 45, 10, 7)
        calls = []

        def detect(_, stamp):
            calls.append(stamp)
            return [dict(label='ring', confidence=.99, capture_seconds=stamp,
                         xyxy=[110, 110, 210, 210])]

        observer = MissionObserver(SimpleNamespace(predict=detect))
        observer.ring_target_locked = True
        observer.process(gray, 10.)
        self.assertEqual(calls, [10.])
        self.assertIsNone(observer.latest.ring)
        self.assertTrue(observer.latest.visible_rings)
        self.assertTrue(observer.ring_diagnostics['box_detected'])
        self.assertFalse(observer.ring_diagnostics['target_acquisition_enabled'])
        observer.ring_target_locked = False
        observer.process(gray, 10.05)
        self.assertEqual(calls, [10., 10.05])
        self.assertIsNotNone(observer.latest.ring)

    def test_crossing_command_ignores_rear_ring_even_with_aliased_depth(self):
        reference, reference_frame = pilot_at((-.5, 0., 1.2), 'S2_CROSS_RING')
        reference.ring_goal = crossing_goal()
        expected, _ = reference.command(reference_frame, 10.)

        pilot, frame = pilot_at((-.5, 0., 1.2), 'S2_CROSS_RING',
            ring=dict(box_center=np.array([280., 80.]), box_stamp=10.))
        pilot.ring_goal = crossing_goal()
        # A noisy rear-ring depth can fall inside the old same-ring tolerance.
        pilot.ring = dict(position=pilot.ring_goal['center'].copy())
        actual, _ = pilot.command(frame, 10.)
        self.assertGreater(actual[0], 0.)
        np.testing.assert_allclose(actual, expected)

    def test_crossing_does_not_acquire_new_ring_geometry(self):
        contour = cv2.ellipse2Poly((160, 160), (55, 55), 0, 0, 360, 3).reshape(-1, 1, 2)
        pilot, frame = pilot_at((-.5, 0., 1.2), 'S2_CROSS_RING',
            ring=dict(center=np.array([160., 160.]), contour=contour, confidence=.9))
        pilot.ring_goal = crossing_goal()
        pilot.observe(frame, np.full((62, 82), 2.), {})
        self.assertIsNone(pilot.ring)
        np.testing.assert_equal(pilot.ring_goal['center'], [0., 0., 1.2])

    def test_plane_passage_alone_cannot_unlock_without_tof(self):
        pilot, frame = pilot_at((-.1, 0., 1.2), 'S2_CROSS_RING')
        pilot.ring_goal = crossing_goal()
        pilot.command(frame, 10.)
        self.assertIsNotNone(pilot.ring_goal)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        self.assertEqual(pilot.crossed, 0)
        pilot.odometry.position[0] = .01
        pilot.command(frame, 10.05)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        self.assertGreater(pilot.previous_command[0], 0.)
        pilot.odometry.position[0] = .11
        pilot.command(frame, 10.10)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        self.assertEqual(pilot.crossed, 0)
        self.assertIsNotNone(pilot.ring_goal)

    def test_missing_ring_never_uses_side_clearance_to_bypass(self):
        pilot, frame = pilot_at()
        pilot.depth.latest['front'] = .8
        cmd, _ = pilot.command(frame, 10.)
        np.testing.assert_equal(cmd[:2], [0., 0.])
        self.assertEqual(pilot.crossed, 0)

    def test_off_center_box_aligns_without_forward_even_if_plane_exists(self):
        pilot, frame = pilot_at(ring=dict(box_center=np.array([195., 185.]), box_stamp=10.))
        pilot.ring = dict(position=np.array([2., 0., 1.2]), normal=np.array([1., 0., 0.]), box_stamp=10.)
        cmd, _ = pilot.command(frame, 10.)
        self.assertEqual(cmd[0], 0.)
        self.assertEqual(pilot.phase, 'S2_ALIGN_RING')
        self.assertLess(cmd[2], 0.)
        self.assertLess(cmd[3], 0.)

    def test_camera_principal_point_is_used_for_image_alignment(self):
        pilot, _ = pilot_at()
        pilot.matrix[:2, 2] = [154., 166.]
        aligned = pilot.center_on_image([154., 166.], .5, 10., 'test')
        np.testing.assert_equal(aligned, [.5, 0., 0., 0.])
        off_center = pilot.center_on_image([180., 140.], .5, 10.05, 'test')
        self.assertEqual(off_center[0], 0.)
        self.assertGreater(off_center[2], 0.)
        self.assertLess(off_center[3], 0.)

    def test_passage_plane_uses_box_center_not_contour_center(self):
        contour = cv2.ellipse2Poly((160, 160), (55, 55), 0, 0, 360, 3).reshape(-1, 1, 2)
        pilot, frame = pilot_at(ring=dict(box_center=np.array([170., 160.]), box_stamp=10., contour=contour, confidence=.9))
        pilot.observe(frame, np.full((62, 82), 2.), {})
        self.assertIsNotNone(pilot.ring)
        np.testing.assert_allclose(pilot.ring['position'], [2., -20./234., 1.2], atol=.01)

    def test_opening_error_does_not_abort_but_pose_alone_cannot_complete(self):
        pilot, frame = pilot_at((.26, .6, 1.2), 'S2_CROSS_RING')
        pilot.ring_goal = crossing_goal()
        pilot.command(frame, 10.)
        self.assertEqual(pilot.crossed, 0)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')

    def test_ring_time_height_clearance_and_tracking_do_not_abort(self):
        pilot, frame = pilot_at((-.5, 0., 3.), 'S2_CROSS_RING')
        pilot.ring_goal = crossing_goal()
        pilot.ring_goal['center'][2] = 3.
        pilot.entered, pilot.started, pilot.progress_stamp = 0., 0., 0.
        pilot.last_progress = pilot.odometry.position.copy()
        pilot.odometry.valid = False
        pilot.depth.latest = None
        pilot.command(frame, 1000.)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        self.assertGreater(pilot.previous_command[0], 0.)

    def test_alignment_starts_forced_passage_without_depth_plane_or_valid_odometry(self):
        pilot, frame = pilot_at(ring=dict(box_center=np.array([161., 159.]), box_stamp=10.,
            box=np.array([120., 120., 200., 200.]), contour=None))
        pilot.odometry.valid = False
        pilot.depth.latest = None
        pilot.entered = 0.
        cmd, _ = pilot.command(frame, 1000.)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        self.assertGreater(cmd[0], 0.)
        self.assertIsNone(pilot.ring_goal['center'])
        np.testing.assert_equal(pilot.ring_goal['start_position'], [0., 0., 1.2])
        pilot.odometry.position[1] = .2
        pilot.command(None, 1000.05)
        np.testing.assert_equal(pilot.ring_goal['start_position'], [0., 0., 1.2])
        self.assertGreater(pilot.previous_command[0], 0.)
        self.assertEqual(pilot.previous_command[1], 0.)

    def test_aligned_box_locks_pose_and_requires_tof_for_next_transition(self):
        pilot, frame = pilot_at(ring=dict(box_center=np.array([160., 160.]), box_stamp=10.,
            box=np.array([40., 40., 280., 280.]), contour=None))
        pilot.ring = dict(position=np.array([2., 0., 1.2]), normal=np.array([1., 0., 0.]), box_stamp=10.)
        pilot.command(frame, 10.)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        self.assertGreater(pilot.previous_command[0], 0.)
        pilot.odometry.position[0] = 2.26
        pilot.command(frame, 10.05)
        self.assertEqual(pilot.crossed, 0)
        cross_with_tof(pilot, frame, 10.1)
        self.assertEqual(pilot.crossed, 1)
        self.assertEqual(pilot.phase, 'S2_FIND_RING')
        cmd, _ = pilot.command(frame, 10.9)
        self.assertEqual(cmd[0], 0.)
        self.assertEqual(pilot.phase, 'S2_FIND_RING')
        frame.ring['box_stamp'] = 11.
        pilot.depth.latest['stamp'] = 11.
        pilot.ring = None
        pilot.command(frame, 11.)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        self.assertIsNone(pilot.ring_goal['center'])
        self.assertFalse(pilot.ring_tof['edge_seen'])
        for stamp in (11.05, 11.15, 11.25, 11.35):
            tof_sample(pilot, stamp, 1.4)
            pilot.command(frame, stamp)
        self.assertEqual(pilot.crossed, 1)  # A new ring requires a new surface pulse.
        np.testing.assert_equal(pilot.ring_goal['start_position'], [2.26, 0., 1.2])

    def test_missing_start_plane_is_measured_from_locked_rim_not_rear_yolo_box(self):
        gray = np.full((320, 320), 220, np.uint8)
        cv2.circle(gray, (160, 160), 120, 10, 7)
        contour = cv2.ellipse2Poly((160, 160), (120, 120), 0, 0, 360, 3).reshape(-1, 1, 2)
        target = dict(box_center=np.array([160., 160.]), box_stamp=10., contour=contour)
        pilot, frame = pilot_at(ring=target)
        frame.gray = gray
        pilot.command(frame, 10.)
        self.assertIsNone(pilot.ring_goal['center'])
        rear = MissionFrame(gray.copy(), 10.05, {}, dict(box_center=np.array([280., 80.])))
        pilot.observe(rear, np.full((62, 82), 2.), {})
        np.testing.assert_allclose(pilot.ring_goal['center'], [2., 0., 1.2], atol=.02)
        np.testing.assert_equal(pilot.ring_goal['start_position'], [0., 0., 1.2])
        self.assertIsNone(pilot.ring)
        self.assertTrue(pilot.ring_goal['completion_ready'])

    def test_distant_noisy_plane_cannot_complete_before_locked_rim_is_near(self):
        gray = np.full((320, 320), 220, np.uint8)
        cv2.circle(gray, (160, 160), 40, 10, 7)
        contour = cv2.ellipse2Poly((160, 160), (40, 40), 0, 0, 360, 3).reshape(-1, 1, 2)
        target = dict(box_center=np.array([160., 160.]), box_stamp=10., contour=contour)
        pilot, frame = pilot_at(ring=target)
        frame.gray = gray
        pilot.ring = dict(position=np.array([.2, 0., 1.2]), normal=np.array([1., 0., 0.]), box_stamp=10.)
        pilot.command(frame, 10.)
        pilot.odometry.position[0] = .6
        pilot.observe(frame, np.full((62, 82), .2), {})
        cmd, _ = pilot.command(frame, 10.05)
        self.assertGreater(cmd[0], 0.)
        self.assertEqual(pilot.crossed, 0)
        self.assertFalse(pilot.ring_goal['completion_ready'])

    def test_very_near_obstacle_depth_does_not_stop_locked_passage(self):
        pilot, frame = pilot_at((-.5, 0., 1.2), 'S2_CROSS_RING')
        pilot.crossed = 1
        pilot.ring_goal = crossing_goal()
        pilot.depth.latest.update(front=.01, left=.01, right=.01)
        cmd, _ = pilot.command(frame, 10.)
        self.assertGreater(cmd[0], 0.)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        self.assertFalse(pilot.ring_obstacle_avoidance)

    def test_locked_rim_geometry_still_needs_tof_before_next_acquisition(self):
        gray = np.full((320, 320), 220, np.uint8)
        cv2.circle(gray, (160, 160), 40, 10, 7)
        contour = cv2.ellipse2Poly((160, 160), (40, 40), 0, 0, 360, 3).reshape(-1, 1, 2)
        pilot, frame = pilot_at(ring=dict(box_center=np.array([160., 160.]), box_stamp=10., contour=contour))
        frame.gray = gray
        pilot.command(frame, 10.)
        plane_x = K[0, 0]*.5/40
        for i, radius in enumerate(range(42, 126, 2), 1):
            gray = np.full((320, 320), 220, np.uint8)
            cv2.circle(gray, (160, 160), radius, 10, 7)
            distance = K[0, 0]*.5/radius
            pilot.odometry.position[0] = plane_x-distance
            frame = MissionFrame(gray, 10+i*.05, {}, dict(box_center=np.array([240., 100.])))
            pilot.observe(frame, np.full((62, 82), distance), {})
            if pilot.ring_goal['completion_ready']:
                break
        self.assertTrue(pilot.ring_goal['completion_ready'])
        self.assertIsNone(pilot.ring)
        self.assertAlmostEqual(pilot.ring_goal['center'][0], plane_x, places=2)
        pilot.odometry.position[0] = plane_x+.05
        cmd, _ = pilot.command(frame, 13.)
        self.assertEqual(pilot.crossed, 0)
        self.assertGreater(cmd[0], 0.)
        pilot.odometry.position[0] = plane_x+.11
        pilot.command(frame, 13.05)
        self.assertEqual(pilot.crossed, 0)
        cross_with_tof(pilot, frame, 13.1)
        self.assertEqual(pilot.crossed, 1)
        self.assertEqual(pilot.phase, 'S2_FIND_RING')

    def test_after_passage_obstacle_blocks_next_lock_and_uses_clear_side(self):
        pilot, frame = pilot_at(ring=dict(box_center=np.array([160., 160.]), box_stamp=10.))
        pilot.crossed = 1
        pilot.depth.latest.update(front=.7, left=3., right=.3)
        for t in np.arange(10., 11., .05):
            pilot.depth.latest['stamp'] = float(t)
            cmd, reason = pilot.command(frame, float(t))
        np.testing.assert_allclose(cmd, [0., .08, 0., 0.])
        self.assertEqual(reason, 'between_rings_obstacle_avoidance')
        self.assertIsNone(pilot.ring_goal)
        # A newly clear depth allows the same centered ring to start immediately.
        pilot.depth.latest.update(stamp=11., front=3.)
        frame.ring['box_stamp'] = 11.
        cmd, _ = pilot.command(frame, 11.)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        self.assertGreater(cmd[0], 0.)
        self.assertEqual(cmd[1], 0.)

    def test_between_passages_without_clear_escape_or_fresh_depth_holds(self):
        for bands in (dict(stamp=10., front=.4, left=.3, right=.2),
                      dict(stamp=9., front=4., left=4., right=4.), None):
            with self.subTest(bands=bands):
                pilot, frame = pilot_at()
                pilot.crossed = 1
                pilot.previous_command[:] = [.20, -.08, .04, .06]
                pilot.depth.latest = bands
                cmd, _ = pilot.command(frame, 10.)
                np.testing.assert_equal(cmd, np.zeros(4))
                self.assertIsNone(pilot.ring_goal)

    def test_bad_pose_cannot_count_crossing_or_inject_lateral_corrections(self):
        pilot, frame = pilot_at((.4, 5., 4.), 'S2_CROSS_RING')
        pilot.ring_goal = crossing_goal()
        pilot.odometry.valid = False
        cmd, _ = pilot.command(frame, 10.)
        self.assertGreater(cmd[0], 0.)
        np.testing.assert_equal(cmd[1:], np.zeros(3))
        self.assertEqual(pilot.crossed, 0)
        np.testing.assert_equal(pilot.ring_goal['previous_position'], [-.55, 0., 1.2])

    def test_ring_odometry_rejects_outlier_pose_without_overwriting_reference(self):
        odom = FrontDepthOdometry(K)
        odom.quality_checks = False
        gray = np.zeros((320, 320), np.uint8)
        points = np.array([[[100., 100.]], [[140., 100.]], [[180., 100.]],
                           [[100., 180.]], [[140., 180.]], [[180., 180.]]], np.float32)
        odom.update(gray, np.full((320, 320), 2.), 10.)
        with patch.object(cv2, 'goodFeaturesToTrack', return_value=points), \
             patch.object(cv2, 'calcOpticalFlowPyrLK', return_value=(points, np.ones((6, 1), np.uint8), None)), \
             patch.object(cv2, 'solvePnPRansac', return_value=(True, np.zeros((3, 1)), np.zeros((3, 1)), np.arange(6).reshape(-1, 1))), \
             patch.object(cv2, 'projectPoints', return_value=(points+29., None)):
            odom.update(gray, np.full((320, 320), 2.), 10.05)
        self.assertFalse(odom.valid)
        self.assertEqual(odom.reason, 'visual_motion_jump_or_residual')
        self.assertEqual(odom.stamp, 10.)
        np.testing.assert_equal(odom.position, np.zeros(3))

    def test_crossing_ignores_metric_drift_and_keeps_only_body_forward(self):
        pilot, frame = pilot_at((-.5, .25, 1.4), 'S2_CROSS_RING')
        pilot.ring_goal = crossing_goal()
        yaw = .4
        pilot.odometry.rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                                            [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        pilot.previous_command[:] = [.20, .04, -.04, -.02]
        cmd, _ = pilot.command(frame, 10.)
        np.testing.assert_equal(cmd, [.20, 0., 0., 0.])
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        np.testing.assert_equal(pilot.ring_goal['start_position'], [-.55, 0., 1.2])

    def test_ring_motion_can_use_consistent_minority_without_accepting_outliers(self):
        odom = FrontDepthOdometry(K)
        odom.quality_checks = False
        gray = np.zeros((320, 320), np.uint8)
        pixels = np.array([[[x, y]] for y in (90., 130., 170., 210.)
                           for x in (90., 120., 150., 180., 210.)], np.float32)
        odom.update(gray, np.full((320, 320), 2.), 10.)
        with patch.object(cv2, 'goodFeaturesToTrack', return_value=pixels), \
             patch.object(cv2, 'calcOpticalFlowPyrLK', return_value=(pixels, np.ones((20, 1), np.uint8), None)), \
             patch.object(cv2, 'solvePnPRansac', return_value=(True, np.zeros((3, 1)), np.zeros((3, 1)), np.arange(6).reshape(-1, 1))), \
             patch.object(cv2, 'projectPoints', return_value=(pixels[:6], None)):
            odom.update(gray, np.full((320, 320), 2.), 10.05)
        self.assertTrue(odom.valid)
        self.assertEqual(odom.inliers, 6)
        np.testing.assert_equal(odom.position, np.zeros(3))

    def test_passage_slows_for_expanding_locked_rim_without_stopping_or_steering(self):
        pilot, frame = pilot_at((-.5, 0., 1.2), 'S2_CROSS_RING')
        pilot.ring_goal = crossing_goal()
        pilot.ring_goal['completion_ready'] = False
        pilot.ring_goal['rim_pixels'] = np.array([[100., 100.], [220., 100.], [220., 220.], [100., 220.]])
        pilot.previous_command[0] = .40
        cmd, _ = pilot.command(frame, 10.)
        np.testing.assert_equal(cmd, [.40, 0., 0., 0.])
        pilot.ring_goal['rim_pixels'] = np.array([[40., 40.], [280., 40.], [280., 280.], [40., 280.]])
        pilot.depth.latest = None
        for t in np.arange(10.1, 12.1, .1):
            cmd, _ = pilot.command(frame, float(t))
            self.assertGreater(cmd[0], 0.)
            np.testing.assert_equal(cmd[1:], np.zeros(3))
        self.assertAlmostEqual(cmd[0], .20)
        self.assertEqual(pilot.crossed, 0)

    def test_locked_box_refresh_rejects_smaller_rear_ring(self):
        def target(radius, stamp):
            return dict(box=np.array([160-radius,160-radius,160+radius,160+radius], float),
                box_center=np.array([160.,160.]), box_stamp=stamp, contour=None)
        pilot, frame = pilot_at(ring=target(40, 10.))
        pilot.command(frame, 10.)
        observed = MissionFrame(frame.gray.copy(), 10.2, {}, None,
            visible_rings=[target(15,10.2),target(50,10.2)])
        with patch.object(cv2, 'calcOpticalFlowPyrLK', side_effect=lambda old,new,points,*a,**kw:
                (points.copy(),np.ones((len(points),1),np.uint8),None)):
            pilot.observe(observed, np.full((62,82),2.), {})
            np.testing.assert_equal(pilot.ring_goal['box'], [110.,110.,210.,210.])
            observed.stamp=10.4
            observed.visible_rings=[target(15,10.4)]
            pilot.observe(observed, np.full((62,82),2.), {})
        np.testing.assert_equal(pilot.ring_goal['box'], [110.,110.,210.,210.])
        np.testing.assert_equal(pilot.ring_goal['start_position'], [0.,0.,1.2])

    def test_same_ring_boxes_recover_geometry_when_flow_sticks_to_background(self):
        pilot, frame = pilot_at(ring=dict(box=np.array([120.,120.,200.,200.]),
            box_center=np.array([160.,160.]),box_stamp=10.,contour=None))
        pilot.command(frame,10.)
        with patch.object(cv2,'calcOpticalFlowPyrLK',side_effect=lambda old,new,points,*a,**kw:
                (points.copy(),np.ones((len(points),1),np.uint8),None)):
            for i,radius in enumerate((50,60,72,86,104,120),1):
                stamp=10.+i*.2
                box=np.array([160-radius,160-radius,160+radius,160+radius],float)
                observed=MissionFrame(frame.gray.copy(),stamp,{},None,visible_rings=[
                    dict(box=box,box_center=np.array([160.,160.]),box_stamp=stamp,contour=None)])
                pilot.observe(observed,np.full((62,82),2.),{})
        self.assertTrue(pilot.ring_goal['completion_ready'])
        np.testing.assert_allclose(pilot.ring_goal['center'],[2.,0.,1.2],atol=.02)
        np.testing.assert_equal(pilot.ring_goal['start_position'],[0.,0.,1.2])

    def test_large_ring_ellipse_is_followed_when_yolo_disappears(self):
        gray=np.full((320,320),220,np.uint8)
        cv2.circle(gray,(160,160),40,10,7)
        pilot,frame=pilot_at(ring=dict(box=np.array([120.,120.,200.,200.]),
            box_center=np.array([160.,160.]),box_stamp=10.,contour=None))
        frame.gray=gray
        pilot.command(frame,10.)
        with patch.object(cv2,'calcOpticalFlowPyrLK',side_effect=lambda old,new,points,*a,**kw:
                (points.copy(),np.ones((len(points),1),np.uint8),None)):
            for i,radius in enumerate((50,60,72,86,104,120),1):
                gray=np.full((320,320),220,np.uint8)
                cv2.circle(gray,(160,160),radius,10,7)
                observed=MissionFrame(gray,10.+i*.2,{},None,visible_rings=[])
                pilot.observe(observed,np.full((62,82),2.),{})
        self.assertTrue(pilot.ring_goal['completion_ready'])
        self.assertGreaterEqual(np.min(pilot.ring_goal['box'][2:]-pilot.ring_goal['box'][:2]),224.)
        np.testing.assert_allclose(pilot.ring_goal['center'],[2.,0.,1.2],atol=.03)
        np.testing.assert_equal(pilot.ring_goal['start_position'],[0.,0.,1.2])

    def test_full_image_rim_takes_priority_over_cropped_yolo_geometry(self):
        pilot,frame = pilot_at(ring=dict(box=np.array([60.,60.,260.,260.]),
            box_center=np.array([160.,160.]),box_stamp=10.,contour=None))
        pilot.command(frame,10.)
        pilot.odometry.valid = False
        gray=np.full((320,320),220,np.uint8)
        cv2.circle(gray,(160,160),120,10,4)
        # An unreliable ellipse from the cropped ROI must not corrupt the
        # actual full-image diameter used to predict this ring's passage.
        cropped=cv2.ellipse2Poly((160,160),(145,145),0,0,360,3).reshape(-1,1,2)
        target=dict(box=np.array([40.,40.,280.,280.]),box_center=np.array([160.,160.]),
            box_stamp=10.2,contour=cropped)
        frame=MissionFrame(gray,10.2,{},None,visible_rings=[target])
        pilot.observe(frame,np.full((62,82),np.nan),{})
        size=np.min(pilot.ring_goal['box'][2:]-pilot.ring_goal['box'][:2])
        self.assertAlmostEqual(size,240.,delta=8.)

    def test_inside_crossing_counts_once_after_tof_profile(self):
        pilot, frame = pilot_at((.26, .05, 1.2), 'S2_CROSS_RING')
        pilot.ring_goal = crossing_goal()
        pilot.command(frame, 10.)
        self.assertEqual(pilot.crossed, 0)
        cross_with_tof(pilot, frame, 10.1)
        self.assertEqual(pilot.crossed, 1)
        self.assertEqual(pilot.phase, 'S2_FIND_RING')
        pilot.command(frame, 10.9)
        self.assertEqual(pilot.crossed, 1)

    def expanded_ring_with_missing_pose(self):
        # Constant approach to camera-plane time 18; only rendered front
        # images enter the controller. Pose remains unavailable throughout.
        radius = round(585./8.)
        gray = np.full((320,320),220,np.uint8)
        cv2.circle(gray,(160,160),radius,10,4)
        pilot,frame = pilot_at(ring=dict(box=np.array([160-radius,160-radius,160+radius,160+radius],float),
            box_center=np.array([160.,160.]),box_stamp=10.,contour=None))
        frame.gray = gray
        pilot.command(frame,10.)
        pilot.odometry.valid = False
        pilot.previous_command[0] = .20
        for stamp in np.arange(10.2,15.61,.2):
            gray = np.full((320,320),220,np.uint8)
            cv2.circle(gray,(160,160),round(585./(18.-stamp)),10,4)
            frame = MissionFrame(gray,float(stamp),{},None)
            pilot.observe(frame,np.full((62,82),np.nan),{})
        return pilot,frame

    def test_tof_unlocks_without_pose_while_visual_prediction_alone_does_not(self):
        pilot,frame = self.expanded_ring_with_missing_pose()
        goal = pilot.ring_goal
        self.assertTrue(goal['visual_filled'])
        self.assertAlmostEqual(goal['visual_pass_time'],18.,delta=.3)
        predicted = goal['visual_pass_time']
        cmd,_ = pilot.command(frame,predicted+.7)
        self.assertEqual(pilot.crossed,0)
        self.assertGreater(cmd[0],0.)
        pilot.command(frame,predicted+.81)
        self.assertEqual(pilot.crossed,0)
        cross_with_tof(pilot,frame,predicted+1.)
        self.assertEqual(pilot.crossed,1)
        self.assertEqual(pilot.phase,'S2_FIND_RING')
        self.assertFalse(pilot.odometry.valid)
        np.testing.assert_equal(pilot.odometry.position,[0.,0.,1.2])
        np.testing.assert_equal(goal['start_position'],[0.,0.,1.2])

    def test_visual_prediction_does_not_unlock_while_locked_rim_is_visible(self):
        pilot,frame = self.expanded_ring_with_missing_pose()
        now = pilot.ring_goal['visual_pass_time']+1.
        pilot.ring_goal['last_rim_observation'] = now
        pilot.command(frame,now)
        self.assertEqual(pilot.crossed,0)
        self.assertEqual(pilot.phase,'S2_CROSS_RING')

    def test_stationary_ring_or_rear_box_cannot_predict_passage(self):
        pilot,frame = pilot_at((-.5,0.,1.2),'S2_CROSS_RING')
        pilot.ring_goal = crossing_goal()
        pilot.previous_command[0] = .20
        pixels = cv2.ellipse2Poly((160,160),(155,155),0,0,360,3).astype(np.float32)
        for stamp in np.arange(10.,12.,.2):
            pilot.observe_ring_expansion(pilot.ring_goal,pixels,float(stamp))
        self.assertNotIn('visual_pass_time',pilot.ring_goal)
        pilot.odometry.valid = False
        pilot.command(frame,30.)
        self.assertEqual(pilot.crossed,0)

    def test_clipped_arc_diameter_outliers_do_not_delay_visual_prediction(self):
        pilot,_ = pilot_at()
        pilot.previous_command[0] = .20
        goal = crossing_goal()
        for i,stamp in enumerate(np.arange(12.8,15.61,.1)):
            radius = 585./(18.-stamp)
            # Intermittent partial arcs fit a smaller ellipse; the consistent
            # majority must determine time instead of shifting it seconds.
            if i%4 == 0:
                radius *= .8
            theta = np.linspace(0.,2*np.pi,150,endpoint=False)
            pixels=np.c_[160+radius*np.cos(theta),160+radius*np.sin(theta)].astype(np.float32)
            pilot.observe_ring_expansion(goal,pixels,float(stamp))
        self.assertTrue(goal['visual_filled'])
        self.assertAlmostEqual(goal['visual_pass_time'],18.,delta=.05)

    def test_sparse_consistent_expansion_survives_many_bad_arc_fits(self):
        pilot,_ = pilot_at()
        pilot.previous_command[0] = .20
        goal = crossing_goal()
        theta = np.linspace(0.,2*np.pi,150,endpoint=False)
        rng = np.random.default_rng(7)
        for i,stamp in enumerate(np.arange(12.8,16.01,.05)):
            radius = 585./(18.-stamp)
            if i%3:
                radius *= rng.uniform(.70,.90)
            pixels=np.c_[160+radius*np.cos(theta),160+radius*np.sin(theta)].astype(np.float32)
            pilot.observe_ring_expansion(goal,pixels,float(stamp))
        self.assertAlmostEqual(goal['visual_pass_time'],18.,delta=.05)

    def test_ring_structures_preserve_existing_models_and_opening(self):
        world = ET.Element('world')
        marker = ET.SubElement(world, 'model', name='route_marker_6')
        ET.SubElement(marker, 'pose').text = '22.4 1.6 1.2 0 0 0'
        original = ET.tostring(marker)
        layout = add_ring_corridor(world, dict(width=1., frame_thickness=.06,
                                              centers=[[26., 0., 1.5]]))
        self.assertEqual(layout['building_count'], 2)
        self.assertEqual(ET.tostring(marker), original)
        for model in list(world)[1:]:
            center = np.fromstring(model.findtext('pose'), sep=' ')[:3]
            size = np.fromstring(model.findtext('.//collision/geometry/box/size'), sep=' ')
            self.assertGreater(center[0]-size[0]/2, 24.)
            self.assertFalse(np.all(np.abs(np.array([26., 0., 1.5])-center) < size/2+.12))


class RingTrackingTests(unittest.TestCase):
    def scene(self, dx=0., dy=0., scale=1., occluded=False):
        gray = np.full((320, 320), 210, np.uint8)
        cv2.circle(gray, (150, 150), 55, 15, 7, cv2.LINE_AA)
        cv2.circle(gray, (250, 240), 18, 15, 4, cv2.LINE_AA)
        transform = cv2.getRotationMatrix2D((150., 150.), 0., scale)
        transform[:, 2] += [dx, dy]
        gray = cv2.warpAffine(gray, transform, (320, 320), borderValue=210)
        if occluded:
            cv2.rectangle(gray, (190, 125), (215, 160), 210, -1)
        return gray

    def detection(self, stamp, box=(90., 90., 210., 210.)):
        return dict(label='ring', confidence=.9, capture_seconds=stamp, xyxy=list(box))

    def observer(self, detect=None):
        return MissionObserver(SimpleNamespace(predict=detect or (lambda _, stamp:
            [self.detection(stamp)] if stamp == 10. else []), rejected_rings=0))

    def test_half_second_dropout_tracks_moving_center_without_rear_switch(self):
        observer = self.observer(lambda _, stamp: [self.detection(stamp)] if stamp == 10.
            else [self.detection(stamp, (228., 218., 272., 262.))])
        observer.process(self.scene(), 10.)
        for i in range(1, 11):
            stamp, dx, dy = 10.+i*.05, i*.7, -i*.4
            observer.process(self.scene(dx, dy), stamp)
            target = observer.latest.ring
            self.assertIsNotNone(target, f'missing frame {i}')
            np.testing.assert_allclose(target['box_center'], [150.+dx, 150.+dy], atol=.5)
            self.assertEqual(target['source'], 'verified_ring_optical_flow')
            self.assertEqual(target['box_stamp'], stamp)
            self.assertEqual(target['detected_stamp'], 10.)
            self.assertGreater(target['box'][2]-target['box'][0], 110.)

    def test_smaller_concentric_detection_cannot_refresh_front_target(self):
        observer = self.observer(lambda _, stamp: [self.detection(stamp)] if stamp == 10.
            else [self.detection(stamp, (99., 99., 201., 201.))])
        observer.process(self.scene(), 10.)
        observer.process(self.scene(), 10.05)
        self.assertEqual(observer.latest.ring['source'], 'verified_ring_optical_flow')
        self.assertEqual(observer.latest.ring['detected_stamp'], 10.)
        self.assertGreater(observer.latest.ring['box'][2]-observer.latest.ring['box'][0], 119.)

    def test_new_larger_verified_ring_takes_priority_over_tracked_rear_ring(self):
        observer = self.observer(lambda _, stamp: [self.detection(stamp, (228., 218., 272., 262.))]
            + ([self.detection(stamp)] if stamp > 10. else []))
        observer.process(self.scene(), 10.)
        observer.process(self.scene(), 10.05)
        np.testing.assert_allclose(observer.latest.ring['box_center'], [150., 150.])
        self.assertEqual(observer.latest.ring['source'], 'yolo')

    def test_partial_occlusion_and_scale_change_retain_measured_center(self):
        observer = self.observer()
        observer.process(self.scene(), 10.)
        for i in range(1, 8):
            observer.process(self.scene(i*.5, -i*.3, 1.+i*.005, occluded=i>2), 10.+i*.05)
            self.assertIsNotNone(observer.latest.ring, f'missing frame {i}')
            np.testing.assert_allclose(observer.latest.ring['box_center'],
                                       [150.+i*.5, 150.-i*.3], atol=1.)

    def test_partly_clipped_rim_keeps_measured_visible_box(self):
        gray = np.full((320, 320), 210, np.uint8)
        cv2.circle(gray, (290, 150), 55, 15, 7, cv2.LINE_AA)
        observer = self.observer(lambda _, stamp:
            [self.detection(stamp, (230., 90., 320., 210.))] if stamp == 10. else [])
        observer.process(gray, 10.)
        for i in range(1, 9):
            moved = cv2.warpAffine(gray, np.float32([[1, 0, i*.5], [0, 1, 0]]),
                                   (320, 320), borderMode=cv2.BORDER_REPLICATE)
            observer.process(moved, 10.+i*.05)
            self.assertIsNotNone(observer.latest.ring)
            np.testing.assert_allclose(observer.latest.ring['box_center'], [275.+i*.25, 150.], atol=.75)

    def test_cached_detection_never_refreshes_identity_or_expires_by_flow(self):
        observer = self.observer(lambda *_: [self.detection(10.)])
        for i in range(14):
            observer.process(self.scene(), 10.+i*.05)
        self.assertIsNone(observer.latest.ring)
        self.assertIsNone(observer.ring_track)
        self.assertEqual(observer.ring_diagnostics['fresh_yolo_candidates'], 0)

    def test_blank_frame_rectangle_replacement_and_time_gap_do_not_track(self):
        for mode in ('blank', 'rectangle', 'time_gap'):
            with self.subTest(mode=mode):
                observer = self.observer()
                observer.process(self.scene(), 10.)
                gray = np.full((320, 320), 210, np.uint8)
                if mode == 'rectangle':
                    cv2.rectangle(gray, (95, 95), (205, 205), 15, 7)
                if mode == 'time_gap':
                    gray = self.scene()
                observer.process(gray, 10.3 if mode == 'time_gap' else 10.05)
                self.assertIsNone(observer.latest.ring)

    def test_crossing_unlock_and_invalid_frame_clear_old_tracking(self):
        observer = self.observer()
        observer.process(self.scene(), 10.)
        observer.ring_target_locked = True
        observer.process(self.scene(), 10.05)
        self.assertIsNone(observer.latest.ring)
        self.assertIsNone(observer.ring_track)
        observer.ring_target_locked = False
        observer.process(self.scene(), 10.1)
        self.assertIsNone(observer.latest.ring)
        observer = self.observer()
        observer.process(self.scene(), 10.)
        observer.process(self.scene(), 10.)  # Duplicate images are not motion.
        self.assertIsNone(observer.latest)
        self.assertIsNone(observer.ring_track)

    def test_verified_flow_reduces_detector_box_jitter(self):
        def detect(_, stamp):
            i = round((stamp-10.)/.05)
            dx = 0. if i == 0 else 4.*(-1)**i
            return [self.detection(stamp, (90.+dx, 90., 210.+dx, 210.))]
        observer = self.observer(detect)
        errors = []
        for i in range(12):
            observer.process(self.scene(), 10.+i*.05)
            self.assertIsNotNone(observer.latest.ring)
            errors.append(abs(observer.latest.ring['box_center'][0]-150.))
        self.assertLess(max(errors), 2.)

    def test_measured_tracking_aligns_then_starts_forward_without_early_commit(self):
        observer = self.observer()
        pilot, _ = pilot_at()
        for i in range(11):
            stamp = 10.+i*.05
            observer.process(self.scene(i, i), stamp)
            command, _ = pilot.command(observer.latest, stamp)
            if i < 8:  # Both actual image errors still exceed 2 pixels.
                self.assertEqual(pilot.phase, 'S2_ALIGN_RING')
                self.assertEqual(command[0], 0.)
        self.assertEqual(pilot.phase, 'S2_CROSS_RING')
        self.assertGreater(command[0], 0.)
        np.testing.assert_array_equal(command[1:], np.zeros(3))
        self.assertEqual(pilot.crossed, 0)  # ToF, not tracking, confirms passage.

    def test_brief_unmeasured_gap_holds_instead_of_searching_or_crossing(self):
        pilot, frame = pilot_at(ring=dict(box_center=np.array([185., 170.]), box_stamp=10.))
        pilot.command(frame, 10.)
        frame.ring = None
        command, reason = pilot.command(frame, 10.1)
        np.testing.assert_array_equal(command, np.zeros(4))
        self.assertEqual(reason, 'ring_brief_detection_gap_hold')
        self.assertEqual(pilot.phase, 'S2_ALIGN_RING')
        self.assertIsNotNone(pilot.ring_image_pid.stamp)
        command, _ = pilot.command(frame, 10.4)
        self.assertNotEqual(command[3], 0.)
        self.assertIsNone(pilot.ring_image_pid.stamp)


class MissionSessionTests(unittest.TestCase):
    def test_two_runs_do_not_share_camera_clock_or_command_topics(self):
        import yaml
        runtime = Path(__file__).resolve().parents[1]/'scripts/runtime'
        sys.path.insert(0, str(runtime))
        from flight import bridge_config
        with tempfile.TemporaryDirectory() as location:
            topic_sets = []
            for key in ['run_a', 'run_b']:
                directory = Path(location)/key
                directory.mkdir()
                path = bridge_config(directory, ros_namespace='/vision_trial_'+key, full_mission=True)
                mappings = yaml.safe_load(path.read_text())
                topics = {row['ros_topic_name'] for row in mappings}
                self.assertEqual(topics, {'/vision_trial_'+key+'/'+suffix for suffix in
                    ['image', 'camera_info', 'clock', 'contact', 'cmd', 'enable', 'truck_cmd']})
                clock = next(row for row in mappings if row['ros_type_name'] == 'rosgraph_msgs/msg/Clock')
                self.assertEqual(clock['gz_topic_name'], '/clock')
                self.assertNotIn('/clock', topics)
                topic_sets.append(topics)
            self.assertFalse(topic_sets[0] & topic_sets[1])


class YOLOInferenceTests(unittest.TestCase):
    def make_detector(self, boxes, names=None):
        calls = []
        values = np.asarray(boxes, np.float32).reshape(-1, 6)
        tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: values))
        model = SimpleNamespace(names={0: 'ring'} if names is None else names)
        def predict(image, **options):
            calls.append((image.copy(), options))
            return [SimpleNamespace(boxes=SimpleNamespace(data=tensor))]
        model.predict = predict
        with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=lambda _: model)}):
            detector = MissionYOLO(Path(__file__).resolve().parents[1])
        # Protocol tests isolate caching/formatting; real geometry is checked below.
        detector.ring_supported = lambda *_: True
        return detector, calls

    def test_original_inference_retains_boxes_without_v4_edge_filter(self):
        detector, _ = self.make_detector([[0., 174., 320., 320., .94, 0.],
                                         [247., 86., 303., 135., .88, 0.]])
        result = detector.predict(np.zeros((320, 320), np.uint8), 10.)
        self.assertEqual(len(result), 2)
        np.testing.assert_equal(result[0]['xyxy'], [0., 174., 320., 320.])

    def test_ring_clipped_at_only_two_edges_is_retained(self):
        detector, _ = self.make_detector([[0., 0., 255., 212., .96, 0.]])
        self.assertEqual(len(detector.predict(np.zeros((320, 320), np.uint8), 10.)), 1)

    def test_grayscale_cache_preserves_capture_time_and_clock_reset_reinfers(self):
        detector, calls = self.make_detector([[100., 100., 220., 220., .9, 0.]])
        gray = np.arange(320*320, dtype=np.uint8).reshape(320, 320)
        detector.predict(gray, 10.)
        for channel in range(3):
            np.testing.assert_equal(calls[-1][0][:, :, channel], gray)
        self.assertEqual(calls[-1][1]['imgsz'], 320)
        self.assertEqual(calls[-1][1]['conf'], .40)
        count = len(calls)
        self.assertEqual(detector.predict(gray, 10.05)[0]['capture_seconds'], 10.)
        self.assertEqual(len(calls), count)
        self.assertEqual(detector.predict(gray, 10.1)[0]['capture_seconds'], 10.1)
        self.assertEqual(len(calls), count+1)
        count = len(calls)
        self.assertEqual(detector.predict(gray, 0.)[0]['capture_seconds'], 0.)
        self.assertEqual(len(calls), count+1)

    def test_non_ring_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ring class'):
            self.make_detector([], {0: 'up'})

    def test_combined_labels_use_single_raw_grayscale_inference(self):
        detector, calls = self.make_detector([[100., 100., 220., 220., .4, 4.],
            [20., 20., 40., 40., .9, 0.]], {0:'up', 1:'down', 2:'left', 3:'right', 4:'ring'})
        result = detector.predict(np.zeros((320, 320), np.uint8), 10.)
        self.assertEqual([d['label'] for d in result], ['ring', 'up'])
        self.assertEqual(calls[-1][1]['imgsz'], 320)
        self.assertEqual(calls[-1][1]['conf'], .40)
        self.assertEqual(len(calls), 2)  # Warmup plus this frame, no contrast retry.

    def test_wall_rectangle_and_blank_patch_are_rejected(self):
        gray = np.full((320, 320), 210, np.uint8)
        self.assertFalse(MissionYOLO.ring_supported(gray, [90, 90, 230, 230]))
        cv2.rectangle(gray, (100, 100), (220, 220), 10, 8)
        self.assertFalse(MissionYOLO.ring_supported(gray, [90, 90, 230, 230]))

    def test_visible_clipped_and_partially_occluded_rings_are_retained(self):
        for kind in ('visible', 'clipped', 'occluded'):
            with self.subTest(kind=kind):
                gray = np.full((320, 320), 210, np.uint8)
                center = (25, 160) if kind == 'clipped' else (160, 160)
                cv2.circle(gray, center, 60, 10, 7)
                if kind == 'occluded':
                    cv2.rectangle(gray, (190, 135), (230, 185), 210, -1)
                box = [0, 95, 90, 225] if kind == 'clipped' else [90, 90, 230, 230]
                self.assertTrue(MissionYOLO.ring_supported(gray, box))

    def test_small_round_detail_does_not_validate_large_wall_box(self):
        gray = np.full((320, 320), 210, np.uint8)
        cv2.circle(gray, (160, 160), 12, 10, 3)
        self.assertFalse(MissionYOLO.ring_supported(gray, [60, 60, 260, 260]))

    def test_larger_high_confidence_wall_cannot_replace_visible_ring(self):
        gray = np.full((320, 320), 210, np.uint8)
        cv2.rectangle(gray, (30, 110), (130, 210), 10, 8)
        cv2.circle(gray, (230, 160), 45, 10, 7)
        detector, _ = self.make_detector([[20, 100, 140, 220, .99, 0],
                                         [175, 105, 285, 215, .80, 0]])
        detector.ring_supported = MissionYOLO.ring_supported
        observer = MissionObserver(detector)
        observer.process(gray, 10.)
        self.assertEqual(len(observer.latest.visible_rings), 1)
        np.testing.assert_equal(observer.latest.ring['box_center'], [230., 160.])
        self.assertEqual(observer.ring_diagnostics['rejected_non_ring_shapes'], 1)


if __name__ == '__main__':
    unittest.main()
