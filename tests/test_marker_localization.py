"""Offline regression checks. No ROS, Gazebo, camera windows or motor output.

Synthetic scene poses below generate test pixels only. The estimator receives
corners, timestamps, calibration and marker size, exactly as in the runtime.
Run: .venv-ai/bin/python -m unittest discover -s tests -v
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ros2_ws/src/dynamic_drone'))
from dynamic_drone.control.marker_odometry import Stage1MarkerOdometry
from dynamic_drone.control.marker_servo import ImageTarget
from dynamic_drone.control.mission import FrontVisionMission, MissionFrame
from dynamic_drone.perception.marker_pose import CAMERA_FROM_BODY, FrontMarkerLocalizer, rotation_distance

K = np.array([[233.8713, 0., 160.], [0., 233.8713, 160.], [0., 0., 1.]])
OFFSET = np.array([.035, 0., .042425])
TAG_ROTATION = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
TAG1 = np.array([4.9, -1.6, 1.2])
TAG2 = np.array([6., 1.6, 1.2])


def observation(odometry, stamp, position=None, tags=None, noise=0., rng=None, yaw=0.):
    position = np.zeros(3) if position is None else np.asarray(position, float)
    targets = {}
    rotation = cv2.Rodrigues(np.array([0., 0., yaw]))[0]
    camera_rotation = CAMERA_FROM_BODY @ rotation.T
    for identifier, center in (tags or {1: TAG1}).items():
        objects = (TAG_ROTATION @ odometry.object_points.T).T+center
        pixels = cv2.projectPoints(objects, cv2.Rodrigues(camera_rotation)[0],
            -camera_rotation @ (position+rotation @ OFFSET), K, np.zeros(5))[0].reshape(4, 2)
        if noise:
            pixels += rng.normal(0., noise, pixels.shape)
        target = ImageTarget(stamp, (pixels.mean(axis=0)-160)/160,
                            cv2.contourArea(pixels.astype(np.float32))/(320*320), pixels)
        target.decoded_stamp = stamp
        target.measurement_source = 'fresh_ArUco_decode'
        targets[identifier] = target
    return MissionFrame(np.zeros((320, 320), np.uint8), stamp, targets, None)


def initialized(tags=None):
    odometry = Stage1MarkerOdometry(K, np.zeros(5), .6, OFFSET)
    for i in range(20):
        odometry.update_markers(observation(odometry, 1.+i*.05, tags=tags))
    return odometry


class MarkerLocalizationTests(unittest.TestCase):
    def test_front_parallel_origin_normal_and_scale(self):
        odometry = initialized()
        self.assertTrue(odometry.valid)
        np.testing.assert_allclose(odometry.landmarks[1]['position'], TAG1, atol=1e-5)
        np.testing.assert_allclose(odometry.landmarks[1]['rotation'], TAG_ROTATION, atol=1e-5)

    def test_exact_trajectory_retains_translation_and_height(self):
        odometry = initialized()
        for i in range(1, 121):
            position = np.array([i*.005, 0., i*.005])
            self.assertTrue(odometry.update_markers(observation(odometry, 2.+i*.05, position)))
            self.assertLess(np.linalg.norm(odometry.position-position), .04)

    def test_noisy_corners_do_not_cause_sustained_loss(self):
        odometry = initialized()
        rng = np.random.default_rng(27)
        invalid = 0
        for i in range(180):
            position = np.array([i*.005, 0., min(i*.005, .6)])
            valid = odometry.update_markers(observation(odometry, 2.+i*.05, position, noise=.2, rng=rng))
            invalid += not valid
            if valid:
                self.assertLess(np.linalg.norm(odometry.position-position), .3)
        self.assertLessEqual(invalid, 10)

    def test_marker_handoff_does_not_feed_pose_into_own_anchor(self):
        odometry = initialized({1: TAG1, 2: TAG2})
        self.assertIn(2, odometry.landmarks)
        reference = odometry.marker_motion_references[2]['stamp']
        for i in range(1, 101):
            position = np.array([i*.01, 0., 0.])
            odometry.update_markers(observation(odometry, 2.+i*.05, position, {2: TAG2}))
        self.assertTrue(odometry.valid)
        self.assertLess(np.linalg.norm(odometry.position-position), .04)
        self.assertEqual(reference, odometry.marker_motion_references[2]['stamp'])

    def test_fixed_anchor_does_not_expire_during_long_visible_visit(self):
        odometry = initialized({1: TAG1, 2: TAG2})
        reference = odometry.marker_motion_references[2]['stamp']
        for i in range(1, 701):
            position = np.array([i*.001, 0., 0.])
            odometry.update_markers(observation(odometry, 2.+i*.05, position, {2: TAG2}))
        self.assertTrue(odometry.valid)
        self.assertIn(2, odometry.relative_pose_ids)
        self.assertLess(np.linalg.norm(odometry.position-position), .02)
        self.assertEqual(reference, odometry.marker_motion_references[2]['stamp'])

    def test_two_markers_and_depth_stage_keep_same_coordinate_frame(self):
        odometry = initialized({1: TAG1, 2: TAG2})
        for i in range(1, 161):
            position = np.array([i*.01, 0., i*.002])
            odometry.update_markers(observation(odometry, 2.+i*.05, position, {1: TAG1, 2: TAG2}))
        self.assertTrue(odometry.valid)
        self.assertEqual(odometry.pose_method, 'multi_marker_RANSAC_refined_PnP')
        self.assertLess(np.linalg.norm(odometry.position-position), .04)
        bridge = odometry.seed_depth_odometry()
        np.testing.assert_array_equal(bridge.position, odometry.position)
        np.testing.assert_array_equal(bridge.rotation, odometry.rotation)
        np.testing.assert_array_equal(bridge.velocity, odometry.velocity)

    def test_weak_markers_use_joint_pose_after_initial_anchor_leaves(self):
        tags = {1: TAG1, 2: TAG2, 3: np.array([6.5, -2., 1.2])}
        odometry = initialized(tags)
        self.assertIn(3, odometry.landmarks)
        visible = {i: tags[i] for i in (2, 3)}
        rng = np.random.default_rng(31)
        for i in range(1, 81):
            position = np.array([i*.005, 0., i*.003])
            odometry.update_markers(observation(odometry, 2.+i*.05, position,
                                               visible, noise=.2, rng=rng))
        self.assertTrue(odometry.valid)
        self.assertEqual(odometry.pose_method, 'multi_marker_RANSAC_refined_PnP')
        self.assertLess(np.linalg.norm(odometry.position-position), .10)

    def test_weak_anchor_cannot_reanchor_an_excluded_weak_marker(self):
        tags = {1: TAG1, 2: TAG2, 3: np.array([6.5, -2., 1.2])}
        odometry = initialized(tags)
        reference = odometry.marker_motion_references[3]
        reference['position'] += np.array([.5, 0., 0.])
        stamp, position = reference['stamp'], reference['position'].copy()
        frame = observation(odometry, 2., tags={i: tags[i] for i in (2, 3)})
        with patch.object(odometry, 'joint_pose', return_value=None):
            self.assertTrue(odometry.update_markers(frame))
        self.assertEqual(odometry.supported_ids, [2])

        # Reacquisition heading must use that same anchor too.
        np.testing.assert_allclose(odometry.memory()[2]['position_m'], TAG2, atol=1e-5)
        self.assertEqual(odometry.marker_motion_references[3]['stamp'], stamp)
        np.testing.assert_array_equal(odometry.marker_motion_references[3]['position'], position)

    def test_joint_to_single_pose_uses_same_refreshed_anchors(self):
        tags = {1: TAG1, 2: TAG2, 3: np.array([6.5, -2., 1.2])}
        odometry = initialized(tags)
        # A distant first view can leave biased map geometry even after a
        # closer independent view refreshes the relative marker references.
        for identifier in (2, 3):
            odometry.landmarks[identifier]['position'] += np.array([1.4, 0., .55])
        for i in range(1, 11):
            position = np.array([i*.01, 0., 0.])
            visible = {j: tags[j] for j in ((2, 3) if i <= 5 else (2,))}
            frame = observation(odometry, 2.+i*.05, position, visible)
            self.assertTrue(odometry.update_markers(frame))
            self.assertLess(np.linalg.norm(odometry.position-position), .04)
        self.assertEqual(odometry.supported_ids, [2])

    def test_duplicate_and_backward_frames_never_refresh_pose(self):
        for stamp in (1.95, 1., float('nan')):
            odometry = initialized()
            accepted = odometry.stamp
            snapshot = odometry.last_valid_marker_snapshot.copy()
            self.assertFalse(odometry.update_markers(observation(odometry, stamp)))
            self.assertEqual(odometry.reason, 'nonmonotonic_marker_timestamp')
            self.assertEqual(accepted, odometry.stamp)
            self.assertEqual(snapshot, odometry.last_valid_marker_snapshot)

    def test_missing_image_keeps_history_but_no_current_pose(self):
        odometry = initialized()
        stamp = odometry.stamp
        self.assertFalse(odometry.update_markers(None))
        self.assertIsNone(odometry.telemetry()['estimated_position_m'])
        self.assertEqual(odometry.last_valid_marker_snapshot['stamp'], stamp)
        self.assertIsNone(odometry.control_altitude(stamp+.1))

    def test_pose_jump_is_accepted_without_resetting_origin(self):
        odometry = initialized()
        center = odometry.landmarks[1]['position'].copy()
        accepted_position = odometry.position.copy()
        for i in range(20):
            self.assertTrue(odometry.update_markers(observation(odometry, 2.+i*.05, [0., 0., 1.])))
        np.testing.assert_array_equal(odometry.landmarks[1]['position'], center)
        self.assertGreater(np.linalg.norm(odometry.position-accepted_position), .2)

    def test_normal_yaw_motion_remains_observable(self):
        odometry = initialized()
        for i in range(1, 41):
            yaw = -i*.0075
            frame = observation(odometry, 2.+i*.05, yaw=yaw)
            self.assertTrue(odometry.update_markers(frame))
            self.assertLess(rotation_distance(odometry.rotation, cv2.Rodrigues(np.array([0., 0., yaw]))[0]), .03)





    def test_tracked_corners_cannot_create_origin(self):
        odometry = Stage1MarkerOdometry(K, np.zeros(5), .6, OFFSET)
        for i in range(12):
            frame = observation(odometry, 1.+i*.05)
            frame.decoded_marker_ids.clear()
            frame.tracked_marker_ids.add(1)
            self.assertFalse(odometry.update_markers(frame))
        self.assertFalse(odometry.landmarks)

    def test_small_and_nonfinite_quads_are_rejected(self):
        odometry = initialized()
        for corners in (np.array([[100., 100.], [105., 100.], [105., 105.], [100., 105.]]),
                        np.full((4, 2), np.nan)):
            frame = observation(odometry, 2.)
            frame.markers[1].corners = corners
            self.assertFalse(odometry.poses(1, frame.markers[1]))


    def test_isolated_localizer_uses_same_correct_planar_solution(self):
        localizer = FrontMarkerLocalizer(K, np.zeros(5), (320, 320), 1, .6, OFFSET)
        odometry = initialized()
        for i in range(12):
            frame = observation(odometry, 1.+i*.05)
            localizer.observe(frame.markers[1].corners, frame.stamp)
        for i in range(1, 41):
            position = np.array([i*.005, 0., i*.005])
            frame = observation(odometry, 2.+i*.05, position)
            pose = localizer.observe(frame.markers[1].corners, frame.stamp)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.position, position, atol=1e-5)


class MarkerRecoveryControlTests(unittest.TestCase):
    def pilot(self):
        pilot = FrontVisionMission()
        pilot.configure_camera(SimpleNamespace(k=K.ravel(), d=np.zeros(5)))
        pilot.stage1_odometry = initialized()
        pilot.odometry = pilot.stage1_odometry
        pilot.phase = 'S1_MARKERS'
        return pilot

    def ready_frame(self, pilot, stamp, distances):
        tags = {i: np.array([distance, 0., 1.5]) for i, distance in distances.items()}
        frame = observation(pilot.odometry, stamp, [0., 0., 1.5], tags)
        pilot.odometry.position = np.array([0., 0., 1.5])
        pilot.odometry.velocity[:] = 0.
        pilot.odometry.stamp = stamp
        pilot.odometry.altitude_reference = dict(height_m=1.5, stamp=stamp, marker_ids=[1])
        pilot.markers = {i: dict(target=frame.markers[i], body=np.array([distance, 0., 0.]),
                                 source='OpenCV_refined_square_PnP_known_size')
                         for i, distance in distances.items()}
        pilot.depth.latest = dict(stamp=stamp, front=min(distances.values()), left=None, right=None)
        return frame

    def test_decoded_next_marker_is_ready_without_vote_wait(self):
        pilot = self.pilot()
        frame = self.ready_frame(pilot, 2., {1: 2.7, 2: 6.})
        pilot.target_marker_id = 2
        pilot.target_marker_activated_stamp = 2.
        pilot.route_id_votes[2] = (2., 0)
        self.assertTrue(pilot.confirm_current_target(frame))
        self.assertFalse(pilot.confirm_target_vicinity(frame, True))

    def test_handoff_commands_next_marker_in_same_control_cycle(self):
        pilot = self.pilot()
        for i in range(5):
            stamp = 2.+i*.05
            frame = self.ready_frame(pilot, stamp, {1: 2.7, 2: 6.})
            command, _ = pilot.command(frame, stamp)
        self.assertEqual(pilot.route_completion_order, [1])
        self.assertEqual(pilot.target_marker_id, 2)
        self.assertTrue(pilot.confirm_current_target(frame))
        self.assertEqual(pilot.stage1_forward_observation['marker_id'], 2)
        self.assertGreater(command[0], 0.)

    def test_aligned_stage1_reaches_increased_forward_speed(self):
        pilot = self.pilot()
        for i in range(80):
            stamp = 2.+i*.05
            frame = self.ready_frame(pilot, stamp, {1: 5.})
            command, _ = pilot.command(frame, stamp)
        self.assertAlmostEqual(command[0], .50)
        self.assertEqual(command[1], 0.)



    def test_empty_image_search_has_no_timeout(self):
        pilot = self.pilot()
        pilot.odometry.update_markers(MissionFrame(np.zeros((320, 320), np.uint8), 2., {}, None))
        frame = MissionFrame(np.zeros((320, 320), np.uint8), 2., {}, None)
        command, _ = pilot.command(frame, 2.)
        np.testing.assert_array_equal(command[:3], np.zeros(3))
        self.assertLessEqual(abs(command[3]), .25)
        frame.stamp = 33.
        command, _ = pilot.command(frame, 33.)
        self.assertEqual(pilot.phase, 'S1_MARKERS')
        np.testing.assert_array_equal(command[:3], np.zeros(3))

    def test_stale_camera_never_moves(self):
        pilot = self.pilot()
        pilot.odometry.valid = False
        frame = observation(pilot.odometry, 2.)
        command, _ = pilot.command(frame, 3.)
        np.testing.assert_array_equal(command, np.zeros(4))

    def test_visible_marker_image_controls_vertical_alignment(self):
        pilot = self.pilot()
        frame = self.ready_frame(pilot, 2., {1: 5.})
        pilot.odometry.position[2] = 2.2
        pilot.odometry.altitude_reference['height_m'] = 2.2
        command, _ = pilot.command(frame, 2.)
        self.assertEqual(pilot.phase, 'S1_MARKERS')
        self.assertGreater(command[0], 0.)
        self.assertLess(command[0], .5)
        self.assertLess(command[2], 0.)
        self.assertEqual(pilot.stage1_altitude_control['source'], 'marker_image_center')

    def test_off_center_visible_marker_keeps_forward_motion_and_alignment(self):
        pilot = self.pilot()
        frame = self.ready_frame(pilot, 2., {1: 5.})
        frame.markers[1].corners[:, 0] = 5.
        frame.markers[1].area_fraction = .2
        pilot.depth.latest['front'] = .5
        command, _ = pilot.command(frame, 2.)
        self.assertGreater(command[0], 0.)
        self.assertNotEqual(command[3], 0.)

    def test_near_marker_is_not_completed_until_image_centered(self):
        pilot = self.pilot()
        frame = self.ready_frame(pilot, 2., {1: 2.7})
        frame.markers[1].corners[:, 1] += 35.
        command, _ = pilot.command(frame, 2.)
        self.assertEqual(pilot.route_completion_order, [])
        self.assertEqual(command[0], 0.)
        self.assertLess(command[2], 0.)

    def test_standoff_arrival_is_latched_while_centering(self):
        pilot = self.pilot()
        pilot.previous_command[0] = .5
        for stamp, distance in [(2., 3.45), (2.05, 3.55), (2.10, 3.48)]:
            frame = self.ready_frame(pilot, stamp, {1: distance})
            frame.markers[1].corners[:, 1] += 35.
            command, _ = pilot.command(frame, stamp)
            self.assertEqual(command[0], 0.)
            self.assertTrue(pilot.target_visit_status['reached_standoff'])
            self.assertEqual(pilot.route_completion_order, [])

    def test_marker_can_complete_from_farther_standoff(self):
        pilot = self.pilot()
        frame = self.ready_frame(pilot, 2., {1: 3.45, 2: 7.})
        command, _ = pilot.command(frame, 2.)
        self.assertEqual(pilot.route_completion_order, [1])
        self.assertEqual(pilot.target_marker_id, 2)
        self.assertGreater(command[0], 0.)

    def test_visible_marker_error_does_not_restart_acceleration(self):
        pilot = self.pilot()
        speeds = []
        for i in range(80):
            stamp = 2.+i*.05
            frame = self.ready_frame(pilot, stamp, {1: 5.})
            frame.markers[1].corners += np.array([(-1)**i*30., (-1)**i*20.])
            command, _ = pilot.command(frame, stamp)
            speeds.append(command[0])
        self.assertTrue(np.all(np.diff(speeds) >= -1e-12))
        self.assertAlmostEqual(speeds[-1], .50)
        self.assertLessEqual(float(np.max(np.diff(speeds))/.05), .20+1e-12)
        acceleration = np.diff(np.r_[0., speeds])/.05
        jerk = np.diff(np.r_[0., acceleration])/.05
        self.assertLessEqual(float(np.max(np.abs(jerk))), .40+1e-9)
        self.assertEqual(pilot.route_completion_order, [])

    def test_visible_next_marker_handoff_preserves_cruise_speed(self):
        pilot = self.pilot()
        pilot.previous_command[0] = .50
        frame = self.ready_frame(pilot, 2., {1: 2.7, 2: 6.})
        frame.markers[2].corners[:, 0] += 40.
        command, _ = pilot.command(frame, 2.)
        self.assertEqual(pilot.target_marker_id, 2)
        self.assertAlmostEqual(command[0], .50)

    def test_image_follow_continues_when_metric_pose_is_unavailable(self):
        pilot = self.pilot()
        frame = self.ready_frame(pilot, 2., {1: 5.})
        pilot.odometry.valid = False
        command, _ = pilot.command(frame, 2.)
        self.assertGreater(command[0], 0.)
        self.assertLess(command[0], .5)
        self.assertLess(command[2], 0.)

    def test_depth_cannot_replace_missing_stage1_pose(self):
        pilot = self.pilot()
        pilot.odometry.update_markers(None)
        frame = observation(pilot.odometry, 2.)
        pilot.depth.update(np.full((62, 82), 3.), frame, frame.gray.shape)
        self.assertIsNone(pilot.depth.latest)
        self.assertFalse(pilot.odometry.valid)
        self.assertIs(pilot.odometry, pilot.stage1_odometry)


class TruckFollowingTests(unittest.TestCase):
    def pilot(self):
        pilot = FrontVisionMission()
        pilot.matrix = K.copy()
        pilot.transition('S3_FIND_TRUCK', 1., 'test')
        self.flag = SimpleNamespace(object_points=np.array(
            [[-.2, .2, 0.], [.2, .2, 0.], [.2, -.2, 0.], [-.2, -.2, 0.]]))
        return pilot

    def frame(self, stamp, flag=(1.5, 0., 0.), position=None, yaw=0.):
        return observation(self.flag, stamp, position=position,
                           tags={23: np.asarray(flag)}, yaw=yaw)

    def test_relative_speed_matches_without_depth_or_world_odometry(self):
        pilot = self.pilot()
        frame = self.frame(1., flag=(3., 0., .2))
        pilot.command(frame, 1.)
        self.assertEqual(pilot.phase, 'S3_FOLLOW_TRUCK')
        pilot.command(frame, 1.02)
        self.assertFalse(pilot.landing_attempted)  # Duplicate image is not velocity.
        command, _ = pilot.command(self.frame(1.05, flag=(3., 0., .2)), 1.05)
        self.assertEqual(pilot.phase, 'S3_DESCEND')
        self.assertTrue(pilot.landing_attempted)
        self.assertLessEqual(command[2], 0.)
        self.assertIsNone(pilot.odometry)
        self.assertIsNone(pilot.depth.latest)
        pilot.depth.update(np.full((62, 82), .2), frame, frame.gray.shape)
        self.assertIsNone(pilot.depth.latest)
        self.assertEqual(pilot.depth.reason, 'truck_relative_flag_PnP_depth_display_only')
        self.assertFalse(pilot.landing_confirmed)

    def test_unmatched_motion_follows_in_both_directions(self):
        for direction in (-1., 1.):
            with self.subTest(direction=direction):
                pilot = self.pilot()
                pilot.command(self.frame(1.), 1.)
                for i in range(1, 11):
                    command, _ = pilot.command(self.frame(1.+i*.05,
                        flag=(1.5+direction*i*.01, 0., 0.)), 1.+i*.05)
                self.assertEqual(pilot.phase, 'S3_FOLLOW_TRUCK')
                self.assertGreater(direction*command[0], 0.)
                self.assertGreater(pilot.truck_status['relative_horizontal_speed_m_s'], .15)

    def test_common_motion_and_body_rotation_do_not_create_relative_speed(self):
        pilot = self.pilot()
        pilot.command(self.frame(1.), 1.)
        command, _ = pilot.command(self.frame(1.05, flag=(1.51, 0., 0.),
                                            position=(.01, 0., 0.), yaw=.03), 1.05)
        self.assertEqual(pilot.phase, 'S3_DESCEND')
        self.assertLess(pilot.truck_status['relative_horizontal_speed_m_s'], .001)
        self.assertLess(command[3], 0.)

    def test_reciprocating_vehicle_retains_standoff_while_descending(self):
        pilot = self.pilot()
        position, velocity = np.zeros(3), 0.
        errors, forward, reverse = [], [], []
        # Horizontal closed loop with a lagging velocity actuator. Vertical
        # motion is isolated here; contact/flag loss have separate checks.
        for i in range(241):
            elapsed = i*.1
            truck_x = .15*12/np.pi*np.sin(np.pi*elapsed/12.)
            command, _ = pilot.command(self.frame(1.+elapsed,
                flag=(1.5+truck_x, 0., 0.), position=position), 1.+elapsed)
            velocity += .1/(.35+.1)*(command[0]-velocity)
            position[0] += .1*velocity
            errors.append(abs(1.5+truck_x-position[0]-1.5))
            if 2. <= elapsed <= 4.:
                forward.append(command[0])
            if 10. <= elapsed <= 14.:
                reverse.append(command[0])
            if pilot.landing_attempted:
                self.assertLess(command[2], 0.)
        self.assertLess(max(errors), .20)
        self.assertGreater(min(forward), 0.)
        self.assertLess(max(reverse), 0.)
        self.assertTrue(pilot.landing_attempted)
        self.assertFalse(pilot.landing_confirmed)

    def test_missing_or_stale_flag_holds_without_fabricating_velocity(self):
        pilot = self.pilot()
        frame = self.frame(1.)
        pilot.command(frame, 1.)
        command, _ = pilot.command(frame, 1.4)
        np.testing.assert_array_equal(command, np.zeros(4))
        self.assertFalse(pilot.landing_attempted)
        pilot.command(self.frame(1.5), 1.5)
        self.assertFalse(pilot.landing_attempted)  # Reacquire before differentiating.
        empty = MissionFrame(np.zeros((320, 320), np.uint8), 1.55, {}, None)
        command, _ = pilot.command(empty, 1.55)
        np.testing.assert_array_equal(command, np.zeros(4))

    def tof(self, pilot, stamp, distance):
        from dynamic_drone.control.flow_deck import FlowDeckAssist
        if pilot.tof_sensor is None:
            pilot.tof_sensor = FlowDeckAssist()
        pilot.tof_transport_fresh = True
        pilot.tof_sensor.update_range(stamp, distance)
        pilot.tof_sensor.update_attitude(stamp, [0., 0., 0., 1.])

    def enter_deck(self, pilot):
        for stamp, distance in [(1., 1.2), (1.05, .6)]:
            self.tof(pilot, stamp, distance)
            pilot.command(self.frame(stamp, flag=(1.5, 0., .8)), stamp)
        self.assertTrue(pilot.truck_tof['over_deck'])

    def near_deck(self, pilot):
        for stamp in (1.08, 1.11):
            self.tof(pilot, stamp, .05)
            pilot.update_truck_tof(stamp)

    def test_final_descent_is_bounded_and_never_confirms_landing_by_time(self):
        pilot = self.pilot()
        self.enter_deck(pilot)
        command, _ = pilot.command(None, 1.5)
        self.assertLess(command[2], 0.)
        self.assertTrue(pilot.truck_status['final_descent_using_last_command'])
        command, _ = pilot.command(None, 2.6)
        np.testing.assert_array_equal(command, np.zeros(4))
        self.assertFalse(pilot.landing_confirmed)
        self.assertEqual(pilot.truck_observation['stamp'], 1.05)

    def test_only_tof_backed_deck_touchdown_completes_and_stops(self):
        pilot = self.pilot()
        self.enter_deck(pilot)
        deck = ['world::mission_vehicle::chassis::collision', 'crazyflie::body::collision']
        self.assertFalse(pilot.confirm_truck_contact(deck, [(0., 0., -1.)], 1.06))
        self.near_deck(pilot)
        # The downward sensor can fall below its 2 cm minimum just before contact.
        self.tof(pilot, 1.13, float('nan'))
        self.assertFalse(pilot.confirm_truck_contact(deck, [(1., 0., 0.)], 1.15))
        self.assertFalse(pilot.confirm_truck_contact(['ground_plane::collision'], [(0., 0., 1.)], 1.15))
        self.assertTrue(pilot.confirm_truck_contact(deck, [(0., 0., -1.)], 1.15))
        self.assertTrue(pilot.landing_confirmed)
        self.assertEqual(pilot.phase, 'COMPLETE')
        command, _ = pilot.command(None, 1.2)
        np.testing.assert_array_equal(command, np.zeros(4))

    def test_missing_tof_never_confirms_deck_or_blind_descent(self):
        pilot = self.pilot()
        pilot.command(self.frame(1., flag=(1.5, 0., .8)), 1.)
        pilot.command(self.frame(1.05, flag=(1.5, 0., .8)), 1.05)
        command, _ = pilot.command(None, 1.5)
        np.testing.assert_array_equal(command, np.zeros(4))
        self.assertFalse(pilot.confirm_truck_contact(
            ['mission_vehicle::chassis::collision'], [(0., 0., 1.)], 1.5))

    def test_slow_descent_or_unaligned_step_is_not_deck_entry(self):
        for unaligned in (False, True):
            pilot = self.pilot()
            for i in range(8):
                stamp = 1.+i*.05
                self.tof(pilot, stamp, 1.2-(.6 if i else 0.) if unaligned else 1.2-i*.04)
                pilot.command(self.frame(stamp, flag=(3. if unaligned else 1.5, 0., .8)), stamp)
            self.assertFalse(pilot.truck_tof['over_deck'])

    def test_leaving_deck_revokes_entry_and_stale_near_cannot_land(self):
        pilot = self.pilot()
        self.enter_deck(pilot)
        self.tof(pilot, 1.10, 1.2)
        pilot.update_truck_tof(1.10)
        self.assertFalse(pilot.truck_tof['over_deck'])
        pilot = self.pilot()
        self.enter_deck(pilot)
        self.near_deck(pilot)
        self.assertFalse(pilot.confirm_truck_contact(
            ['mission_vehicle::chassis::collision'], [(0., 0., 1.)], 1.9))

    def test_duplicate_near_range_does_not_count_twice(self):
        pilot = self.pilot()
        self.enter_deck(pilot)
        self.tof(pilot, 1.1, .05)
        pilot.update_truck_tof(1.1)
        pilot.update_truck_tof(1.11)
        self.assertIsNone(pilot.truck_tof['last_near_stamp'])


class FlowDeckAssistanceTests(unittest.TestCase):
    def setUp(self):
        from dynamic_drone.control.flow_deck import FlowDeckAssist
        self.flow = FlowDeckAssist()
        self.k = np.array([[151.4, 0., 64.], [0., 151.4, 64.], [0., 0., 1.]])
        self.flow.configure(self.k)
        self.image = cv2.GaussianBlur(np.random.default_rng(40).integers(
            0, 256, (128, 128), dtype=np.uint8), (3, 3), .6)

    def sample(self, stamp, distance=1., image=None, quaternion=(0., 0., 0., 1.)):
        self.flow.update_range(stamp, distance)
        self.flow.update_attitude(stamp, quaternion)
        if image is not None:
            self.flow.update_image(image, stamp)

    def moving(self, vx=.3, vy=-.2):
        self.sample(1., image=self.image)
        displacement = np.array([[1., 0., vy*.05*151.4], [0., 1., vx*.05*151.4]])
        self.sample(1.05, image=cv2.warpAffine(self.image, displacement, (128, 128)))

    def test_measured_flow_sign_scale_and_stationary_damping(self):
        self.moving()
        np.testing.assert_allclose(self.flow.velocity, [.3, -.2, 0.], atol=.025)
        command = self.flow.apply(np.zeros(4), 'S2_ALIGN_RING', 1.05)
        self.assertLess(command[0], 0.)
        self.assertGreater(command[1], 0.)
        self.assertLessEqual(np.max(np.abs(command[:2])), .05)

    def test_pure_attitude_change_is_not_translation(self):
        from dynamic_drone.control.flow_deck import CAMERA_FROM_BODY, CAMERA_OFFSET
        self.sample(1., image=self.image)
        rotvec = np.array([.012, -.018, .025])
        rotation = cv2.Rodrigues(rotvec)[0]
        angle = np.linalg.norm(rotvec)
        quaternion = np.r_[rotvec/angle*np.sin(angle/2.), np.cos(angle/2.)]
        # Render a stationary horizontal floor from two camera poses, with
        # zero body translation and the physical camera lever arm included.
        camera_rotation = CAMERA_FROM_BODY @ rotation.T @ CAMERA_FROM_BODY.T
        camera_translation = CAMERA_FROM_BODY @ rotation.T @ (CAMERA_OFFSET-rotation @ CAMERA_OFFSET)
        homography = self.k @ (camera_rotation+np.outer(camera_translation, [0., 0., 1.])) @ np.linalg.inv(self.k)
        distance = (1.-CAMERA_OFFSET[2]+(rotation @ CAMERA_OFFSET)[2])/rotation[2, 2]
        moved = cv2.warpPerspective(self.image, homography, (128, 128))
        self.sample(1.05, distance, moved, quaternion)
        self.assertIsNotNone(self.flow.velocity_stamp, self.flow.reason)
        self.assertLess(np.linalg.norm(self.flow.velocity), .035)

    def test_duplicate_image_and_invalid_range_never_refresh_flow(self):
        self.moving()
        stamp = self.flow.velocity_stamp
        self.flow.update_image(self.image, 1.05)
        self.assertEqual(self.flow.velocity_stamp, stamp)
        command = np.zeros(4)
        np.testing.assert_array_equal(self.flow.apply(command, 'S2_ALIGN_RING', 1.3), command)
        self.sample(1.10, float('nan'))
        np.testing.assert_array_equal(self.flow.apply(command, 'S2_ALIGN_RING', 1.10), command)
        self.assertFalse(self.flow.telemetry()['velocity_valid'])

    def test_blank_texture_and_surface_steps_disable_feedback(self):
        blank = np.zeros((128, 128), np.uint8)
        self.sample(1., image=blank)
        self.sample(1.05, image=blank)
        self.assertIsNone(self.flow.velocity_stamp)
        self.assertEqual(self.flow.reason, 'flow_insufficient_floor_texture')
        self.sample(1.1, distance=.5, image=self.image)
        self.assertIsNone(self.flow.velocity_stamp)

    def test_ring_crossing_and_truck_follow_commands_are_preserved(self):
        self.moving()
        for phase in ('S2_CROSS_RING', 'S3_FOLLOW_TRUCK', 'ABORT', 'COMPLETE'):
            with self.subTest(phase=phase):
                command = np.array([.4, 0., 0., 0.])
                np.testing.assert_array_equal(self.flow.apply(command, phase, 1.05), command)

    def test_search_height_hold_uses_local_range_and_rejects_surface_jump(self):
        self.sample(1.)
        self.flow.apply(np.zeros(4), 'S2_FIND_RING', 1., searching=True)
        self.sample(1.05, .97)
        command = self.flow.apply(np.zeros(4), 'S2_FIND_RING', 1.05, searching=True)
        self.assertGreater(command[2], 0.)
        self.sample(1.1, .6)
        command = self.flow.apply(np.zeros(4), 'S2_FIND_RING', 1.1, searching=True)
        self.assertEqual(command[2], 0.)
        # Deliberate visual climb must take priority over range hold.
        command = np.array([0., 0., .1, 0.])
        np.testing.assert_array_equal(self.flow.apply(command, 'S1_MARKERS', 1.1, searching=True), command)

    def test_touchdown_softening_preserves_vehicle_horizontal_follow(self):
        self.sample(1., .20)
        command = np.array([-.15, .03, -.10, .02])
        actual = self.flow.apply(command, 'S3_DESCEND', 1., landing_observed=True)
        np.testing.assert_array_equal(actual[[0, 1, 3]], command[[0, 1, 3]])
        self.assertEqual(actual[2], -.04)
        np.testing.assert_array_equal(self.flow.apply(command, 'S3_DESCEND', 1.3, landing_observed=True), command)
        np.testing.assert_array_equal(self.flow.apply(command, 'S3_DESCEND', 1., landing_observed=False), command)

    def test_policy_requires_explicit_assist_and_rejects_unknown_flow(self):
        import xml.etree.ElementTree as ET
        from dynamic_drone.safety.sensor_policy import validate_sdf, validate_bridges
        sensor = ET.fromstring('<sensor name="flow_deck_camera" type="camera"><topic>/vision_trial/flow/image</topic></sensor>')
        with self.assertRaises(RuntimeError):
            validate_sdf(sensor)
        validate_sdf(sensor, allow_flow_deck=True)
        sensor.set('name', 'old_flow_proxy')
        with self.assertRaises(RuntimeError):
            validate_sdf(sensor, allow_flow_deck=True)
        bridge = ('/trial/flow/range', '/vision_trial/flow/range', 'sensor_msgs/msg/LaserScan', 'gz.msgs.LaserScan', 'GZ_TO_ROS')
        validate_bridges([bridge], allow_flow_deck=True)
        with self.assertRaises(RuntimeError):
            validate_bridges([bridge])
        with self.assertRaises(RuntimeError):
            validate_bridges([bridge[:-1]+('ROS_TO_GZ',)], allow_flow_deck=True)


if __name__ == '__main__':
    unittest.main()
