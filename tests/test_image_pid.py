"""Image-servo timing and damping regressions; no simulator or motor output."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ros2_ws/src/dynamic_drone'))
from dynamic_drone.control.image_pid import ImageCenterPID
from dynamic_drone.control.mission import FrontVisionMission

K = np.array([[234., 0., 160.], [0., 234., 160.], [0., 0., 1.]])


class ImagePIDTests(unittest.TestCase):
    def test_cached_box_does_not_change_integral_or_output(self):
        pid = ImageCenterPID()
        output = pid.update([180., 180.], K, 1., 'ring', [-.25, -.1], [.25, .22])
        integral = pid.integral.copy()
        for _ in range(10):
            np.testing.assert_equal(pid.update([180., 180.], K, 1., 'ring', [-.25, -.1], [.25, .22]), output)
        np.testing.assert_equal(pid.integral, integral)

    def test_new_target_discards_old_integral_and_derivative(self):
        pid = ImageCenterPID()
        for i in range(20):
            pid.update([190., 180.], K, 1.+i*.05, 'marker1', [-.25, -.1], [.25, .22])
        output = pid.update([160., 160.], K, 2., 'marker2', [-.25, -.1], [.25, .22])
        np.testing.assert_equal(output, [0., 0.])
        np.testing.assert_equal(pid.integral, [0., 0.])
        np.testing.assert_equal(pid.derivative, [0., 0.])

    def test_saturated_output_does_not_accumulate_integral(self):
        pid = ImageCenterPID()
        for i in range(100):
            output = pid.update([300., 300.], K, 1.+i*.05, 'marker', [-.02, -.02], [.02, .02])
            self.assertLessEqual(float(np.max(np.abs(output))), .02)
        np.testing.assert_equal(pid.integral, [0., 0.])

    def test_derivative_brakes_when_target_approaches_center(self):
        pid = ImageCenterPID()
        pid.update([200., 160.], K, 1., 'marker', [-.25, -.1], [.25, .22])
        output = pid.update([180., 160.], K, 1.1, 'marker', [-.25, -.1], [.25, .22])
        self.assertGreater(output[0], pid.kp[0]*pid.error[0])

    def test_forward_speed_is_continuous_at_alignment_boundary(self):
        pilot = FrontVisionMission()
        pilot.matrix = K.copy()
        inside = pilot.center_on_image([169.9, 160.], .5, 1., 'marker')
        outside = pilot.center_on_image([170.1, 160.], .5, 1.05, 'marker')
        self.assertGreater(inside[0], 0.)
        self.assertEqual(outside[0], 0.)
        self.assertLess(abs(inside[0]-outside[0]), .01)


if __name__ == '__main__':
    unittest.main()
