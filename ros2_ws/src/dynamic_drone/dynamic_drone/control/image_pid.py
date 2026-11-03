"""Sampled image-centering PID shared by marker and ring alignment."""
import numpy as np


class ImageCenterPID:
    def __init__(self):
        self.kp = np.array([.55, .50])
        self.ki = np.array([.025, .035])
        self.kd = np.array([.08, .12])
        self.reset()

    def reset(self):
        self.key = None
        self.stamp = None
        self.error = np.zeros(2)
        self.integral = np.zeros(2)
        self.derivative = np.zeros(2)
        self.output = np.zeros(2)

    def update(self, center, matrix, stamp, key, lower, upper):
        error = -np.arctan((np.asarray(center)-matrix[:2, 2])/np.diag(matrix)[:2])
        if self.key != key or self.stamp is None or stamp-self.stamp > .5:
            self.reset()
            self.key, self.stamp, self.error = key, stamp, error
            self.output = np.clip(self.kp*error, lower, upper)
            return self.output.copy()
        # A cached image/YOLO box must not accumulate I or produce another D.
        if stamp <= self.stamp:
            return self.output.copy()
        dt = stamp-self.stamp
        filtered = self.error+dt/(.10+dt)*(error-self.error)
        derivative = (filtered-self.error)/dt
        self.derivative += dt/(.15+dt)*(derivative-self.derivative)
        integral = np.clip(self.integral+filtered*dt, -.3, .3)
        raw = self.kp*filtered+self.ki*integral+self.kd*self.derivative
        # Do not integrate farther into a saturated speed/rate command.
        winding_up = ((raw > upper) & (filtered > 0)) | ((raw < lower) & (filtered < 0))
        self.integral = np.where(winding_up, self.integral, integral)
        self.output = np.clip(self.kp*filtered+self.ki*self.integral+self.kd*self.derivative, lower, upper)
        self.stamp, self.error = stamp, filtered
        return self.output.copy()
