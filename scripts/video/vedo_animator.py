# TODO: PR to Johnson's repo.
import numpy as np
from vedo import Plotter
from view_animator.base_animator import Orbiter
from math import cos, pi, sin


class VedoOrbiter(Orbiter):

    def __init__(self, plot: Plotter, objs_to_show, pitch=0, offset_yaw=0, dist=0.8, target=(0, 0, 0), ccw=True,
                 **kwargs):
        super().__init__(pitch, offset_yaw, dist, target, ccw, **kwargs)
        self.objs_to_show = objs_to_show
        self.plot = plot

    def generate_transform(self, dt):
        dt = dt + self.offset_yaw * 2 * pi / self.period

        x = self.dist * cos(2 * pi * dt / self.period) + self.target[0]
        y = self.dist * sin(2 * pi * dt / self.period) + self.target[1]
        z = self.target[2] + self.dist * sin(self.pitch)
        position = np.array([x, y, z])

        up = np.array([0, 0, 1])

        return position, up

    def update_transform(self, T):
        position, up = T

        self.plot.camera.SetPosition(position[0], position[1], position[2])
        self.plot.camera.SetFocalPoint(self.target[0], self.target[1], self.target[2])
        self.plot.camera.SetViewUp(up[0], up[1], up[2])
        self.plot.camera.SetDistance(self.dist)
        self.plot.camera.SetClippingRange(0.01, 1000)

        self.plot.render(resetcam=False)
