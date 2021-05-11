import numpy as np
import glob
import os
import sys
from sys import exit

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
from carEnv import *

class Controller:
    # goal
    def __init__(self,vehicle):
        self.vehicle = vehicle
        self.X = np.zeros(3)
        self.dX = np.zeros(3)
        # self.goal = np.zeros(2)
        self.arrived_goal = False

    def apply(self,throttle,steer):
        self.vehicle.apply_control(carla.VehicleControl(throttle, steer))

    def update_state(self):
        T = self.vehicle.get_transform()
        self.X[0] = T.location.x
        self.X[1] = T.location.y
        self.X[2] = T.rotation.yaw
        self.arrived_goal = np.linalg.norm(self.X[:2] - self.goal) < 2.0
        # p = np.linalg.norm(self.X[:2] - self.goal)
        # print(p,self.X[:2])

class V1Controller(Controller):
    def __init__(self,vehicle):
        super().__init__(vehicle)

class CurveFollowController(Controller):
    def __init__(self,vehicle):
        p1 = [255,-183]
        self.goal = np.array([269,-169])
        p2 = self.goal
        self.o = np.array([p2[0],p1[1]])
        self.r_d = (abs(p1[0]-p2[0])+abs(p1[1]-p2[1]))/2
        super().__init__(vehicle)

    def calculate_steer(self):
        r = np.linalg.norm(self.o-self.X[:2])
        steer = (self.r_d-r)/(self.r_d*0.4)
        return steer

    def calculate_throttle(self):
        return 0.5

    def step(self):
        super().update_state()
        super().apply(self.calculate_throttle(),self.calculate_steer())
