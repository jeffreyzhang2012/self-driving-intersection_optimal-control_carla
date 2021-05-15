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
import copy
import time
from carEnv import *

class Controller:
    # goal
    def __init__(self,vehicle):
        self.vehicle = vehicle
        self.data_len = 1000000
        self.data_tick = 0
        self.X_hist = np.zeros((3,self.data_len))
        self.dX = np.zeros(3)
        self.X_init = np.zeros(3)
        self.arrived_goal = False
        self.init_state()

    def init_state(self):
        T = self.vehicle.get_transform()
        self.X_init = np.array([T.location.x,T.location.y,T.rotation.yaw])

    def apply(self,throttle,steer):
        self.vehicle.apply_control(carla.VehicleControl(throttle, steer))

    def update_state(self):
        T = self.vehicle.get_transform()
        self.X_hist[0,self.data_tick] = T.location.x
        self.X_hist[1,self.data_tick] = T.location.y
        self.X_hist[2,self.data_tick] = T.rotation.yaw
        if self.data_tick + 1 == self.data_len:
            new_hist = np.zeros((3,self.data_len*2))
            new_hist[:,0:self.data_len] = copy.deepcopy(self.X_hist)
            self.data_len *= 2
            self.X_hist = copy.deepcopy(new_hist)
        self.arrived_goal = np.linalg.norm(self.X_hist[:2,self.data_tick] - self.goal) < 2.0
        # p = np.linalg.norm(self.X[:2] - self.goal)

    def get_traj(self):
        ## return init-zeroed trajectory

        return (self.X_hist[:,:self.data_tick][:,::100].T-self.X_init[:,np.newaxis].T).T

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
        r = np.linalg.norm(self.o-self.X_hist[:2,self.data_tick])
        steer = (self.r_d-r)/(self.r_d*0.25)
        return steer

    def calculate_throttle(self):
        return 0.5

    def step(self):
        super().update_state()
        super().apply(self.calculate_throttle(),self.calculate_steer())
        self.data_tick += 1
