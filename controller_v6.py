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
from solver_v5 import mpc, get_hist
from PID import *

waypoints_all = np.zeros((3, 4, 2))  # #possible_paths X #waypoints X xy
waypoints_all[0] = np.array([[235, -169], [245.2, -169], [254.9, -158], [254.9, -148]])
waypoints_all[1] = np.array([[235, -169], [245.2, -169], [268, -169], [278, -169]])
waypoints_all[2] = np.array([[235, -169], [245.2, -169], [258.5, -183], [258.5, -193]])


class Controller:
    # goal
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.data_len = 1000000
        self.data_tick = 0
        self.X_hist = np.zeros((3, self.data_len))
        self.V_hist = np.zeros(self.data_len)
        self.C_hist = np.zeros(self.data_len)
        self.v = 0
        self.dX = np.zeros(3)
        self.X_init = np.zeros(3)
        self.arrived_goal = False
        self.init_state()
        self.mode = 0  # 0 == default, 1==follow traj
        self.traj = None
        self.traj_dt = None
        self.traj_start_time = None
        self.traj_12 = None
        self.sn = None
        self.prev_idx = None
        self.T = None
        self.vehicle_control = carla.VehicleControl()
        # self.init_PID()

    def init_state(self):
        T = self.vehicle.get_transform()
        self.X_init = np.array([T.location.x, T.location.y, T.rotation.yaw])

    def simple_apply(self, throttle, steer, brake):
        self.vehicle.apply_control(carla.VehicleControl(throttle, steer, brake))

    def apply2(self, throttle, steer):
        if self.v < 0.01:  # switch btw reverse and forward
            self.vehicle_control.reverse = bool(throttle < 0)
        if self.vehicle_control.reverse:  # reverse
            self.vehicle_control.throttle = max(0, -throttle)
            self.vehicle_control.brake = max(0, throttle)
        else:
            self.vehicle_control.throttle = max(0, throttle)
            self.vehicle_control.brake = max(0, -throttle)
            self.vehicle_control.steer = steer
        self.vehicle.apply_control(self.vehicle_control)
        return

    def apply(self, throttle, steer):
        '''
        if self.v < 0.01:  # switch btw reverse and forward
            self.vehicle_control.reverse = bool(throttle < 0)
        if self.vehicle_control.reverse:  # reverse
            self.vehicle_control.throttle = max(0, -throttle)
            self.vehicle_control.brake = max(0, throttle)
        else:
        '''
        self.vehicle_control.throttle = max(0, throttle)
        self.vehicle_control.brake = max(0, -throttle)
        self.vehicle_control.steer = steer
        # print(self.vehicle_control, throttle)
        self.vehicle.apply_control(self.vehicle_control)

    def update_state(self):
        self.T = self.vehicle.get_transform()
        self.X_hist[0, self.data_tick] = self.T.location.x
        self.X_hist[1, self.data_tick] = self.T.location.y
        self.X_hist[2, self.data_tick] = self.T.rotation.yaw
        v_help = self.vehicle.get_velocity()
        self.v = (v_help.x ** 2 + v_help.y ** 2) ** 0.5
        self.V_hist[self.data_tick] = self.v
        if self.data_tick + 1 == self.data_len:
            new_hist = np.zeros((3, self.data_len * 2))
            new_hist[:, 0:self.data_len] = copy.deepcopy(self.X_hist)
            self.data_len *= 2
            self.X_hist = copy.deepcopy(new_hist)
        self.arrived_goal = np.linalg.norm(self.X_hist[:2, self.data_tick] - self.goal) < 2.0
        # p = np.linalg.norm(self.X[:2] - self.goal)

    def set_follower_traj(self, traj, time_for_completion):
        self.traj = traj  # could be (v,w) or (x,y)
        self.traj_dt = time_for_completion / (traj.shape[0])
        self.mode = 1  # follow trajectory
        self.prev_idx = -1
        self.traj_start_time = time.time()
        return

    def get_hist(self):
        ## return init-zeroed trajectory
        return (self.X_hist[:, :self.data_tick].T - self.X_init[:, np.newaxis].T).T
        # return (self.X_hist[:, :self.data_tick][:, ::100].T - self.X_init[:, np.newaxis].T).T

    def get_input(self):
        idx = int((time.time() - self.traj_start_time) // self.traj_dt)
        if idx != self.prev_idx:
            try:
                goal = self.traj[idx]
                self.follower.new_goal(goal)
            except:
                self.mode = 0
                return self.get_default_inputs()
        return self.follower.get_input()
        # return

    def step(self):
        self.update_state()
        if self.mode == 1:
            # throttle,steer = self.get_input_traj_step()
            throttle, steer = self.get_input()
            self.get_input()
        else:
            throttle, steer = self.get_default_inputs()
        print(throttle, steer)
        self.apply(throttle, steer)
        self.data_tick += 1


class V1ControllerSimple(Controller):
    def __init__(self, vehicle):
        self.goal = np.array([300, -169])
        super().__init__(vehicle)

    def calculate_throttle(self, slow):
        if slow:
            return [0, 1.]
        else:
            return [0.3, 0]

    def calculate_steer(self):
        return 0

    def simple_step(self, slow):
        super().update_state()
        super().simple_apply(self.calculate_throttle(slow)[0],
                             self.calculate_steer(),
                             self.calculate_throttle(slow)[1])
        self.data_tick += 1


class V1Controller(Controller):
    def __init__(self, vehicle):
        self.path = random.randint(0, 2)
        self.path = 1
        self.waypoints = waypoints_all[self.path]
        self.goal = np.array([300, -169])
        self.path_len = 0
        self.point_per_len = 100
        self.d = 0
        super().__init__(vehicle)
        self.update_state()
        self.data_tick += 1
        self.F = np.array([0, 0])
        self.init_V1_traj()

    def init_V1_traj(self):
        N_straight_1 = 10 * self.point_per_len
        N_straight_2 = 10 * self.point_per_len
        if self.path != 1:
            o = [self.waypoints[1, 0], self.waypoints[2, 1]]
            r = np.mean(np.abs(self.waypoints[1] - self.waypoints[2]))
            d_curve = r * np.pi / 2
            N_curve = int(d_curve * self.point_per_len)
            t = np.linspace(-np.pi / 2, 0, N_curve) if self.path == 0 else np.linspace(np.pi / 2, 0, N_curve)
            s2 = np.zeros((N_curve, 2))  # x,y location
            s2[:, 0] = np.cos(t) * r + o[0]
            s2[:, 1] = np.sin(t) * r + o[1]
        else:
            d = np.linalg.norm(self.waypoints[2] - self.waypoints[1])
            n = int(d * self.point_per_len)
            s2 = np.linspace(self.waypoints[1], self.waypoints[2], n)
        s1 = np.linspace(self.waypoints[0], self.waypoints[1], N_straight_1)
        s3 = np.linspace(self.waypoints[2], self.waypoints[3], N_straight_2)
        self.xy_traj = np.vstack((s1, s2, s3))
        return

    def calculate_throttle(self):
        return self.F[0]

    def calculate_steer(self):
        return np.arctan2(self.F[1], self.F[0])

    def simple_step(self, slow):
        super().update_state()
        self.d += np.linalg.norm(self.X_hist[:2, self.data_tick] - self.X_hist[:2, self.data_tick - 1])
        F_global = self.xy_traj[int((self.d + 0.3) * self.point_per_len)]
        self.F = self.T.get_inverse_matrix() @ np.hstack((F_global, [0, 1]))
        # print("Force",self.F)
        # super().simple_apply(self.calculate_throttle(slow)[0],
        #                      self.calculate_steer(),
        #                      self.calculate_throttle(slow)[1])
        super().apply2(self.calculate_throttle(), self.calculate_steer())
        self.data_tick += 1


class CurveFollowController(Controller):
    def __init__(self, vehicle):
        p1 = [255, -183]
        self.goal = np.array([269, -169])
        p2 = self.goal
        self.o = np.array([p2[0], p1[1]])
        self.r_d = (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) / 2
        # self.hist = get_hist('vehicle2_traj.txt')
        super().__init__(vehicle)

    def calculate_steer(self):
        r = np.linalg.norm(self.o - self.X_hist[:2, self.data_tick])
        steer = (self.r_d - r) / (self.r_d * 0.25)
        return steer

    def calculate_control(self, traj_1, traj_2):
        u = mpc(hist=None, traj_1=traj_1, traj_2=traj_2, tick=self.data_tick, goal=self.goal)
        print('u=', u)
        if u[0] >= 0:
            return [u[0], u[1], 0.]
        else:
            return [0., u[1], -u[0]]

    def mpc_step(self, traj_1, traj_2):
        super().update_state()
        print('update')
        if traj_1 == 'constant':
            control = [.2, 0., 0.]
        else:
            control = self.calculate_control(traj_1, traj_2)
        super().simple_apply(control[0], control[1], control[2])
        self.data_tick += 1


class vwController(Controller):
    def __init__(self, vehicle):
        self.start = [255, -183]
        self.goal = np.array([269, -169])
        self.follower = vwFollower(vehicle)
        # N = 100
        # T = 1
        # t = np.linspace(0, T, N)
        # theta = np.cos(t * 2)
        # traj = np.zeros((N, 2))
        # traj[:, 0] = np.sin(t * 2) * 5
        # traj[:, 1] = theta * 0.2
        # traj[:,2] = np.zeros(N)
        super().__init__(vehicle)
        # self.set_follower_traj(traj, T)

    def get_default_inputs(self):
        return 0.0, 0.0

    def vw_step(self, traj_11, traj_10, traj_2):
        # super().update_state()
        N = 100
        T = 1
        traj = np.zeros((N, 2))
        if traj_11 == 'constant':
            u = [.2, 0.]
            traj[:, 0] = np.linspace(0, u[0] * 10, N)
            traj[:, 1] = np.linspace(0, u[1], N)
            self.C_hist[self.data_tick] = 0
            self.traj_12 = [self.start[1], 243]
            self.sn = [self.start[1], self.start[0]]
        else:
            s, sn, u, c, traj_12 = mpc(hist=None, traj_11=traj_11, traj_10=traj_10,
                                       traj_2=traj_2, tick=self.data_tick, goal=self.goal)
            # print(s)
            # vel = np.sqrt((s[0] - self.start[0]) ** 2 + (s[1] - self.start[1]) ** 2)
            vel = u[0] * 10
            traj[:, 0] = np.linspace(vel, vel, N)
            traj[:, 1] = np.linspace(u[1], u[1], N)
            self.C_hist[self.data_tick] = c
            self.traj_12 = traj_12
            self.sn = sn
        # print('u=', traj)
        # self.data_tick += 1
        super().set_follower_traj(traj, T)
        super().step()
        # return traj
