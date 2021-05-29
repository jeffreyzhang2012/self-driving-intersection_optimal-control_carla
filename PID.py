import numpy as np
import time
import carla
class PID_base:
    def __init__(self,Ks):
        self.Kp = Ks[0]
        self.Ki = Ks[1]
        self.Kd = Ks[2]
        self.prev_error = 0
        self.last_time = time.time()
        self.cur_err = 0
        self.des = 0
        self.cum_err = 0

    def set_des(self,des):
        self.des = des

    def get_response(self,value):
        error = self.des - value
        cur_time = time.time()
        dt = cur_time - self.last_time
        self.last_time = cur_time
        if(self.cur_err * error) < 0: self.cur_err = 0
        self.cum_err += error
        out = self.Kp * error + self.Kd * (error - self.prev_error)/dt + self.Ki*self.cum_err*dt
        return out

class vwFollower:
    def __init__(self,vehicle):
        self.v_PID = PID_base([0.1,0,0])
        self.v_des = 0
        self.w_des = 0
        self.vehicle = vehicle

    def new_goal(self,goal):
        v_des = goal[0]
        w_des = goal[1]
        self.v_PID.set_des(v_des)
        self.w_des = w_des
        self.v_des = v_des

    def get_input(self):
        v_help = self.vehicle.get_velocity()
        v = (v_help.x**2+v_help.y**2)**0.5
        w = self.vehicle.get_angular_velocity().z
        steer = self.w_des
        throttle = v * 0.1 + self.v_PID.get_response(v)
        return throttle,steer
