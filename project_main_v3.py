import glob
import os
import sys
from sys import exit
import matplotlib.pyplot as plt
import numpy as np


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
import multiprocessing
import threading

def v1_control(env):
    while not (env.c2.arrived_goal or env.c1.arrived_goal):
        env.c1.simple_step(slow=False)
        time.sleep(0.001)
    return

def v2_control(env):
    while not (env.c2.arrived_goal or env.c1.arrived_goal):
        env.c2.step()
        time.sleep(0.001)
    return

def v2_opt(env):
    traj_10 = None
    while not (env.c2.arrived_goal or env.c1.arrived_goal):
        if env.c2.data_tick == 0:
            env.c2.opt_step('constant', None, None)
        else:
            traj_11 = env.c1.X_hist[:2, max(env.c1.data_tick-1, 0)]
            traj_11 = np.array([traj_11[1], traj_11[0]])
            traj_2 = env.c2.X_hist[:, max(env.c2.data_tick-1, 0)]
            traj_2 = np.array([traj_2[1], traj_2[0], traj_2[2] / 180 * np.pi])
            env.c2.opt_step(traj_11, traj_10, traj_2)
            traj_10 = traj_11
        time.sleep(0.001)
    return

class myThread (threading.Thread):
   def __init__(self, threadID, name, env):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.env = env
   def run(self):
       if self.threadID == 1:
           v1_control(self.env)
       elif self.threadID == 2:
           v2_control(self.env)
       else:
           v2_opt(self.env)


def main():
    # env = carEnv()
    # os.system('python manual_control_custom.py -z 1')
    # time.sleep(2)
    # env.init_Controller()
    # carEnv use with: controller_v6, solver_v5
    env = carEnv()
    time.sleep(2)
    env.init_Controller()
    # p2 = multiprocessing.Process(target=v2_opt,args=[env])
    # p1 = multiprocessing.Process(target=v1_control,args=[env])
    # p2 = threading.Thread(target=v2_opt(env))
    # p1 = threading.Thread(target=v1_control(env))
    #
    thread1 = myThread(1, "Thread-1",env)
    thread2 = myThread(2, "Thread-2",env)
    thread3 = myThread(3, "Thread-3",env)

    # Start new Threads
    thread3.start()
    thread2.start()
    thread1.start()
    # # while True:
    # p2.start()
    # p1.start()
    # p2.join()
    # p1.join()

    # trial = '_predict'
    # try:
    #     # P, (ax1, ax2, ax3) = plt.subplots(3, 1)
    #     plt.figure(figsize=(5, 3))
    #     manager = plt.get_current_fig_manager()
    #     manager.window.wm_geometry("+10+10")
    #     traj_10 = None
    #     while True:
    #         # while not env.c2.arrived_goal:
    #         #     if env.c2.data_tick == 0:
    #         #         env.c2.vw_step('constant', None, None)
    #         #     else:
    #         #         traj_11 = env.c1.X_hist[:2, max(env.c1.data_tick-1, 0)]
    #         #         traj_11 = np.array([traj_11[1], traj_11[0]])
    #         #         traj_2 = env.c2.X_hist[:, max(env.c2.data_tick-1, 0)]
    #         #         traj_2 = np.array([traj_2[1], traj_2[0], traj_2[2] / 180 * np.pi])
    #         #         env.c2.vw_step(traj_11, traj_10, traj_2)
    #         #         traj_10 = traj_11
    #         #     env.c1.simple_step(slow=False)
    #             # ax1.plot(env.c2.data_tick, env.c2.X_hist[0, env.c2.data_tick - 1], 'ro', markersize=3)
    #             # ax2.plot(env.c2.data_tick, env.c2.X_hist[1, env.c2.data_tick - 1], 'go', markersize=3)
    #             # ax3.plot(env.c2.data_tick, env.c2.X_hist[2, env.c2.data_tick - 1], 'bo', markersize=3)
    #             '''
    #             plt.plot(env.c2.X_hist[1, env.c2.data_tick - 1],
    #                      env.c2.X_hist[0, env.c2.data_tick - 1], 'ro', markersize=3)
    #             plt.plot(env.c1.X_hist[1, env.c1.data_tick - 1],
    #                      env.c1.X_hist[0, env.c1.data_tick - 1], 'bo', markersize=3)
    #             plt.plot(env.c2.traj_12[0], env.c2.traj_12[1], 'go', markersize=3)
    #             plt.plot(env.c2.sn[0], env.c2.sn[1], 'yo', markersize=3)
    #             '''
    #             p2 = multiprocessing.Process(target=v2_opt,args=[env])
    #             p1 = multiprocessing.Process(target=v1_control,args=[env])
    #             p2.start()
    #             p1.start()
    #             p2.join()
    #             p1.join()
    #             print("DONE")
    #             plt.plot(env.c2.data_tick, env.c2.C_hist[env.c2.data_tick - 1], 'ro', markersize=3)
    #             plt.pause(0.01)
    #             for actor in env.actor_list:
    #                 actor.destroy()
    #             break
    #         # env.reset()
    #     plt.show()
    #     '''
    #     traj = env.c2.get_hist()
    #     traj1 = env.c1.get_hist()
    #     # print(traj)
    #     # print(traj1)
    #     # np.savetxt('vehicle2_traj.txt',traj,delimiter=',')
    #     P, (ax1, ax2, ax3) = plt.subplots(3, 1)
    #     ax1.plot(traj[0,:].T,'r');ax1.set_title('x')
    #     ax2.plot(traj[1,:].T,'g');ax2.set_title('y')
    #     ax3.plot(traj[2,:].T,'b');ax3.set_title(r"$\theta$")
    #     plt.ticklabel_format(useOffset=False)
    #     # plt.savefig('vehicle2'+trial)
    #     P, (ax1, ax2, ax3) = plt.subplots(3, 1)
    #     ax1.plot(traj1[0, :].T, 'r')
    #     ax1.set_title('x')
    #     ax2.plot(traj1[1, :].T, 'g')
    #     ax2.set_title('y')
    #     ax3.plot(traj1[2, :].T, 'b')
    #     ax3.set_title(r"$\theta$")
    #     plt.ticklabel_format(useOffset=False)
    #     # plt.savefig('vehicle1'+trial)
    #     plt.show()
    #     '''
    # except KeyboardInterrupt:
    #     for actor in env.actor_list:
    #         actor.destroy()
    #     exit()

if __name__ == '__main__':
    main()
