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


def main():
    env = carEnv()
    # os.system('python manual_control_custom.py -z 1')
    time.sleep(2)
    env.init_Controller()
    # carEnv use with: controller_v6, solver_v5
    trial = '_predict'
    try:
        # P, (ax1, ax2, ax3) = plt.subplots(3, 1)
        plt.figure(figsize=(5, 3))
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+10+10")
        traj_10 = None
        while True:
            while not env.c2.arrived_goal:
                # '''
                if env.c2.data_tick == 0:
                    env.c2.vw_step('constant', None, None)
                else:
                    traj_11 = env.c1.X_hist[:2, max(env.c1.data_tick-1, 0)]
                    traj_11 = np.array([traj_11[1], traj_11[0]])
                    traj_2 = env.c2.X_hist[:, max(env.c2.data_tick-1, 0)]
                    traj_2 = np.array([traj_2[1], traj_2[0], traj_2[2] / 180 * np.pi])
                    env.c2.vw_step(traj_11, traj_10, traj_2)
                    traj_10 = traj_11
                env.c1.simple_step(slow=False)
                # ax1.plot(env.c2.data_tick, env.c2.X_hist[0, env.c2.data_tick - 1], 'ro', markersize=3)
                # ax2.plot(env.c2.data_tick, env.c2.X_hist[1, env.c2.data_tick - 1], 'go', markersize=3)
                # ax3.plot(env.c2.data_tick, env.c2.X_hist[2, env.c2.data_tick - 1], 'bo', markersize=3)
                '''
                plt.plot(env.c2.X_hist[1, env.c2.data_tick - 1],
                         env.c2.X_hist[0, env.c2.data_tick - 1], 'ro', markersize=3)
                plt.plot(env.c1.X_hist[1, env.c1.data_tick - 1],
                         env.c1.X_hist[0, env.c1.data_tick - 1], 'bo', markersize=3)
                plt.plot(env.c2.traj_12[0], env.c2.traj_12[1], 'go', markersize=3)
                plt.plot(env.c2.sn[0], env.c2.sn[1], 'yo', markersize=3)
                '''
                plt.plot(env.c2.data_tick, env.c2.C_hist[env.c2.data_tick - 1], 'ro', markersize=3)
                plt.pause(0.0001)
            for actor in env.actor_list:
                actor.destroy()
            break
            # env.reset()
        plt.show()
        '''
        traj = env.c2.get_hist()
        traj1 = env.c1.get_hist()
        # print(traj)
        # print(traj1)
        # np.savetxt('vehicle2_traj.txt',traj,delimiter=',')
        P, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(traj[0,:].T,'r');ax1.set_title('x')
        ax2.plot(traj[1,:].T,'g');ax2.set_title('y')
        ax3.plot(traj[2,:].T,'b');ax3.set_title(r"$\theta$")
        plt.ticklabel_format(useOffset=False)
        # plt.savefig('vehicle2'+trial)
        P, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(traj1[0, :].T, 'r')
        ax1.set_title('x')
        ax2.plot(traj1[1, :].T, 'g')
        ax2.set_title('y')
        ax3.plot(traj1[2, :].T, 'b')
        ax3.set_title(r"$\theta$")
        plt.ticklabel_format(useOffset=False)
        # plt.savefig('vehicle1'+trial)
        plt.show()
        '''
    except KeyboardInterrupt:
        for actor in env.actor_list:
            actor.destroy()
        exit()

if __name__ == '__main__':
    main()
