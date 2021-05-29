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
    time.sleep(2);
    env.init_Controller()
    # env.c2.
    # env.c1.apply(0.5,0.5)

    try:
        while True:
            # while not env.c2.arrived_goal:
            while True:
                env.c2.step()
            for actor in env.actor_list:
                actor.destroy()
            break;
            # env.reset()
        # traj = env.c2.get_hist()
        # np.savetxt('vehicle2_traj.txt',traj,delimiter=',')
        # P, (ax1, ax2, ax3) = plt.subplots(3, 1)
        # ax1.plot(traj[0,:].T,'r');ax1.set_title('x')
        # ax2.plot(traj[1,:].T,'g');ax2.set_title('y')
        # ax3.plot(traj[2,:].T,'b');ax3.set_title(r"$\theta$")
        # plt.ticklabel_format(useOffset=False)
        # plt.show()
    except KeyboardInterrupt:
        for actor in env.actor_list:
            actor.destroy()
        exit()

if __name__ == '__main__':
    main()
