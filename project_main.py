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

def main():
    env = carEnv();
    # os.system('python manual_control_custom.py -z 1')
    try:
        while True:
            while not env.c2.arrived_goal:
                env.c2.step()
            for actor in env.actor_list:
                actor.destroy()
            env.reset()
    except KeyboardInterrupt:
        for actor in env.actor_list:
            actor.destroy()
        exit()

if __name__ == '__main__':
    main()
