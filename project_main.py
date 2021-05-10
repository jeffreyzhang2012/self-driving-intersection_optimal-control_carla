import glob
import os
import sys

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
from load import *

def main():
    client = carla.Client('localhost', 2000)
    world = load_map(client)
    load_vehicle(world)

if __name__ == '__main__':
    main()
