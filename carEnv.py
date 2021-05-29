import glob
import os
import sys
import random

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
from controller import *

class carEnv:
    SHOW_CAM = 1
    STEER_AMT = 1.0

    im_width = 480
    im_height = 360
    actor_list = []

    front_camera = None
    collision_hist = []
    def __init__(self):
        # init world and view
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.load_world('Town04_Opt')
        self.world.load_map_layer(carla.MapLayer.Buildings)
        self.blueprint_library = self.world.get_blueprint_library()
        camera_transform = carla.Transform(carla.Location(x=235.118439, y=-170.812881, z=11.980475),carla.Rotation(pitch=-25.883698, yaw=-1.275940, roll=0.000000))
        self.world.get_spectator().set_transform(camera_transform)
        # init vehicles
        self.bp_v1 = self.blueprint_library.find('vehicle.audi.etron')
        self.bp_v2 = self.blueprint_library.find('vehicle.mercedes-benz.coupe')
        self.v1_init_loc = carla.Transform(carla.Location(x=235.0, y=-169.0, z=2.0),carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        self.v2_init_loc = carla.Transform(carla.Location(x=255.0, y=-187.0, z=2.0),carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))
        self.reset()
        # while True:
        #     print(world.get_spectator().get_transform())

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.v1 = self.world.spawn_actor(self.bp_v1, self.v1_init_loc)
        self.v2 = self.world.spawn_actor(self.bp_v2, self.v2_init_loc)
        self.actor_list.append(self.v1)
        self.actor_list.append(self.v2)

    def init_Controller(self):
        self.c1 = V1Controller(self.v1)
        # self.c2 = CurveFollowController(self.v2)
        self.c2 = vwController(self.v2)
