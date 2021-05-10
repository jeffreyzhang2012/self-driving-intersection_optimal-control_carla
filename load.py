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

def load_map(client, map_name='Town04_Opt'):
    world = client.load_world(map_name)
    world.load_map_layer(carla.MapLayer.Buildings)
    # print(client.get_available_maps())
    # while True:
        # print(world.get_spectator().get_transform())
    camera_transform = carla.Transform(carla.Location(x=235.118439, y=-170.812881, z=11.980475),carla.Rotation(pitch=-25.883698, yaw=-1.275940, roll=0.000000))
    world.get_spectator().set_transform(camera_transform)
    return world

def load_vehicle(world):
    blueprint_library = world.get_blueprint_library()
    # print(blueprint_library.filter('vehicle'))
    bp1 = blueprint_library.find('vehicle.audi.etron')
    bp2 = blueprint_library.find('vehicle.mercedes-benz.coupe')
    transform1 = carla.Transform(carla.Location(x=235.0, y=-169.0, z=2.0),carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
    transform2 = carla.Transform(carla.Location(x=255.0, y=-187.0, z=2.0),carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))
    # while True:
    #     print(world.get_spectator().get_transform())
    v1 = world.spawn_actor(bp1, transform1)
    v2 = world.spawn_actor(bp2, transform2)
    return v1, v2
# if __name__ == '__main__':
#
#     main()
