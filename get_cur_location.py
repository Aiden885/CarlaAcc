import carla

client = carla.Client('localhost', 2000)
carla_world = client.get_world()

spectator = carla_world.get_spectator()

transform = spectator.get_transform()
print(transform)

#Transform(Location(x=-1239.380249, y=3104.088135, z=351.407501), Rotation(pitch=0.112739, yaw=0.886165, roll=0.000120))

#直道起始点