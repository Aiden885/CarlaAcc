import carla

client = carla.Client('localhost', 2000)
carla_world = client.get_world()

spectator = carla_world.get_spectator()

transform = spectator.get_transform()
print(transform)