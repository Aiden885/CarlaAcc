import carla


class SensorsManager:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.sensors = []

    def setup_radar(self, callback):
        blueprint_library = self.world.get_blueprint_library()
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('range', '100.0')
        radar_bp.set_attribute('horizontal_fov', '120.0')
        radar_bp.set_attribute('vertical_fov', '30.0')
        radar_bp.set_attribute('points_per_second', '10000')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)
        self.sensors.append(radar)
        return radar

    def setup_camera(self, callback):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        self.sensors.append(camera)
        return camera

    def setup_lidar(self, callback):
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('points_per_second', '1000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-10')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)
        self.sensors.append(lidar)
        return lidar

    def destroy_sensors(self):
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors = []