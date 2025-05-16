import carla
import math
import numpy as np
import cv2
import lane_detection
import kalman_filter
import radar_cluster
import pygame
import threading

class acc:
    def __init__(self):
        self.tracker = kalman_filter.RadarTracker()
        self.lane_detector = lane_detection.LaneDetector()
        self.radar_point_cluster = radar_cluster.RadarClusterNode()
        self.max_follow_distance = 50

        self.radar_detections = []
        self.latest_camera_image = None
        # self.latest_lidar_points = []
        self.radar_2_world = []
        self.world_2_camera = []
        self.cluster = []
        self.track_id = []

        self.init_carla()

        # self.radar.listen(self.radar_callback)
        # self.camera.listen(self.camera_callback)
        # self.lidar.listen(self.lidar_callback)

    def init_carla(self):
        # Connect to the CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        # Load layered map for Town 01 with minimum layout plus buildings and parked vehicles
        self.world = self.client.load_world('Town05', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

        # Toggle all buildings off
        self.world.unload_map_layer(carla.MapLayer.Buildings)

        # Toggle all buildings on   
        self.world.load_map_layer(carla.MapLayer.Buildings)

        # Get all actors in the world
        actors = self.world.get_actors()

        # Filter for traffic lights
        traffic_lights = [actor for actor in actors if actor.type_id.startswith('traffic.traffic_light')]

        # Set all traffic lights to green
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.set_green_time(1000.0)  # Set green time to a large value (in seconds)
            traffic_light.set_red_time(0.0)  # Set red time to 0
            traffic_light.set_yellow_time(0.0)  # Set yellow time to 0

        # Get the blueprint library and map
        blueprint_library = self.world.get_blueprint_library()
        map = self.world.get_map()

        self.manual_mode = False
        self.manual_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        # Initialize Pygame for manual control
        if self.manual_mode:
            pygame.init()
            pygame.display.set_mode((200, 200))  # Small window for Pygame events
            pygame.key.set_repeat(50, 50)  # Enable key repeat for smooth control

        # Select vehicle blueprints
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]

        # Get all valid spawn points on the road
        spawn_points = map.get_spawn_points()[0]

        #uphill town05
        spawn_points.location.x = 80.663731
        spawn_points.location.y = -203.651886
        spawn_points.location.z = 0
        spawn_points.rotation.pitch = 21.448849
        spawn_points.rotation.yaw = 179.069489
        spawn_points.rotation.roll = 0

        # #downhill town05
        # spawn_points.location.x = -143.173355
        # spawn_points.location.y = 205.082870
        # spawn_points.location.z = 0
        # spawn_points.rotation.pitch = -2.972689
        # spawn_points.rotation.yaw = -6.116826
        # spawn_points.rotation.roll = 0

        waypoint = map.get_waypoint(spawn_points.location)
        lanes  = []
        current_waypoint = waypoint
        # for _ in range(3):
        #     lanes.append(current_waypoint)
        #     current_waypoint = current_waypoint.get_left_lane()
        #     if current_waypoint is None:
        #         break
        lanes.append(current_waypoint.get_left_lane())
        lanes.append(current_waypoint)
        lanes.append(current_waypoint.get_right_lane())

        self.vehicles = []
        locations = []
        for lane in lanes:
            if lane is not None:
                transform = lane.transform
                transform.location.z += 0.1
                locations.append(transform.location)
                vehicle = self.world.spawn_actor(vehicle_bp, transform)
                self.vehicles.append(vehicle)

        # Spawn the ego vehicle
        ego_spawn_point = spawn_points
        # ego_spawn_point.location = vehicles[1].location
        ego_spawn_point.location = locations[1]
        # self.target_vehicle = self.vehicles[1]
        ego_spawn_point.location.x += 15
        ego_spawn_point.location.z += 0.5
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, ego_spawn_point)


        # Enable autopilot for all vehicles
        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_synchronous_mode(True)
        tm_port = tm.get_port()
        
        for vehicle in self.vehicles:
            vehicle.set_autopilot(True, tm_port)
            tm.auto_lane_change(vehicle, False)
        if not self.manual_mode:
            self.ego_vehicle.set_autopilot(True, tm_port)
            tm.auto_lane_change(self.ego_vehicle, False)


        # Set up radar sensor on the ego vehicle
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('range', '100.0')
        radar_bp.set_attribute('horizontal_fov', '120.0')
        radar_bp.set_attribute('vertical_fov', '30.0')
        radar_bp.set_attribute('points_per_second', '10000')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)

        # Set up camera sensor on the ego vehicle
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)

        # Set up LIDAR sensor on the ego vehicle
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('points_per_second', '1000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-10')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)

    def get_ego_speed(self, vehicle):
        velocity = vehicle.get_velocity()  # carla.Vector3D
        speed_m_s = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_m_s * 3.6  # Convert m/s to km/h
        return speed_m_s

        # Radar callback
    def radar_callback(self, radar_data):
        # print(f"Radar callback triggered, frame: {radar_data.frame}, detections: {len(radar_data)}")
        self.radar_points = []
        self.filted_points = []
        ego_velocity = self.get_ego_speed(self.ego_vehicle)
        velocity_tolerance = 1.0  # m/s tolerance for static points
        
        for detection in radar_data:
            try:
                distance = detection.depth
                azimuth = math.degrees(detection.azimuth)
                altitude = math.degrees(detection.altitude)
                velocity = detection.velocity
                x = distance * math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth))
                y = -distance * math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth))
                z = distance * math.sin(math.radians(altitude))
                vx = velocity * math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth))
                vy = velocity * math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth))
                vz = velocity * math.sin(math.radians(altitude))
                expected_static_velocity = -ego_velocity * math.cos(math.radians(azimuth)) * math.cos(math.radians(altitude))
                if z > -0.5:
                    self.radar_points.append([x, y, z, vx, vy, vz, velocity])
                    if abs(velocity - expected_static_velocity) > velocity_tolerance:
                        self.filted_points.append([x, y, z, vx, vy, vz, velocity])
            except AttributeError as e:
                print(f"AttributeError: {e}. Raw detection: {detection}")
                break

        if self.filted_points:
            # if cluster_points:
            self.cluster = self.radar_point_cluster.radar_cluster(self.filted_points)
            if self.cluster:
                # track_id = tracker.update(np.concatenate((np.array(cluster)[:, 0:2], np.array(cluster)[:, 6:8]), axis=1))
                self.track_id = self.tracker.update(self.cluster)
            else:
                self.track_id = []

    # Camera callback
    def camera_callback(self, image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]
            self.latest_camera_image = array

    # LIDAR callback
    def lidar_callback(self, lidar_data):
        points = []
        for point in lidar_data:
            x = point.point.x
            y = point.point.y
            z = point.point.z
            intensity = point.intensity
            points.append([x, y, z, intensity])
        self.latest_lidar_points = points

    # Function to get extrinsic parameters from radar to camera
    def get_extrinsic_params(self, radar_sensor, camera_sensor):
        self.radar_2_world = radar_sensor.get_transform().get_matrix()
        self.world_2_camera = np.array(camera_sensor.get_transform().get_inverse_matrix())

    # Function to project radar points to camera image
    def project_radar_to_camera(self, radar_points, image_width=1280, image_height=720, fov=90):
        fx = image_width / (2.0 * np.tan(fov * np.pi / 360.0))
        fy = image_height / (2.0 * np.tan(fov * np.pi / 360.0))
        cx = image_width / 2
        cy = image_height / 2
        # K = np.identity(3)
        # K[0, 0] = K[1, 1] = f
        # K[0, 2] = image_width / 2.0
        # K[1, 2] = image_height / 2.0
        projected_points = []

        for x, y, z, w, l, h, vx, vy, vz, id in radar_points:
            radar_point = np.array([x, y, z, 1])
            world_point = np.dot(self.radar_2_world, radar_point)
            camera_point = np.dot(self.world_2_camera, world_point)
            point_in_camera_coords = np.array([
                    camera_point[1],
                    camera_point[2] * -1,
                    camera_point[0]])
            # if point_in_camera_coords[2] > 0:
            u = cx + (fx * point_in_camera_coords[0] / point_in_camera_coords[2])
            v = cy + (fy * point_in_camera_coords[1] / point_in_camera_coords[2])
            # u = int((cx + fx * point_in_camera_coords[0]) / point_in_camera_coords[2])
            # v = int((cy + fy * point_in_camera_coords[1]) / point_in_camera_coords[2])
            # print(u, "  ", v)
            
            # if 0 <= u < image_width and 300 <= v < image_height:
            ipm_point = np.dot(self.lane_detector.M, np.array([u, v - 300, 1]))
            ipm_point[0] = ipm_point[0] / ipm_point[2]
            ipm_point[1] = ipm_point[1] / ipm_point[2]
            # ipm_point = np.array([u, v - 300])
            # print(u, " ", v, " -> ", ipm_point[0], "   ", ipm_point[1])
            projected_points.append([int(u), int(v), int(ipm_point[0]), int(ipm_point[1])])
        
        return projected_points#, track_id
    
    # Function to compute IPM transformation matrix
    def get_ipm_transform_matrix(self, camera_sensor, K, image_width=800, image_height=600):
        camera_height = camera_sensor.get_transform().location.z
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        src_points = np.float32([
            [image_width * 0.2, image_height],
            [image_width * 0.8, image_height],
            [image_width * 0.6, image_height * 0.4],
            [image_width * 0.4, image_height * 0.4]
        ])
        ground_width = 10.0
        ground_length = 20.0
        dst_points = np.float32([
            [-ground_width / 2, 0],
            [ground_width / 2, 0],
            [ground_width / 2, ground_length],
            [-ground_width / 2, ground_length]
        ])
        H, _ = cv2.findHomography(src_points, dst_points)
        print("Camera IPM Transformation Matrix (Homography H):")
        print(H)
        return H

    # Function to handle manual control
    def handle_manual_control(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.manual_mode = not self.manual_mode
                    print(f"Switched to {'manual' if self.manual_mode else 'autonomous'} mode")
                    if self.manual_mode:
                        self.manual_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        
        if self.manual_mode:
            keys = pygame.key.get_pressed()
            throttle_increment = 0.1
            steer_increment = 0.1
            
            # Throttle control
            if keys[pygame.K_w]:
                self.manual_control.throttle = min(self.manual_control.throttle + throttle_increment, 1.0)
            if keys[pygame.K_s]:
                self.manual_control.throttle = max(self.manual_control.throttle - throttle_increment, 0.0)
            
            # Steering control
            if keys[pygame.K_a]:
                self.manual_control.steer = max(self.manual_control.steer - steer_increment, -1.0)
            if keys[pygame.K_d]:
                self.manual_control.steer = min(self.manual_control.steer + steer_increment, 1.0)
            
            # Brake control
            if keys[pygame.K_SPACE]:
                self.manual_control.brake = 1.0
                self.manual_control.throttle = 0.0
            else:
                self.manual_control.brake = 0.0
            
            # Reset steer if no input
            if not (keys[pygame.K_a] or keys[pygame.K_d]):
                self.manual_control.steer *= 0.9  # Dampen steering
            
            self.ego_vehicle.apply_control(self.manual_control)
            print(f"Manual: Throttle={self.manual_control.throttle:.2f}, Steer={self.manual_control.steer:.2f}, Brake={self.manual_control.brake:.2f}")
        
    def generate_target(self):
        try:
            print("Starting radar, camera, LIDAR, and object detection ROS publishing (press Ctrl+C to stop)...")
            self.get_extrinsic_params(self.radar, self.camera)
            # ipm_matrix = get_ipm_transform_matrix(camera)
            # lidar_to_camera_transform = get_lidar_to_camera_transform(lidar, camera)


            while True:
                self.world.tick()

                if self.manual_mode:
                    self.handle_manual_control()
                # select target
                if self.latest_camera_image is not None:
                    image_with_radar = self.latest_camera_image.copy()
                    lane_windows, lane_image, detected_windows = self.lane_detector.lane_detect(image_with_radar)
                    track_id = self.track_id.copy()
                    if track_id is not None:
                        projected_points = self.project_radar_to_camera(track_id)
                        current_target_idx = -1
                        # print(projected_points)

                        for idx in range(len(track_id)):
                            u, v, ipm_u, ipm_v = projected_points[idx]
                            color = (255, 0, 0)# if self.track_id[idx][0] < 10 else (0, 255, 0)
                            cv2.circle(image_with_radar, (u, v), 5, color, -1)
                            # if (idx < len(track_id)):
                            
                            if ipm_v < 0:
                                ipm_v = 0


                            curve_left = 0
                            curve_right = 0
                            curve_left_dis = 0
                            curve_right_dis = 0
                            if detected_windows is not None:
                                for i in range(1, len(detected_windows)):
                                    if detected_windows[i][0][1] == detected_windows[i - 1][0][1]:
                                        continue
                                    if detected_windows[i][0][1] < detected_windows[i - 1][0][1]:
                                        curve_right += 1
                                    else:
                                        curve_left += 1
                                if curve_right - curve_left > len(detected_windows)//2:
                                    curve_right_dis = 0.5
                                    curve_left_dis = 0.2
                                elif curve_right - curve_left < len(detected_windows)//2:
                                    curve_left_dis = -0.5
                                    curve_right_dis = -0.2
                            # # emergency stop
                            if ((-1.5 + curve_left_dis) < track_id[idx][1] < (1.5 + curve_right_dis)):
                                if ((current_target_idx != -1) and (track_id[idx][0] < track_id[current_target_idx][0])) or (current_target_idx == -1):
                                    
                                    current_target_idx = idx

                        if current_target_idx >= 0:
                            cv2.circle(image_with_radar, (projected_points[current_target_idx][0], projected_points[current_target_idx][1]), 10, (255, 255, 255), -1)
                            cv2.circle(lane_image, (projected_points[current_target_idx][2], projected_points[current_target_idx][3]), 10, (255, 255, 0), -1)
                            cv2.putText(image_with_radar, "id=" + str(track_id[current_target_idx][-1]), (projected_points[current_target_idx][0] + 5, projected_points[current_target_idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,225,100), 2)
                            print("current target info (x, y, z, w, l, h, vx, vy, vz, track_id): ", track_id[current_target_idx])

                    if lane_image is not None:
                        cv2.imshow("lane", lane_image)
                    cv2.imshow("Radar and Objects on Camera", image_with_radar)
                    cv2.waitKey(1)

                
                # time.sleep(0.005)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cv2.destroyAllWindows()

    def destroy(self):
        # Cleanup
        self.radar.stop()
        self.camera.stop()
        # self.lidar.stop()
        self.radar.destroy()
        self.camera.destroy()
        # self.lidar.destroy()
        for vehicle in self.vehicles:
            vehicle.set_autopilot(False, self.tm_port)
            vehicle.destroy()
        print(f"Destroyed {len(self.vehicles)} vehicles, radar, camera, and LIDAR.")


def main():
    acc_actor = acc()
    
    thread_1 = threading.Thread(target=acc_actor.radar.listen(acc_actor.radar_callback), name='T1')
    thread_2 = threading.Thread(target=acc_actor.camera.listen(acc_actor.camera_callback), name='T2')
    thread_3 = threading.Thread(target=acc_actor.lidar.listen(acc_actor.lidar_callback), name='T3')
    # thread_4 = threading.Thread(target=acc_actor.generate_target(), name='T4')

    thread_1.start()  # 开启T1
    thread_2.start()  # 开启T2
    thread_3.start()
    # thread_4.start()
    # # 不加 join 打印信息的前后顺序，取决去线程处理数据的速度
    # # 在写 join 的时候可以把处理数据量少的写在前面，主要目的是减少主线程或其他依赖线程的等待时间
    # thread_2.join()
    # thread_1.join()
    # # thread_3.join()
    # thread_4.join()
    acc_actor.generate_target()

if __name__ == '__main__':
    main()