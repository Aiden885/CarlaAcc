import carla
import random
import time
import math
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Twist
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
import cv2
import lane_detection
import radar_cluster
import pygame
import threading

class acc:
    def __init__(self):
        self.lane_detector = lane_detection.LaneDetector()
        # self.radar_point_cluster = radar_cluster.RadarClusterNode()
        self.max_follow_distance = 50

        self.radar_points = []
        self.latest_camera_image = None
        self.latest_lidar_points = []
        self.radar_2_world = []
        self.world_2_camera = []
        self.radar_points = []

        self.init_ros()
        self.init_carla()

        # self.radar.listen(self.radar_callback)
        # self.camera.listen(self.camera_callback)
        # self.lidar.listen(self.lidar_callback)
        
        
        # self.lidar.listen(self.lidar_callback)


    def init_ros(self):
        # Initialize ROS node
        rospy.init_node('carla_ego_publisher', anonymous=True)
        self.image_pub = rospy.Publisher('/carla/ego_vehicle/camera', Image, queue_size=10)
        self.lane_image_pub = rospy.Publisher('/carla/ego_vehicle/lane_image', Image, queue_size=10)
        self.radar_pub = rospy.Publisher('/carla/ego_vehicle/radar', PointCloud2, queue_size=10)
        # self.radar_cluster_pub = rospy.Publisher('/carla/ego_vehicle/radar_cluster', PointCloud2, queue_size=10)
        self.lidar_pub = rospy.Publisher('/carla/ego_vehicle/lidar', PointCloud2, queue_size=10)
        # self.objects_pub = rospy.Publisher('/carla/ego_vehicle/detected_objects', PointCloud2, queue_size=10)  # Simplified as PointCloud2
        self.clock_pub = rospy.Publisher('/clock', Clock, queue_size=10)
        self.bridge = CvBridge()

    def init_carla(self):
        # Connect to the CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        # Load layered map for Town 01 with minimum layout plus buildings and parked vehicles
        self.world = self.client.load_world('Town04', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

        # Toggle all buildings off
        self.world.unload_map_layer(carla.MapLayer.Buildings)

        # Toggle all buildings on   
        self.world.load_map_layer(carla.MapLayer.Buildings)

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
        spawn_points = map.get_spawn_points()
        print(f"Available spawn points: {len(spawn_points)}")

        # Limit to 10 regular vehicles + 1 ego vehicle
        num_regular_vehicles = min(1, len(spawn_points) - 1)
        vehicles = []

        # Spawn 10 regular vehicles
        for i in range(num_regular_vehicles):
            spawn_point = spawn_points[i]
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicles.append(vehicle)
                print(f"Regular Vehicle {i+1} spawned at: {spawn_point.location}")
            except RuntimeError as e:
                print(f"Failed to spawn regular vehicle {i+1}: {e}")
                break

        # Spawn the ego vehicle
        ego_spawn_point = spawn_points[0]
        self.target_vehicle = vehicles[0]
        ego_spawn_point.location.x -= 10
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, ego_spawn_point)
        vehicles.append(self.ego_vehicle)
        # print(f"Ego Vehicle spawned at: {ego_spawn_point.location}")

        # Enable autopilot for all vehicles
        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_synchronous_mode(True)
        tm_port = tm.get_port()
        
        for vehicle in vehicles[:-1]:
            vehicle.set_autopilot(True, tm_port)
            tm.auto_lane_change(vehicle, False)
        if not self.manual_mode:
            vehicles[-1].set_autopilot(True, tm_port)
            tm.auto_lane_change(vehicles[-1], False)


        # Set up radar sensor on the ego vehicle
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('range', '100.0')
        radar_bp.set_attribute('horizontal_fov', '120.0')
        radar_bp.set_attribute('vertical_fov', '30.0')
        radar_bp.set_attribute('points_per_second', '1500')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)

        # Set up camera sensor on the ego vehicle
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)

        # Set up LIDAR sensor on the ego vehicle
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-10')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)


        # Radar callback
    def radar_callback(self, radar_data):
        # print(f"Radar callback triggered, frame: {radar_data.frame}, detections: {len(radar_data)}")
        for detection in radar_data:
            try:
                distance = detection.depth
                azimuth = math.degrees(detection.azimuth)
                altitude = math.degrees(detection.altitude)
                velocity = detection.velocity
                x = distance * math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth))
                y = distance * math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth))
                z = distance * math.sin(math.radians(altitude))
                vx = velocity * math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth))
                vy = velocity * math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth))
                vz = velocity * math.sin(math.radians(altitude))
                if z > -0.5 and x < 50 and -5 < y < 5:
                    self.radar_points.append([x, y, z, velocity])
            except AttributeError as e:
                print(f"AttributeError: {e}. Raw detection: {detection}")
                break

        if self.radar_points:
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('velocity', 12, PointField.FLOAT32, 1)]
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "ego_vehicle_radar"
            pc2_msg = pc2.create_cloud(header, fields, self.radar_points)
            self.radar_pub.publish(pc2_msg)
            # print(f"Published to /carla/ego_vehicle/radar, points: {len(points)}")


    # Camera callback
    def camera_callback(self, image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]
            self.latest_camera_image = array
            ros_image = self.bridge.cv2_to_imgmsg(array, encoding="rgb8")
            ros_image.header.stamp = rospy.Time.now()
            ros_image.header.frame_id = "ego_vehicle_camera"
            self.image_pub.publish(ros_image)
            # print(f"Published to /carla/ego_vehicle/camera, frame: {image.frame}")
        except CvBridgeError as e:
            print(f"CvBridgeError: {e}")

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
        if points:
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('intensity', 12, PointField.FLOAT32, 1)]
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "ego_vehicle_lidar"
            pc2_msg = pc2.create_cloud(header, fields, points)
            self.lidar_pub.publish(pc2_msg)
            # print(f"Published to /carla/ego_vehicle/lidar, points: {len(points)}")

    # Function to get extrinsic parameters from radar to camera
    def get_extrinsic_params(self, radar_sensor, camera_sensor):
        self.radar_2_world = radar_sensor.get_transform().get_matrix()
        self.world_2_camera = np.array(camera_sensor.get_transform().get_inverse_matrix())
        print(self.radar_2_world)
        print("---------------------")
        print(self.world_2_camera)

    # Function to project radar points to camera image
    def project_radar_to_camera(self, radar_points, image_width=800, image_height=600, fov=90):
        fx = image_width / (2.0 * np.tan(fov * np.pi / 360.0))
        fy = image_height / (2.0 * np.tan(fov * np.pi / 360.0))
        cx = image_width / 2
        cy = image_height / 2
        # K = np.identity(3)
        # K[0, 0] = K[1, 1] = f
        # K[0, 2] = image_width / 2.0
        # K[1, 2] = image_height / 2.0
        projected_points = []
        if radar_points is None:
            return 
        
        for x, y, z, velocity in radar_points:
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
            
            if 0 <= u < image_width and 300 <= v < image_height:
                ipm_point = np.dot(self.lane_detector.M, np.array([u, v - 300, 1]))
                ipm_point[0] = ipm_point[0] / ipm_point[2]
                ipm_point[1] = ipm_point[1] / ipm_point[2]
                # ipm_point = np.array([u, v - 300])
                # print(u, " ", v, " -> ", ipm_point[0], "   ", ipm_point[1])
                projected_points.append([int(u), int(v), x, y, z, velocity, int(ipm_point[0]), int(ipm_point[1])])
        
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
    
    # Function to estimate distance to target vehicle using LIDAR
    def estimate_target_distance(self, ego_vehicle, target_vehicle, lidar_data):
        if not target_vehicle or not lidar_data:
            return None
        
        target_location = target_vehicle.get_location()
        ego_location = ego_vehicle.get_location()
        min_dist = float('inf')
        closest_point = None
        
        for x, y, z, _ in lidar_data:
            # Transform LIDAR point to world coordinates
            lidar_point = carla.Location(x, y, z)
            lidar_transform = self.lidar.get_transform()
            world_point = lidar_transform.transform(lidar_point)
            dist_to_target = math.sqrt(
                (world_point.x - target_location.x)**2 +
                (world_point.y - target_location.y)**2 +
                (world_point.z - target_location.z)**2
            )
            if dist_to_target < min_dist and dist_to_target < 2.0:  # Assume target vehicle is ~2m wide/long
                min_dist = dist_to_target
                closest_point = world_point
        
        if closest_point:
            dist = math.sqrt(
                (closest_point.x - ego_location.x)**2 +
                (closest_point.y - ego_location.y)**2 +
                (closest_point.z - ego_location.z)**2
            )
            return dist
        return None
    
    # Function to control ego vehicle to follow target
    def follow_target_vehicle(self, ego_vehicle, target_distance, desired_distance=10.0, kp=0.1):
        if target_distance is None:
            control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            ego_vehicle.apply_control(control)
            return
        
        # Proportional control
        error = target_distance - desired_distance
        throttle = kp * error
        throttle = max(0.0, min(1.0, throttle))  # Clamp throttle
        brake = 0.0
        if throttle < 0.1 and error < 0:  # Apply brake if too close
            throttle = 0.0
            brake = 0.5
        
        # Simple steering (align with target direction)
        ego_location = ego_vehicle.get_location()
        target_location = self.target_vehicle.get_location()
        dx = target_location.x - ego_location.x
        dy = target_location.y - ego_location.y
        target_yaw = math.degrees(math.atan2(dy, dx))
        ego_yaw = ego_vehicle.get_transform().rotation.yaw
        yaw_error = target_yaw - ego_yaw
        if yaw_error > 180:
            yaw_error -= 360
        elif yaw_error < -180:
            yaw_error += 360
        steer = yaw_error / 90.0  # Normalize to [-1, 1]
        steer = max(-1.0, min(1.0, steer))
        
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        ego_vehicle.apply_control(control)
        print(f"Following target: Distance={target_distance:.2f}m, Throttle={throttle:.2f}, Steer={steer:.2f}, Brake={brake:.2f}")

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


            while not rospy.is_shutdown():
                clock_msg = Clock()
                clock_msg.clock = rospy.Time.from_sec(time.time())
                self.clock_pub.publish(clock_msg)
                self.world.tick()

                if self.manual_mode:
                    self.handle_manual_control()

                # Estimate distance to target and control ego vehicle
                # target_distance = self.estimate_target_distance(self.ego_vehicle, self.target_vehicle, self.latest_lidar_points)
                # self.follow_target_vehicle(self.ego_vehicle, target_distance)

                # if latest_camera_image is not None:
                #     cv2.imshow("camera", latest_camera_image)
                #     cv2.waitKey(1)out_img

                # select target
                if self.latest_camera_image is not None and self.radar_points is not None:
                    image_with_radar = self.latest_camera_image.copy()
                    # lane_windows, lane_image = self.lane_detector.lane_detect(image_with_radar)

                    projected_points = self.project_radar_to_camera(self.radar_points)
                    
                    current_target = [-1, 0, 100, 0, 0, 0, 0, 0]
                    # print(projected_points)
                    for idx, point in enumerate(projected_points):
                        u, v, x, y, z, velocity, ipm_u, ipm_v = point
                        color = (255, 0, 0) if x< 10 else (0, 255, 0)
                        # cv2.circle(image_with_radar, (u, v), 5, color, -1)
                        # if 0 <= ipm_u < lane_image.shape[0] and 0 <= ipm_v < lane_image.shape[1]:
                        #     cv2.circle(lane_image, (ipm_u, ipm_v), 5, (255, 0, 0), -1)

                        # #select target in lane windows
                        # for windows in lane_windows:
                        #     if (windows[0][0] < ipm_u < windows[3][0]) and ( windows[0][1] < ipm_v < windows[1][1]) and (x < current_target[2]):
                        #         current_target = point

                        # emergency stop
                        if (-2 < y < 2) and (x < current_target[2]):
                            current_target = point

                    if current_target[0] != -1:
                        cv2.circle(image_with_radar, (current_target[0], current_target[1]), 10, (255, 255, 255), -1)
                        # cv2.circle(lane_image, (current_target[6], current_target[7]), 10, (255, 255, 255), -1)
                        print("current target info (u, v, x, y, z, v, ipm_u, ipm_v): ", current_target)
                    # cv2.imshow("lane", lane_image)
                    # cv2.imshow("Radar and Objects on Camera", image_with_radar)
                    # cv2.waitKey(1)
                    ros_image = self.bridge.cv2_to_imgmsg(image_with_radar, encoding="rgb8")
                    ros_image.header.stamp = rospy.Time.now()
                    ros_image.header.frame_id = "ego_vehicle_camera"
                    self.image_pub.publish(ros_image)

                    # ros_lane_image = self.bridge.cv2_to_imgmsg(lane_image, encoding="rgb8")
                    # ros_lane_image.header.stamp = rospy.Time.now()
                    # ros_lane_image.header.frame_id = "ego_vehicle_camera_lane"
                    # self.lane_image_pub.publish(ros_lane_image)
                    
                    self.radar_points = []
                
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cv2.destroyAllWindows()

    def destroy(self):
        # Cleanup
        self.radar.stop()
        self.camera.stop()
        self.lidar.stop()
        self.radar.destroy()
        self.camera.destroy()
        self.lidar.destroy()
        for vehicle in self.vehicles:
            vehicle.set_autopilot(False, self.tm_port)
            vehicle.destroy()
        print(f"Destroyed {len(self.vehicles)} vehicles, radar, camera, and LIDAR.")


def main():
    acc_actor = acc()
    
    thread_1 = threading.Thread(target=acc_actor.radar.listen(acc_actor.radar_callback), name='T1')
    thread_1.daemon = True
    thread_2 = threading.Thread(target=acc_actor.camera.listen(acc_actor.camera_callback), name='T2')
    thread_2.daemon = True
    thread_4 = threading.Thread(target=acc_actor.lidar.listen(acc_actor.lidar_callback), name='T4')
    thread_4.daemon = True
    # thread_3 = threading.Thread(target=acc_actor.generate_target(), name='T3')
    thread_1.start()  # 开启T1
    thread_2.start()  # 开启T2
    # thread_3.start()
    thread_4.start()
    # 不加 join 打印信息的前后顺序，取决去线程处理数据的速度
    # 在写 join 的时候可以把处理数据量少的写在前面，主要目的是减少主线程或其他依赖线程的等待时间
    # thread_2.detach()
    # thread_1.detach()
    # thread_3.detach()
    # thread_4.detach()

    acc_actor.generate_target()
    

if __name__ == '__main__':
    main()