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
import kalman_filter
import radar_cluster


# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
tracker = kalman_filter.RadarTracker()
lane_detector = lane_detection.LaneDetector()
radar_point_cluster = radar_cluster.RadarClusterNode()
max_follow_distance = 30

# Confirm CARLA version
# print(f"Using CARLA version: {carla.__version__}")

# Initialize ROS node
rospy.init_node('carla_ego_publisher', anonymous=True)
image_pub = rospy.Publisher('/carla/ego_vehicle/camera', Image, queue_size=10)
lane_image_pub = rospy.Publisher('/carla/ego_vehicle/lane_image', Image, queue_size=10)
radar_pub = rospy.Publisher('/carla/ego_vehicle/radar', PointCloud2, queue_size=10)
radar_cluster_pub = rospy.Publisher('/carla/ego_vehicle/radar_cluster', PointCloud2, queue_size=10)
lidar_pub = rospy.Publisher('/carla/ego_vehicle/lidar', PointCloud2, queue_size=10)
objects_pub = rospy.Publisher('/carla/ego_vehicle/detected_objects', PointCloud2, queue_size=10)  # Simplified as PointCloud2
clock_pub = rospy.Publisher('/clock', Clock, queue_size=10)
bridge = CvBridge()

# Get the blueprint library and map
blueprint_library = world.get_blueprint_library()
map = world.get_map()

# Select vehicle blueprints
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
ego_vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]

# Get all valid spawn points on the road
spawn_points = map.get_spawn_points()
print(f"Available spawn points: {len(spawn_points)}")

print(spawn_points[0])

# Limit to 10 regular vehicles + 1 ego vehicle
num_regular_vehicles = min(10, len(spawn_points) - 1)
vehicles = []

# Spawn 10 regular vehicles
for i in range(num_regular_vehicles):
    spawn_point = spawn_points[i]
    try:
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicles.append(vehicle)
        print(f"Regular Vehicle {i+1} spawned at: {spawn_point.location}")
    except RuntimeError as e:
        print(f"Failed to spawn regular vehicle {i+1}: {e}")
        break

# Spawn the ego vehicle
ego_spawn_point = spawn_points[3]
target_vehicle = vehicles[3]
ego_spawn_point.location.x -= 10
ego_vehicle = world.spawn_actor(ego_vehicle_bp, ego_spawn_point)
vehicles.append(ego_vehicle)
print(f"Ego Vehicle spawned at: {ego_spawn_point.location}")

# Enable autopilot for all vehicles
tm = client.get_trafficmanager(8000)
tm_port = tm.get_port()
for vehicle in vehicles:
    vehicle.set_autopilot(True, tm_port)

# Set up radar sensor on the ego vehicle
radar_bp = blueprint_library.find('sensor.other.radar')
radar_bp.set_attribute('range', '100.0')
radar_bp.set_attribute('horizontal_fov', '120.0')
radar_bp.set_attribute('vertical_fov', '30.0')
radar_bp.set_attribute('points_per_second', '1500')
radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
radar = world.spawn_actor(radar_bp, radar_transform, attach_to=ego_vehicle)

# Set up camera sensor on the ego vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')
camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

# Set up LIDAR sensor on the ego vehicle
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '100.0')
lidar_bp.set_attribute('points_per_second', '100000')
lidar_bp.set_attribute('rotation_frequency', '20')
lidar_bp.set_attribute('upper_fov', '10')
lidar_bp.set_attribute('lower_fov', '-10')
lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)

# obstacle_detector_bp = blueprint_library.find('sensor.other.obstacle')
# obstacle_detector_bp.distance = 100
# obstacle_detector_bp.only_dynamics = True
# obstacle_detector_bp.sensor_tick = 0.1
# obstacle_detector_transform = carla.Transform(carla.Location(x=0.0, z=0.0))
# obstacle_detector = world.spawn_actor(obstacle_detector_bp, obstacle_detector_transform, attach_to=ego_vehicle)

# Store sensor data
radar_detections = []
latest_camera_image = None
latest_lidar_points = []
radar_2_world = []
world_2_camera = []
cluster = []
track_id = []

# Radar callback
def radar_callback(radar_data):
    global radar_detections
    radar_detections = []
    points = []
    # print(f"Radar callback triggered, frame: {radar_data.frame}, detections: {len(radar_data)}")
    for detection in radar_data:
        try:
            distance = detection.depth
            azimuth = math.degrees(detection.azimuth)
            altitude = math.degrees(detection.altitude)
            velocity = detection.velocity
            radar_detections.append((distance, azimuth, altitude, velocity))
            x = distance * math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth))
            y = distance * math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth))
            z = distance * math.sin(math.radians(altitude))
            vx = velocity * math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth))
            vy = velocity * math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth))
            vz = velocity * math.sin(math.radians(altitude))
            if z > -0.2 and x < 50 and -10 < y < 10:
                points.append([x, y, z, vx, vy, vz])
        except AttributeError as e:
            print(f"AttributeError: {e}. Raw detection: {detection}")
            break

    if points:
        global cluster
        cluster = radar_point_cluster.radar_cluster(points)
        global track_id
        if cluster is not None:
            # track_id = tracker.update(np.concatenate((np.array(cluster)[:, 0:2], np.array(cluster)[:, 6:8]), axis=1))
            track_id = tracker.update(cluster)

        # print_objects_radar_detected()
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('vx', 12, PointField.FLOAT32, 1),
                  PointField('vy', 16, PointField.FLOAT32, 1),
                  PointField('vz', 20, PointField.FLOAT32, 1)]
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "ego_vehicle_radar"
        pc2_msg = pc2.create_cloud(header, fields, points)
        radar_pub.publish(pc2_msg)

        if cluster:
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "ego_vehicle_radar"
            pc2_msg = pc2.create_cloud(header, fields, np.array(cluster)[:, :3])
            radar_cluster_pub.publish(pc2_msg)
        # print(f"Published to /carla/ego_vehicle/radar, points: {len(points)}")

# Camera callback
def camera_callback(image):
    global latest_camera_image
    try:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        latest_camera_image = array
        # ros_image = bridge.cv2_to_imgmsg(array, encoding="rgb8")
        # ros_image.header.stamp = rospy.Time.now()
        # ros_image.header.frame_id = "ego_vehicle_camera"
        # image_pub.publish(ros_image)
        # print(f"Published to /carla/ego_vehicle/camera, frame: {image.frame}")
    except CvBridgeError as e:
        print(f"CvBridgeError: {e}")

# LIDAR callback
def lidar_callback(lidar_data):
    global latest_lidar_points
    points = []
    for point in lidar_data:
        x = point.point.x
        y = point.point.y
        z = point.point.z
        intensity = point.intensity
        points.append([x, y, z, intensity])
    latest_lidar_points = points
    if points:
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "ego_vehicle_lidar"
        pc2_msg = pc2.create_cloud(header, fields, points)
        lidar_pub.publish(pc2_msg)
        # print(f"Published to /carla/ego_vehicle/lidar, points: {len(points)}")

def obstacle_callback(event):
    if isinstance(event, carla.ObstacleDetectionEvent) and event.distance != 0:
        print(f"Detected obstacle at {event.distance}")
    

# Function to get extrinsic parameters from radar to camera
def get_extrinsic_params(radar_sensor, camera_sensor):
    global radar_2_world
    radar_2_world = radar_sensor.get_transform().get_matrix()
    global world_2_camera
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # radar_transform = radar_sensor.get_transform() # radar to world
    # camera_transform = camera_sensor.get_transform() # camera to world

    # radar_rotation = radar_transform.rotation
    # radar_location = radar_transform.location
    # inverse_radar_rotation = carla.Rotation(pitch=-radar_rotation.pitch, yaw=-radar_rotation.yaw, roll=-radar_rotation.roll)
    # inverse_radar_location = carla.Location(x=-radar_location.x, y=-radar_location.y, z=-radar_location.z)

    # inverse_radar_transform = carla.Transform(inverse_radar_location, inverse_radar_rotation)
    
    # radar_to_camera = inverse_radar_transform.transform(camera_transform.location)
    # radar_to_camera_rotation = carla.Rotation(
    #     pitch=camera_transform.rotation.pitch - radar_transform.rotation.pitch,
    #     yaw=camera_transform.rotation.yaw - radar_transform.rotation.yaw,
    #     roll=camera_transform.rotation.roll - radar_transform.rotation.roll
    # )
    # radar_to_camera_transform = carla.Transform(radar_to_camera, radar_to_camera_rotation)
    # print("Extrinsic parameters from radar to camera:")
    # print(f"Position: {radar_to_camera_transform.location}")
    # print(f"Rotation: {radar_to_camera_transform.rotation}")
    # return radar_to_camera_transform


# Function to project radar points to camera image
def project_radar_to_camera(radar_points, image_width=800, image_height=600, fov=90):
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
    
    for x, y, z, w, l, h, vx, vy, vz, id in radar_points:
        if z > -0.2:
            radar_point = np.array([x, y, z, 1])
            world_point = np.dot(radar_2_world, radar_point)
            camera_point = np.dot(world_2_camera, world_point)
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
                ipm_point = np.dot(lane_detector.M, np.array([u, v - 300, 1]))
                ipm_point[0] = ipm_point[0] / ipm_point[2]
                ipm_point[1] = ipm_point[1] / ipm_point[2]
                # ipm_point = np.array([u, v - 300])
                # print(u, " ", v, " -> ", ipm_point[0], "   ", ipm_point[1])
                projected_points.append([int(u), int(v), int(ipm_point[0]), int(ipm_point[1])])
    
    return projected_points#, track_id

# Function to compute IPM transformation matrix
def get_ipm_transform_matrix(camera_sensor, K, image_width=800, image_height=600):
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

# Function to get extrinsic parameters from LIDAR to camera
def get_lidar_to_camera_transform(lidar_sensor, camera_sensor):
    lidar_transform = lidar_sensor.get_transform()
    camera_transform = camera_sensor.get_transform()
    lidar_rotation = lidar_transform.rotation
    lidar_location = lidar_transform.location
    inverse_lidar_rotation = carla.Rotation(pitch=-lidar_rotation.pitch, yaw=-lidar_rotation.yaw, roll=-lidar_rotation.roll)
    inverse_lidar_location = carla.Location(x=-lidar_location.x, y=-lidar_location.y, z=-lidar_location.z)
    inverse_lidar_transform = carla.Transform(inverse_lidar_location, inverse_lidar_rotation)
    lidar_to_camera = inverse_lidar_transform.transform(camera_transform.location)
    lidar_to_camera_rotation = carla.Rotation(
        pitch=camera_transform.rotation.pitch - lidar_transform.rotation.pitch,
        yaw=camera_transform.rotation.yaw - lidar_transform.rotation.yaw,
        roll=camera_transform.rotation.roll - lidar_transform.rotation.roll
    )
    lidar_to_camera_transform = carla.Transform(lidar_to_camera, lidar_to_camera_rotation)
    print("Extrinsic parameters from LIDAR to camera:")
    print(f"Position: {lidar_to_camera_transform.location}")
    print(f"Rotation: {lidar_to_camera_transform.rotation}")
    return lidar_to_camera_transform

# Attach sensor listeners
radar.listen(radar_callback)
camera.listen(camera_callback)
# lidar.listen(lidar_callback)
# obstacle_detector.listen(obstacle_callback)

# Function to estimate distance to target vehicle using LIDAR
def estimate_target_distance(ego_vehicle, target_vehicle, lidar_data):
    if not target_vehicle or not lidar_data:
        return None
    
    target_location = target_vehicle.get_location()
    ego_location = ego_vehicle.get_location()
    min_dist = float('inf')
    closest_point = None
    
    for x, y, z, _ in lidar_data:
        # Transform LIDAR point to world coordinates
        lidar_point = carla.Location(x, y, z)
        lidar_transform = lidar.get_transform()
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
def follow_target_vehicle(ego_vehicle, target_distance, desired_distance=10.0, kp=0.1):
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
    target_location = target_vehicle.get_location()
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

# Main loop
try:
    print("Starting radar, camera, LIDAR, and object detection ROS publishing (press Ctrl+C to stop)...")
    get_extrinsic_params(radar, camera)
    # ipm_matrix = get_ipm_transform_matrix(camera)
    # lidar_to_camera_transform = get_lidar_to_camera_transform(lidar, camera)
    while not rospy.is_shutdown():
        clock_msg = Clock()
        clock_msg.clock = rospy.Time.from_sec(time.time())
        clock_pub.publish(clock_msg)
        world.tick()

        # Estimate distance to target and control ego vehicle
        target_distance = estimate_target_distance(ego_vehicle, target_vehicle, latest_lidar_points)
        follow_target_vehicle(ego_vehicle, target_distance)

        # if latest_camera_image is not None:
        #     cv2.imshow("camera", latest_camera_image)
        #     cv2.waitKey(1)

        # Visualize radar on camera
        if latest_camera_image is not None:
            image_with_radar = latest_camera_image.copy()
            lane_windows, lane_image = lane_detector.lane_detect(image_with_radar)

            if track_id is not None:
                projected_points = project_radar_to_camera(track_id)
                current_target_idx = -1
                # print(projected_points)
                if projected_points is not None:
                    for idx, point in enumerate(projected_points):
                        u, v, ipm_u, ipm_v = point
                        # v = point[1]
                        # distance = point[3]
                        # ipm_u = point[7]
                        # ipm_v = point[8]
                        color = (255, 0, 0) if track_id[idx][0] < 10 else (0, 255, 0)
                        cv2.circle(image_with_radar, (u, v), 5, color, -1)
                        if (idx < len(track_id)):
                            cv2.putText(image_with_radar, str(track_id[idx][-1]),(u,v), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,225,100), 2)
                        if 0 <= ipm_u < lane_image.shape[0] and 0 <= ipm_v < lane_image.shape[1]:
                            cv2.circle(lane_image, (ipm_u, ipm_v), 5, (255, 0, 0), -1)

                        # #select target in lane windows
                        for windows in lane_windows:
                            if (windows[0][0] < ipm_u < windows[3][0]) & ( windows[0][1] < ipm_v < windows[1][1]):  # object in current lane
                                if (current_target_idx != -1) & (track_id[idx][0] < track_id[current_target_idx][0]) | \
                                    (current_target_idx == -1) & (track_id[idx][0] < max_follow_distance) :
                                    current_target_idx = idx

                        #select target based on radar points
                        if (-1.5 < track_id[idx][1] < 1.5) and (track_id[idx][0] < max_follow_distance):
                            # if ((current_target_idx != -1) and (track_id[idx][0] < track_id[current_target_idx][0])) or (current_target_idx == -1):
                            current_target_idx = idx

                    if current_target_idx >= 0:
                        cv2.circle(image_with_radar, (projected_points[current_target_idx][0], projected_points[current_target_idx][1]), 10, (255, 255, 255), -1)
                        cv2.circle(lane_image, (projected_points[current_target_idx][2], projected_points[current_target_idx][3]), 10, (255, 255, 0), -1)
                    
                        print("current target info (x, y, z, w, l, h, vx, vy, vz): ", track_id[current_target_idx])

                    cluster = []
                    track_id = []

            # cv2.imshow("lane", lane_image)
            # cv2.imshow("Radar and Objects on Camera", image_with_radar)
            # cv2.waitKey(1)
            ros_image = bridge.cv2_to_imgmsg(image_with_radar, encoding="rgb8")
            ros_image.header.stamp = rospy.Time.now()
            ros_image.header.frame_id = "ego_vehicle_camera"
            image_pub.publish(ros_image)

            ros_lane_image = bridge.cv2_to_imgmsg(lane_image, encoding="rgb8")
            ros_lane_image.header.stamp = rospy.Time.now()
            ros_lane_image.header.frame_id = "ego_vehicle_camera_lane"
            lane_image_pub.publish(ros_lane_image)
            
        
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    cv2.destroyAllWindows()

# Cleanup
radar.stop()
camera.stop()
lidar.stop()
radar.destroy()
camera.destroy()
lidar.destroy()
for vehicle in vehicles:
    vehicle.set_autopilot(False, tm_port)
    vehicle.destroy()
print(f"Destroyed {len(vehicles)} vehicles, radar, camera, and LIDAR.")