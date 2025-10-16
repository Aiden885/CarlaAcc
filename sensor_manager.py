import math
import numpy as np
import cv2
import carla

class SensorManager:
    def __init__(self, tracker, lane_detector, radar_point_cluster, ego_vehicle, world):
        self.tracker = tracker
        self.lane_detector = lane_detector
        self.radar_point_cluster = radar_point_cluster
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.show_opencv = True
        self.latest_camera_image = None
        self.radar_points = []
        self.filted_points = []
        self.latest_lidar_points = []
        self.radar_2_world = []
        self.world_2_camera = []

    def radar_callback(self, radar_data, ego_vehicle):
        """雷达数据回调，处理雷达数据并返回 cluster 和 track_id"""
        self.radar_points = []
        self.filted_points = []
        ego_velocity = self.get_vehicle_speed(ego_vehicle) / 3.6
        velocity_tolerance = 1.0
        cluster = []
        track_id = []
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
                expected_static_velocity = -ego_velocity * math.cos(math.radians(azimuth)) * math.cos(
                    math.radians(altitude))
                if z > -0.5:
                    self.radar_points.append([x, y, z, vx, vy, vz, velocity])
                    if abs(velocity - expected_static_velocity) > velocity_tolerance:
                        self.filted_points.append([x, y, z, vx, vy, vz, velocity])
            except AttributeError as e:
                print(f"AttributeError: {e}. Raw detection: {detection}")
        if self.filted_points:
            cluster = self.radar_point_cluster.radar_cluster(self.filted_points)
            if cluster:
                track_id = self.tracker.update(cluster)
                for track in track_id:
                    if not all(np.isfinite(track)):
                        print(f"Invalid track data: {track}")
                        track_id = []
                        break
        return cluster, track_id

    def camera_callback(self, image):
        """相机数据回调"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]
            self.latest_camera_image = array
            print(f"Camera image updated: shape={array.shape}")
            return array
        except Exception as e:
            print(f"Camera callback error: {e}")
            return None

    def lidar_callback(self, lidar_data):
        """激光雷达数据回调"""
        points = []
        for point in lidar_data:
            x = point.point.x
            y = point.point.y
            z = point.point.z
            intensity = point.intensity
            points.append([x, y, z, intensity])
        self.latest_lidar_points = points
        return points

    def get_extrinsic_params(self, radar_sensor, camera_sensor):
        """获取传感器外参"""
        self.radar_2_world = radar_sensor.get_transform().get_matrix()
        self.world_2_camera = np.array(camera_sensor.get_transform().get_inverse_matrix())

    def project_radar_to_camera(self, radar_points, image_width=1280, image_height=720, fov=90):
        """将雷达点投影到相机图像"""
        fx = image_width / (2.0 * np.tan(fov * np.pi / 360.0))
        fy = image_height / (2.0 * np.tan(fov * np.pi / 360.0))
        cx = image_width / 2
        cy = image_height / 2
        projected_points = []
        for x, y, z, w, l, h, vx, vy, vz, id in radar_points:
            radar_point = np.array([x, y, z, 1])
            world_point = np.dot(self.radar_2_world, radar_point)
            camera_point = np.dot(self.world_2_camera, world_point)
            point_in_camera_coords = np.array([
                camera_point[1],
                camera_point[2] * -1,
                camera_point[0]])
            u = cx + (fx * point_in_camera_coords[0] / point_in_camera_coords[2])
            v = cy + (fy * point_in_camera_coords[1] / point_in_camera_coords[2])
            ipm_point = np.dot(self.lane_detector.M, np.array([u, v - 300, 1]))
            ipm_point[0] = ipm_point[0] / ipm_point[2]
            ipm_point[1] = ipm_point[1] / ipm_point[2]
            projected_points.append([int(u), int(v), int(ipm_point[0]), int(ipm_point[1])])
        return projected_points

    def find_best_target(self, track_id, projected_points):
        """选择最佳目标"""
        current_target_idx = -1
        min_distance = float('inf')

        for idx in range(len(track_id)):
            if -3 < track_id[idx][1] < 3:
                if track_id[idx][0] < min_distance:
                    min_distance = track_id[idx][0]
                    current_target_idx = idx

        return current_target_idx

    def get_target_info(self, track_id):
        """获取目标信息"""
        if not track_id:
            return None
        projected_points = self.project_radar_to_camera(track_id)
        current_target_idx = self.find_best_target(track_id, projected_points)
        if current_target_idx >= 0 and current_target_idx < len(track_id):
            return track_id[current_target_idx]
        return None

    def get_lane_center(self, image_with_radar):
        """获取车道中心"""
        lane_center = 510  # 默认值（图像中心，1280/2）
        print(f"Input image_with_radar: {'Valid' if image_with_radar is not None else 'None'}")
        if image_with_radar is not None:
            try:
                result = self.lane_detector.lane_detect(image_with_radar)
                if result is None:
                    print("Lane detection returned None")
                else:
                    lane_windows, lane_image, detected_windows = result
                    print(f"Lane windows: {lane_windows}")
                    if lane_windows is not None and len(lane_windows) > 0:
                        valid_row = None
                        for row in lane_windows:
                            if row is not None and len(row) == 6 and row[2] == 1 and row[5] == 1:
                                valid_row = row
                                break
                        if valid_row is not None:
                            lane_center = (valid_row[0] + valid_row[3]) / 2
                            print(f"Valid lane center: {lane_center}")
                        else:
                            print("No valid lane row detected")
                    else:
                        print("Lane windows is None or empty")
            except Exception as e:
                print(f"Lane detection error: {e}")
        else:
            print("No camera image, using CARLA waypoint")
        # Fallback to CARLA waypoint
        if lane_center == 510 and self.world and self.ego_vehicle:
            try:
                vehicle_location = self.ego_vehicle.get_location()
                waypoint = self.world.get_map().get_waypoint(
                    vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
                )
                if waypoint:
                    lane_center_x = waypoint.transform.location.x
                    vehicle_x = vehicle_location.x
                    # 校准缩放因子：假设相机FOV=90°，车道宽度~3.5m
                    fx = 1280 / (2 * np.tan(90 * np.pi / 360))  # 焦距（像素）
                    scale_factor = fx / 1.0  # 假设车道距离1m
                    lane_center = 510 + (lane_center_x - vehicle_x) * scale_factor
                    print(f"CARLA waypoint lane center: {lane_center}")
            except Exception as e:
                print(f"CARLA waypoint error: {e}")
        return lane_center

    def process_camera_image(self, camera_image, track_id, acc_status, acc_control_active, acc_params):
        """处理相机图像并添加ACC信息"""
        print(f"Processing camera image: {'Valid' if camera_image is not None else 'None'}")
        if camera_image is None:
            return None

        image_with_radar = camera_image.copy()

        target_info = None
        if track_id:
            try:
                projected_points = self.project_radar_to_camera(track_id)
                current_target_idx = self.find_best_target(track_id, projected_points)

                for idx in range(min(len(track_id), len(projected_points))):
                    if len(projected_points[idx]) >= 2:
                        u, v = projected_points[idx][0], projected_points[idx][1]
                        cv2.circle(image_with_radar, (u, v), 5, (255, 0, 0), -1)

                if current_target_idx >= 0 and current_target_idx < len(projected_points):
                    u, v = projected_points[current_target_idx][0], projected_points[current_target_idx][1]
                    cv2.circle(image_with_radar, (u, v), 10, (255, 255, 255), -1)
                    cv2.putText(image_with_radar, f"id={track_id[current_target_idx][-1]:.0f}",
                                (u + 5, v), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 225, 100), 2)
                    target_info = track_id[current_target_idx]

            except Exception as e:
                print(f"Target detection error: {e}")

        y_offset = 10
        cv2.putText(image_with_radar, f"ACC: {acc_status['state_description']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

        cv2.putText(image_with_radar, f"Active: {'YES' if acc_control_active else 'NO'}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if acc_control_active else (255, 255, 255), 2)
        y_offset += 25

        if self.show_opencv:
            cv2.imshow("Radar and Lane Detection", image_with_radar)
            cv2.waitKey(1)

        return image_with_radar

    def get_vehicle_speed(self, vehicle):
        """获取车辆速度（km/h）"""
        if vehicle is None:
            return 0.0
        velocity = vehicle.get_velocity()
        speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_kmh = speed_m_s * 3.6
        return speed_kmh
