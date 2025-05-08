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

        self.init_ros()

        # self.radar.listen(self.radar_callback)
        # self.camera.listen(self.camera_callback)
        # self.lidar.listen(self.lidar_callback)


    def init_ros(self):
        # Initialize ROS node
        rospy.init_node('carla_ego_publisher', anonymous=True)
        self.image_sub = rospy.Subscriber('/carla/ego_vehicle/camera', Image, self.camera_callback)
        self.lane_image_pub = rospy.Publisher('/carla/ego_vehicle/lane_image', Image, queue_size=10)
        self.point_image_pub = rospy.Publisher('/carla/ego_vehicle/point_image', Image, queue_size=10)
        self.radar_sub = rospy.Subscriber('/carla/ego_vehicle/radar', PointCloud2, self.radar_callback)
        self.radar_cluster_pub = rospy.Publisher('/carla/ego_vehicle/radar_cluster', PointCloud2, queue_size=10)
        self.lidar_sub = rospy.Subscriber('/carla/ego_vehicle/lidar', PointCloud2, self.lidar_callback)
        self.objects_pub = rospy.Publisher('/carla/ego_vehicle/detected_objects', PointCloud2, queue_size=10)  # Simplified as PointCloud2
        self.clock_pub = rospy.Publisher('/clock', Clock, queue_size=10)
        self.bridge = CvBridge()

        # Radar callback
    def radar_callback(self, radar_data):
        points = list(pc2.read_points(radar_data, field_names=("x", "y", "z", "vx", "vy", "vz", "v"), skip_nans=True))
        print(len(points), "  ", len(points[0]))
        for row in points:
            row[1] = -row[1]
        cluster_points = []
        for point in points:
            cluster_points.append(point)
        
        if points:
            # if cluster_points:
            self.cluster = self.radar_point_cluster.radar_cluster(cluster_points)
            if self.cluster:
                # track_id = tracker.update(np.concatenate((np.array(cluster)[:, 0:2], np.array(cluster)[:, 6:8]), axis=1))
                self.track_id = self.tracker.update(self.cluster)
                fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1)]
                header = rospy.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "ego_vehicle_radar"
                pc2_msg = pc2.create_cloud(header, fields, np.array(self.cluster)[:, :3])
                self.radar_cluster_pub.publish(pc2_msg)
            else:
                self.track_id = []


    # Camera callback
    def camera_callback(self, image):
        self.latest_camera_image = cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")

    # LIDAR callback
    def lidar_callback(self, lidar_data):
        self.latest_lidar_points = list(pc2.read_points(lidar_data, field_names=("x", "y", "z", "intensity"), skip_nans=True))


    # Function to get extrinsic parameters from radar to camera
    def get_extrinsic_params(self):
        self.radar_2_world = [[0.9999867081642151, 0.005154258105903864, 0.0, 217.25201416015625], 
                              [-0.005154258105903864, 0.9999867081642151, 0.0, -364.11968994140625], 
                              [0.0, -0.0, 1.0, 0.9192002415657043], 
                              [0.0, 0.0, 0.0, 1.0]]
        self.world_2_camera = [[ 9.99986708e-01, -5.15425811e-03,  0.00000000e+00, -2.18625885e+02],
                                [ 5.15425811e-03,  9.99986708e-01, -0.00000000e+00,  3.62995056e+02],
                                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00, -1.41920030e+00],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

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

    def generate_target(self):
        try:
            print("Starting radar, camera, LIDAR, and object detection ROS publishing (press Ctrl+C to stop)...")
            self.get_extrinsic_params()
            # ipm_matrix = get_ipm_transform_matrix(camera)
            # lidar_to_camera_transform = get_lidar_to_camera_transform(lidar, camera)


            while not rospy.is_shutdown():
                # if latest_camera_image is not None:
                #     cv2.imshow("camera", latest_camera_image)
                #     cv2.waitKey(1)out_img

                # select target
                if self.latest_camera_image is not None:
                    image_with_radar = self.latest_camera_image.copy()
                    lane_windows, lane_image = self.lane_detector.lane_detect(image_with_radar)
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
                                
                            if 0 <= ipm_u < lane_image.shape[0] and 0 <= ipm_v < lane_image.shape[1]:
                                cv2.circle(lane_image, (ipm_u, ipm_v), 5, (255, 0, 0), -1)

                            # #select target in lane windows
                            for windows in lane_windows:
                                if (windows[0][0] < ipm_u < windows[3][0]) and ( windows[0][1] < ipm_v < windows[1][1]):  # object in current lane
                                    if ((current_target_idx != -1) and (self.track_id[idx][0] < self.track_id[current_target_idx][0])) or \
                                        ((current_target_idx == -1) and (self.track_id[idx][0] < self.max_follow_distance)) :
                                        current_target_idx = idx

                            # # emergency stop
                            if (-1.5 < track_id[idx][1] < 1.5) and track_id[idx][6] > -1 : #and (track_id[idx][0] < 20):
                                if ((current_target_idx != -1) and (track_id[idx][0] < track_id[current_target_idx][0])) or (current_target_idx == -1):
                                    current_target_idx = idx

                        if current_target_idx >= 0:
                            cv2.circle(image_with_radar, (projected_points[current_target_idx][0], projected_points[current_target_idx][1]), 10, (255, 255, 255), -1)
                            cv2.circle(lane_image, (projected_points[current_target_idx][2], projected_points[current_target_idx][3]), 10, (255, 255, 0), -1)
                            cv2.putText(image_with_radar, "id=" + str(track_id[current_target_idx][-1]), (projected_points[current_target_idx][0] + 5, projected_points[current_target_idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,225,100), 2)
                            print("current target info (x, y, z, w, l, h, vx, vy, vz, track_id): ", track_id[current_target_idx])

                    # cv2.imshow("lane", lane_image)
                    # cv2.imshow("Radar and Objects on Camera", image_with_radar)
                    # cv2.waitKey(1)
                    ros_image = self.bridge.cv2_to_imgmsg(image_with_radar, encoding="rgb8")
                    ros_image.header.stamp = rospy.Time.now()
                    ros_image.header.frame_id = "ego_vehicle_camera"
                    self.point_image_pub.publish(ros_image)

                    ros_lane_image = self.bridge.cv2_to_imgmsg(lane_image, encoding="rgb8")
                    ros_lane_image.header.stamp = rospy.Time.now()
                    ros_lane_image.header.frame_id = "ego_vehicle_camera_lane"
                    self.lane_image_pub.publish(ros_lane_image)
                    
                
                # time.sleep(0.005)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cv2.destroyAllWindows()

def main():
    acc_actor = acc()
    
    # thread_1 = threading.Thread(target=acc_actor.radar.listen(acc_actor.radar_callback), name='T1')
    # thread_2 = threading.Thread(target=acc_actor.camera.listen(acc_actor.camera_callback), name='T2')
    # thread_3 = threading.Thread(target=acc_actor.lidar.listen(acc_actor.lidar_callback), name='T3')
    # # thread_4 = threading.Thread(target=acc_actor.generate_target(), name='T4')

    # thread_1.start()  # 开启T1
    # thread_2.start()  # 开启T2
    # thread_3.start()
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