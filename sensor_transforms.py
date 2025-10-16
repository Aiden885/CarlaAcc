"""
Sensor coordinate transformation utilities
传感器坐标变换工具模块 - 提供雷达/相机坐标系转换
"""
import numpy as np


class SensorTransforms:
    """传感器坐标变换工具类"""

    @staticmethod
    def get_extrinsic_params(radar_sensor, camera_sensor):
        """
        获取雷达和相机的外参矩阵

        Args:
            radar_sensor: CARLA雷达传感器对象
            camera_sensor: CARLA相机传感器对象

        Returns:
            tuple: (radar_2_world, world_2_camera) 变换矩阵
        """
        radar_2_world = radar_sensor.get_transform().get_matrix()
        world_2_camera = np.array(camera_sensor.get_transform().get_inverse_matrix())
        return radar_2_world, world_2_camera

    @staticmethod
    def project_radar_to_camera(radar_points, radar_2_world, world_2_camera, M,
                                image_width=1280, image_height=720, fov=90):
        """
        将雷达点投影到相机图像坐标系

        Args:
            radar_points: 雷达点列表，每个点格式为 [x, y, z, w, l, h, vx, vy, vz, id]
            radar_2_world: 雷达到世界坐标系的变换矩阵
            world_2_camera: 世界坐标系到相机坐标系的变换矩阵
            M: IPM (Inverse Perspective Mapping) 变换矩阵
            image_width: 图像宽度（像素）
            image_height: 图像高度（像素）
            fov: 相机视场角（度）

        Returns:
            list: 投影点列表，每个点格式为 [u, v, ipm_u, ipm_v]
                  u, v: 图像坐标
                  ipm_u, ipm_v: IPM变换后的坐标
        """
        # 计算相机内参
        fx = image_width / (2.0 * np.tan(fov * np.pi / 360.0))
        fy = image_height / (2.0 * np.tan(fov * np.pi / 360.0))
        cx = image_width / 2
        cy = image_height / 2

        projected_points = []
        for x, y, z, w, l, h, vx, vy, vz, id in radar_points:
            # 雷达坐标 -> 世界坐标
            radar_point = np.array([x, y, z, 1])
            world_point = np.dot(radar_2_world, radar_point)

            # 世界坐标 -> 相机坐标
            camera_point = np.dot(world_2_camera, world_point)

            # 相机坐标系调整（CARLA坐标系转换）
            point_in_camera_coords = np.array([
                camera_point[1],
                camera_point[2] * -1,
                camera_point[0]
            ])

            # 投影到图像平面
            u = cx + (fx * point_in_camera_coords[0] / point_in_camera_coords[2])
            v = cy + (fy * point_in_camera_coords[1] / point_in_camera_coords[2])

            # IPM变换（鸟瞰图变换）
            ipm_point = np.dot(M, np.array([u, v - 300, 1]))
            ipm_point[0] = ipm_point[0] / ipm_point[2]
            ipm_point[1] = ipm_point[1] / ipm_point[2]

            projected_points.append([int(u), int(v), int(ipm_point[0]), int(ipm_point[1])])

        return projected_points