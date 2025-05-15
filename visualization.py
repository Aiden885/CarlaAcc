import cv2
import numpy as np


class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.color = (0, 255, 255)  # 黄色
        self.thickness = 2

    def draw_radar_points(self, image, points, color=(255, 0, 0), radius=5):
        """绘制雷达检测点"""
        for point in points:
            u, v = point[0], point[1]
            cv2.circle(image, (u, v), radius, color, -1)
        return image

    def draw_target_info(self, image, target_info, projected_point, track_id):
        """绘制目标信息"""
        if target_info is None or projected_point is None:
            return image

        # 绘制目标圆圈
        cv2.circle(image, (projected_point[0], projected_point[1]), 10, (255, 255, 255), -1)

        # 绘制目标ID
        cv2.putText(image, f"ID={track_id}",
                    (projected_point[0] + 5, projected_point[1]),
                    self.font, self.font_scale, (150, 225, 100), self.thickness)

        # 绘制距离信息
        distance = np.sqrt(target_info[0] ** 2 + target_info[1] ** 2)
        cv2.putText(image, f"Distance: {distance:.1f}m",
                    (projected_point[0] + 5, projected_point[1] + 25),
                    self.font, self.font_scale, (150, 225, 100), self.thickness)

        # 绘制相对速度
        rel_velocity = target_info[6]
        cv2.putText(image, f"Rel V: {rel_velocity:.1f}m/s",
                    (projected_point[0] + 5, projected_point[1] + 50),
                    self.font, self.font_scale, (150, 225, 100), self.thickness)

        return image

    def draw_vehicle_status(self, image, current_speed, is_curved_road, target_speed_diff):
        """绘制车辆状态信息"""
        # 计算当前百分比速度（假设道路限速为100km/h）
        speed_percentage = 100 - target_speed_diff

        # 道路状态
        road_status = "弯道" if is_curved_road else "直道"

        # 绘制信息
        cv2.putText(image, f"目标车速: {current_speed:.1f} km/h ({speed_percentage:.1f}%)",
                    (50, 50), self.font, self.font_scale, self.color, self.thickness)
        cv2.putText(image, f"道路状态: {road_status}",
                    (50, 80), self.font, self.font_scale, self.color, self.thickness)

        return image

    def draw_lane_info(self, image, lane_windows):
        """绘制车道信息"""
        if lane_windows is None:
            return image

        for window in lane_windows:
            x, y, w, h = window
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image