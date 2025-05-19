import carla
import numpy as np
import math


class CarlaTrajectoryVisualizer:
    """CARLA 3D轨迹可视化器"""

    def __init__(self, world):
        self.world = world
        self.trajectory_points = []
        self.trajectory_lines = []
        self.last_trajectory_version = -1

        # 可视化配置
        self.config = {
            'trajectory_color': carla.Color(0, 255, 0, 200),  # 绿色轨迹线
            'waypoint_color': carla.Color(255, 255, 0, 255),  # 黄色轨迹点
            'curvature_color_high': carla.Color(255, 0, 0, 200),  # 高曲率红色
            'curvature_color_low': carla.Color(0, 255, 0, 200),  # 低曲率绿色
            'tangent_color': carla.Color(0, 0, 255, 150),  # 蓝色切线
            'ground_height': 0.1,  # 距离地面高度
            'life_time': 0.1,  # 显示持续时间
            'thickness': 0.15,  # 线条粗细
            'point_size': 0.08,  # 点大小
        }

    def clear_previous_trajectory(self):
        """清除之前的轨迹可视化（通过设置较短的life_time自动清除）"""
        pass  # CARLA的debug绘制会自动消失

    def visualize_trajectory_3d(self, trajectory_manager, show_curvature=True,
                                show_tangents=True, show_speed=False):
        """在CARLA 3D世界中可视化轨迹"""

        # 获取轨迹点
        waypoints = trajectory_manager.buffer.get_waypoints()
        if not waypoints:
            return

        # 检查轨迹是否更新
        trajectory_info = trajectory_manager.buffer.get_info()
        if trajectory_info['version'] == self.last_trajectory_version:
            return  # 没有更新，跳过绘制
        self.last_trajectory_version = trajectory_info['version']

        # 1. 绘制轨迹线
        self._draw_trajectory_lines(waypoints, show_curvature)

        # 2. 绘制轨迹点
        self._draw_trajectory_points(waypoints, show_speed)

        # 3. 绘制切线方向（可选）
        if show_tangents:
            self._draw_tangent_vectors(waypoints)

        # 4. 绘制轨迹信息
        self._draw_trajectory_info(waypoints, trajectory_info)

    def _draw_trajectory_lines(self, waypoints, show_curvature=True):
        """绘制轨迹连线"""
        if len(waypoints) < 2:
            return

        for i in range(len(waypoints) - 1):
            wp1, wp2 = waypoints[i], waypoints[i + 1]

            # 起点和终点
            start_loc = carla.Location(
                x=wp1.world_x,
                y=wp1.world_y,
                z=wp1.world_z + self.config['ground_height']
            )
            end_loc = carla.Location(
                x=wp2.world_x,
                y=wp2.world_y,
                z=wp2.world_z + self.config['ground_height']
            )

            # 根据曲率选择颜色
            if show_curvature:
                avg_curvature = (wp1.curvature + wp2.curvature) / 2
                color = self._get_curvature_color(avg_curvature)
            else:
                color = self.config['trajectory_color']

            # 绘制线段
            self.world.debug.draw_line(
                start_loc, end_loc,
                thickness=self.config['thickness'],
                color=color,
                life_time=self.config['life_time']
            )

    def _draw_trajectory_points(self, waypoints, show_speed=False):
        """绘制轨迹点"""
        for i, wp in enumerate(waypoints):
            location = carla.Location(
                x=wp.world_x,
                y=wp.world_y,
                z=wp.world_z + self.config['ground_height']
            )

            # 根据速度或者索引选择点的大小和颜色
            if show_speed:
                # 根据速度调整点的大小
                size = max(0.05, min(0.15, wp.speed / 50.0 * 0.15))
                color = self._get_speed_color(wp.speed)
            else:
                size = self.config['point_size']
                # 起点绿色，终点红色，中间点黄色
                if i == 0:
                    color = carla.Color(0, 255, 0)  # 起点绿色
                elif i == len(waypoints) - 1:
                    color = carla.Color(255, 0, 0)  # 终点红色
                else:
                    color = self.config['waypoint_color']  # 中间点黄色

            # 绘制点
            self.world.debug.draw_point(
                location,
                size=size,
                color=color,
                life_time=self.config['life_time']
            )

            # 每5个点显示一次编号
            if i % 5 == 0:
                self.world.debug.draw_string(
                    carla.Location(location.x, location.y, location.z + 0.5),
                    str(i),
                    draw_shadow=False,
                    color=carla.Color(255, 255, 255),
                    life_time=self.config['life_time']
                )

    def _draw_tangent_vectors(self, waypoints):
        """绘制切线方向向量"""
        for wp in waypoints[::3]:  # 每3个点绘制一次切线
            start_loc = carla.Location(
                x=wp.world_x,
                y=wp.world_y,
                z=wp.world_z + self.config['ground_height']
            )

            # 计算切线终点
            tangent_length = 2.0
            end_loc = carla.Location(
                x=wp.world_x + tangent_length * np.cos(wp.tangent_angle),
                y=wp.world_y + tangent_length * np.sin(wp.tangent_angle),
                z=wp.world_z + self.config['ground_height']
            )

            # 绘制箭头
            self.world.debug.draw_arrow(
                start_loc, end_loc,
                thickness=0.05,
                arrow_size=0.3,
                color=self.config['tangent_color'],
                life_time=self.config['life_time']
            )

    def _draw_trajectory_info(self, waypoints, trajectory_info):
        """在轨迹起点显示信息"""
        if not waypoints:
            return

        # 信息文本
        info_text = f"Points: {trajectory_info['waypoint_count']}\n"
        info_text += f"Length: {trajectory_info['total_length']:.1f}m\n"
        info_text += f"Valid: {trajectory_info['is_valid']}"

        # 在轨迹起点上方显示信息
        info_location = carla.Location(
            x=waypoints[0].world_x,
            y=waypoints[0].world_y,
            z=waypoints[0].world_z + 2.0
        )

        self.world.debug.draw_string(
            info_location,
            info_text,
            draw_shadow=True,
            color=carla.Color(255, 255, 255),
            life_time=self.config['life_time']
        )

    def _get_curvature_color(self, curvature):
        """根据曲率返回颜色"""
        # 曲率范围 [0, 0.2] 映射到 [绿色, 红色]
        curvature = min(max(curvature, 0), 0.2)
        t = curvature / 0.2

        r = int(255 * t)
        g = int(255 * (1 - t))
        b = 0

        return carla.Color(r, g, b, 200)

    def _get_speed_color(self, speed):
        """根据速度返回颜色"""
        # 速度范围 [0, 50] km/h 映射到 [蓝色, 绿色, 红色]
        speed = min(max(speed, 0), 50)
        t = speed / 50.0

        if t < 0.5:
            # 蓝色到绿色
            r = 0
            g = int(255 * 2 * t)
            b = int(255 * (1 - 2 * t))
        else:
            # 绿色到红色
            r = int(255 * 2 * (t - 0.5))
            g = int(255 * (2 - 2 * t))
            b = 0

        return carla.Color(r, g, b, 255)

    def visualize_lookahead_path(self, trajectory_manager, ego_vehicle,
                                 lookahead_distance=30.0):
        """可视化前瞻路径（用于控制）"""
        # 获取前瞻轨迹点
        control_waypoints = trajectory_manager.get_control_waypoints(lookahead_distance)
        if not control_waypoints:
            return

        # 获取自车位置
        ego_location = ego_vehicle.get_location()

        # 绘制前瞻路径
        for i, wp in enumerate(control_waypoints):
            location = carla.Location(
                x=wp.world_x,
                y=wp.world_y,
                z=wp.world_z + 0.3  # 比主轨迹稍高
            )

            # 距离自车越近，颜色越亮
            distance_to_ego = math.sqrt(
                (wp.world_x - ego_location.x) ** 2 +
                (wp.world_y - ego_location.y) ** 2
            )
            intensity = max(0, 1.0 - distance_to_ego / lookahead_distance)

            color = carla.Color(
                int(255 * intensity),
                int(255 * (1.0 - intensity)),
                0,
                200
            )

            # 绘制点，大小随距离变化
            size = 0.05 + 0.1 * intensity
            self.world.debug.draw_point(
                location,
                size=size,
                color=color,
                life_time=self.config['life_time']
            )

        # 绘制从自车到第一个前瞻点的连接线
        if control_waypoints:
            start = carla.Location(
                x=ego_location.x,
                y=ego_location.y,
                z=ego_location.z + 0.5
            )
            end = carla.Location(
                x=control_waypoints[0].world_x,
                y=control_waypoints[0].world_y,
                z=control_waypoints[0].world_z + 0.5
            )

            self.world.debug.draw_line(
                start, end,
                thickness=0.1,
                color=carla.Color(0, 255, 0),
                life_time=self.config['life_time']
            )

    def visualize_trajectory_dense(self, trajectory_manager, point_interval=1.0):
        """绘制密集的轨迹点（适合长距离可视化）"""
        waypoints = trajectory_manager.buffer.get_waypoints()
        if not waypoints:
            return

        # 按距离间隔选择点
        selected_points = []
        last_distance = 0

        for wp in waypoints:
            if wp.distance_from_start - last_distance >= point_interval:
                selected_points.append(wp)
                last_distance = wp.distance_from_start

        # 绘制选中的点和线
        for i, wp in enumerate(selected_points):
            location = carla.Location(
                x=wp.world_x,
                y=wp.world_y,
                z=wp.world_z + self.config['ground_height']
            )

            # 绘制点
            self.world.debug.draw_point(
                location,
                size=self.config['point_size'],
                color=self.config['waypoint_color'],
                life_time=self.config['life_time']
            )

            # 绘制到下一点的连线
            if i < len(selected_points) - 1:
                next_wp = selected_points[i + 1]
                next_loc = carla.Location(
                    x=next_wp.world_x,
                    y=next_wp.world_y,
                    z=next_wp.world_z + self.config['ground_height']
                )

                # 根据平均曲率选择颜色
                avg_curvature = (wp.curvature + next_wp.curvature) / 2
                color = self._get_curvature_color(avg_curvature)

                self.world.debug.draw_line(
                    location, next_loc,
                    thickness=self.config['thickness'],
                    color=color,
                    life_time=self.config['life_time']
                )


# 在acc.py中的集成方法
def integrate_3d_trajectory_visualization(self):
    """集成3D轨迹可视化到acc类中"""

    # 在__init__方法中添加
    self.trajectory_visualizer = CarlaTrajectoryVisualizer(self.world)

    # 可视化配置
    self.visualization_config = {
        'enable_3d_trajectory': True,
        'enable_curvature_visualization': True,
        'enable_tangent_visualization': True,
        'enable_speed_visualization': False,
        'enable_lookahead_visualization': True,
        'visualization_frequency': 1,  # 每帧可视化
    }


def update_trajectory_visualization(self):
    """在主循环中调用的轨迹可视化更新方法"""

    if not self.visualization_config['enable_3d_trajectory']:
        return

    # 基本轨迹可视化
    self.trajectory_visualizer.visualize_trajectory_3d(
        self.trajectory_manager,
        show_curvature=self.visualization_config['enable_curvature_visualization'],
        show_tangents=self.visualization_config['enable_tangent_visualization'],
        show_speed=self.visualization_config['enable_speed_visualization']
    )

    # 前瞻路径可视化
    if self.visualization_config['enable_lookahead_visualization']:
        self.trajectory_visualizer.visualize_lookahead_path(
            self.trajectory_manager,
            self.ego_vehicle,
            lookahead_distance=30.0
        )