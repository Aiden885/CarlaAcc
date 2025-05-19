import numpy as np
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import threading
import time
from scipy.interpolate import splprep, splev
import warnings

warnings.filterwarnings('ignore')


@dataclass
class TrajectoryWaypoint:
    """类似CARLA waypoint的轨迹点"""
    # 世界坐标系位置
    world_x: float
    world_y: float
    world_z: float = 0.0

    # 相对坐标系位置（相对于自车）
    relative_x: float = 0.0
    relative_y: float = 0.0
    relative_z: float = 0.0

    # 几何信息
    tangent_vector: Tuple[float, float] = (1.0, 0.0)  # 切线方向向量
    tangent_angle: float = 0.0  # 切线角度（弧度）
    curvature: float = 0.0  # 曲率

    # 运动信息
    speed: float = 0.0  # 期望速度
    acceleration: float = 0.0  # 期望加速度

    # 元数据
    timestamp: float = 0.0
    distance_from_start: float = 0.0  # 从轨迹起点的距离
    parameter: float = 0.0  # B样条参数值 [0,1]

    def get_transform_matrix(self) -> np.ndarray:
        """获取该点的变换矩阵（类似CARLA Transform）"""
        cos_angle = math.cos(self.tangent_angle)
        sin_angle = math.sin(self.tangent_angle)

        transform = np.array([
            [cos_angle, -sin_angle, 0, self.world_x],
            [sin_angle, cos_angle, 0, self.world_y],
            [0, 0, 1, self.world_z],
            [0, 0, 0, 1]
        ])
        return transform

    def distance_to(self, other) -> float:
        """计算到另一个轨迹点的距离"""
        dx = self.world_x - other.world_x
        dy = self.world_y - other.world_y
        return math.sqrt(dx * dx + dy * dy)


class RealTimeTrajectoryBuffer:
    """线程安全的实时轨迹缓冲区"""

    def __init__(self, max_length: float = 50.0, spatial_resolution: float = 0.3):
        self.max_length = max_length
        self.spatial_resolution = spatial_resolution
        self.max_points = int(max_length / spatial_resolution)

        # 线程安全保护
        self._lock = threading.RLock()

        # 轨迹数据
        self._waypoints = deque(maxlen=self.max_points)
        self._spline_params = None
        self._last_update_time = 0.0
        self._version = 0

        # 缓存
        self._cached_arrays = None
        self._cache_version = -1

    def update_trajectory(self, waypoints: List[TrajectoryWaypoint]):
        """更新轨迹数据"""
        with self._lock:
            self._waypoints.clear()
            self._waypoints.extend(waypoints)
            self._last_update_time = time.time()
            self._version += 1
            self._cached_arrays = None

    def get_waypoints(self, max_count: Optional[int] = None) -> List[TrajectoryWaypoint]:
        """获取轨迹点列表"""
        with self._lock:
            waypoints = list(self._waypoints)
            if max_count and len(waypoints) > max_count:
                waypoints = waypoints[:max_count]
            return waypoints

    def get_waypoints_in_distance(self, max_distance: float) -> List[TrajectoryWaypoint]:
        """获取指定距离内的轨迹点"""
        with self._lock:
            waypoints = []
            for wp in self._waypoints:
                if wp.distance_from_start <= max_distance:
                    waypoints.append(wp)
                else:
                    break
            return waypoints

    def get_nearest_waypoint(self, x: float, y: float) -> Optional[TrajectoryWaypoint]:
        """获取最近的轨迹点"""
        with self._lock:
            if not self._waypoints:
                return None

            min_distance = float('inf')
            nearest_wp = None

            for wp in self._waypoints:
                distance = math.sqrt((wp.world_x - x) ** 2 + (wp.world_y - y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_wp = wp

            return nearest_wp

    def is_valid(self, max_age: float = 1.0) -> bool:
        """检查轨迹是否有效"""
        with self._lock:
            return (time.time() - self._last_update_time) < max_age and len(self._waypoints) > 0

    def get_info(self) -> Dict:
        """获取轨迹信息"""
        with self._lock:
            total_length = self._waypoints[-1].distance_from_start if self._waypoints else 0.0
            return {
                'waypoint_count': len(self._waypoints),
                'total_length': total_length,
                'last_update': self._last_update_time,
                'version': self._version,
                'is_valid': self.is_valid(),
                'spatial_resolution': self.spatial_resolution
            }

    def __len__(self):
        with self._lock:
            return len(self._waypoints)

    def __bool__(self):
        return self.is_valid()


class BSplineTrajectoryManager:
    """基于B样条的实时轨迹管理器"""

    def __init__(self, trajectory_buffer: RealTimeTrajectoryBuffer,
                 min_points_for_fit: int = 5, fit_interval: int = 3):
        self.buffer = trajectory_buffer
        self.min_points_for_fit = min_points_for_fit
        self.fit_interval = fit_interval

        # 原始数据历史
        self.raw_points_history = deque(maxlen=20)
        self.frame_count = 0

        # B样条参数
        self.current_spline = None
        self.last_fit_time = 0.0

        # 平滑参数
        self.smoothing_factor = 0.0  # B样条平滑因子
        self.derivative_smoothing = True  # 是否平滑导数计算

    def add_target_info_realtime(self, target_info, ego_vehicle) -> Optional[TrajectoryWaypoint]:
        """添加目标信息并更新轨迹（保持原接口）"""
        if target_info is None or len(target_info) < 9:
            return None

        try:
            # 1. 提取相对位置
            relative_pos = {
                'x': float(target_info[0]),
                'y': float(target_info[1]),
                'z': float(target_info[2])
            }

            # 2. 获取自车状态
            ego_location = ego_vehicle.get_location()
            ego_rotation = ego_vehicle.get_transform().rotation
            ego_pose = {
                'x': float(ego_location.x),
                'y': float(ego_location.y),
                'yaw': math.radians(ego_rotation.yaw)
            }

            # 3. 转换到世界坐标
            world_pos = self._transform_to_world(relative_pos, ego_pose)

            # 4. 添加到历史记录
            point_data = {
                'world': world_pos,
                'relative': relative_pos,
                'ego_pose': ego_pose,
                'timestamp': time.time(),
                'speed': math.sqrt(target_info[6] ** 2 + target_info[7] ** 2) * 3.6  # 转换为km/h
            }
            self.raw_points_history.append(point_data)

            # 5. 定期更新B样条拟合
            self.frame_count += 1
            if self.frame_count % self.fit_interval == 0:
                self._update_bspline_trajectory()

            # 6. 返回最新的轨迹点（如果有）
            waypoints = self.buffer.get_waypoints(max_count=1)
            return waypoints[0] if waypoints else None

        except Exception as e:
            print(f"Error in add_target_info_realtime: {e}")
            return None

    def _transform_to_world(self, relative_pos: Dict, ego_pose: Dict) -> Dict:
        """坐标变换：相对坐标 -> 世界坐标"""
        rel_x, rel_y = relative_pos['x'], relative_pos['y']
        ego_x, ego_y, ego_yaw = ego_pose['x'], ego_pose['y'], ego_pose['yaw']

        # 旋转变换
        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)

        world_x = ego_x + rel_x * cos_yaw - rel_y * sin_yaw
        world_y = ego_y + rel_x * sin_yaw + rel_y * cos_yaw

        return {
            'x': world_x,
            'y': world_y,
            'z': relative_pos['z']
        }

    def _update_bspline_trajectory(self):
        """使用B样条拟合更新轨迹"""
        # 确保有足够的数据点
        if len(self.raw_points_history) < max(self.min_points_for_fit, 4):
            print("警告：数据点不足，无法进行B样条拟合")
            return

        try:
            # 1. 准备拟合数据
            x_coords = [p['world']['x'] for p in self.raw_points_history]
            y_coords = [p['world']['y'] for p in self.raw_points_history]

            # 检查坐标长度和有效性
            if len(x_coords) != len(y_coords):
                print("错误：x和y坐标长度不匹配")
                return
            coords = np.array([x_coords, y_coords]).T
            if not np.all(np.isfinite(coords)):
                print("警告：检测到无效坐标，跳过拟合")
                return

            # 2. B样条拟合
            num_points = len(x_coords)
            max_degree = min(3, num_points - 1)  # 样条次数不超过点数-1
            result = splprep([x_coords, y_coords],
                             s=self.smoothing_factor or 0.0,  # 默认平滑因子
                             k=max_degree)

            # 调试：检查返回值
            if len(result) != 2:
                print(f"错误：splprep 应返回2个值，实际返回 {len(result)} 个")
                return
            tck, u = result

            self.current_spline = tck
            self.last_fit_time = time.time()

            # 3. 生成插值轨迹点
            waypoints = self._generate_interpolated_waypoints(tck)

            # 4. 更新缓冲区
            self.buffer.update_trajectory(waypoints)

        except ValueError as e:
            print(f"B样条拟合错误：{e}")
        except Exception as e:
            print(f"意外错误：{e}")

    def _generate_interpolated_waypoints(self, tck) -> List[TrajectoryWaypoint]:
        """生成固定空间间隔的插值轨迹点"""
        # 1. 计算样条总长度
        u_dense = np.linspace(0, 1, 1000)
        x_dense, y_dense = splev(u_dense, tck)

        # 计算累积长度
        dx = np.diff(x_dense)
        dy = np.diff(y_dense)
        segment_lengths = np.sqrt(dx * dx + dy * dy)
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]

        if total_length < self.buffer.spatial_resolution:
            return []

        # 2. 按固定空间间隔采样
        num_points = int(total_length / self.buffer.spatial_resolution) + 1
        target_distances = np.linspace(0, total_length, num_points)

        # 插值得到对应的参数值
        u_interp = np.interp(target_distances, cumulative_lengths, u_dense)

        # 3. 计算插值点的位置和导数
        x_interp, y_interp = splev(u_interp, tck)
        dx_interp, dy_interp = splev(u_interp, tck, der=1)

        # 平滑导数（可选）
        if self.derivative_smoothing and len(dx_interp) > 5:
            # 对导数进行轻微平滑，减少抖动
            window_size = min(5, len(dx_interp) // 2 * 2 + 1)  # 确保奇数
            if window_size >= 3:
                from scipy.signal import savgol_filter
                dx_interp = savgol_filter(dx_interp, window_size, 2)
                dy_interp = savgol_filter(dy_interp, window_size, 2)

        # 4. 计算二阶导数用于曲率计算
        ddx_interp, ddy_interp = splev(u_interp, tck, der=2)

        # 5. 生成轨迹点
        waypoints = []
        latest_ego_pose = self.raw_points_history[-1]['ego_pose'] if self.raw_points_history else None
        avg_speed = np.mean([p['speed'] for p in self.raw_points_history]) if self.raw_points_history else 0.0

        for i in range(len(x_interp)):
            # 计算切线角度
            tangent_angle = math.atan2(dy_interp[i], dx_interp[i])

            # 计算曲率
            speed_squared = dx_interp[i] ** 2 + dy_interp[i] ** 2
            if speed_squared > 1e-8:
                curvature = abs(dx_interp[i] * ddy_interp[i] - dy_interp[i] * ddx_interp[i]) / (speed_squared ** 1.5)
            else:
                curvature = 0.0

            # 转换到相对坐标系（如果有自车位姿信息）
            relative_x, relative_y = 0.0, 0.0
            if latest_ego_pose:
                relative_x, relative_y = self._world_to_relative(
                    x_interp[i], y_interp[i], latest_ego_pose
                )

            # 创建轨迹点
            waypoint = TrajectoryWaypoint(
                world_x=float(x_interp[i]),
                world_y=float(y_interp[i]),
                world_z=0.0,
                relative_x=relative_x,
                relative_y=relative_y,
                relative_z=0.0,
                tangent_vector=(float(dx_interp[i]), float(dy_interp[i])),
                tangent_angle=float(tangent_angle),
                curvature=float(curvature),
                speed=float(avg_speed),
                timestamp=time.time(),
                distance_from_start=float(target_distances[i]),
                parameter=float(u_interp[i])
            )
            waypoints.append(waypoint)

        return waypoints

    def _world_to_relative(self, world_x: float, world_y: float, ego_pose: Dict) -> Tuple[float, float]:
        """世界坐标转相对坐标"""
        ego_x, ego_y, ego_yaw = ego_pose['x'], ego_pose['y'], ego_pose['yaw']

        # 平移
        dx = world_x - ego_x
        dy = world_y - ego_y

        # 逆旋转
        cos_yaw = math.cos(-ego_yaw)
        sin_yaw = math.sin(-ego_yaw)

        rel_x = dx * cos_yaw - dy * sin_yaw
        rel_y = dx * sin_yaw + dy * cos_yaw

        return rel_x, rel_y

    # 保持向后兼容的方法
    def get_recent_trajectory_fast(self, duration: float = 5.0, max_points: int = 50) -> List[Dict]:
        """获取最近轨迹（兼容原接口）"""
        waypoints = self.buffer.get_waypoints(max_count=max_points)
        current_time = time.time()

        # 转换为原接口格式
        trajectory_points = []
        for wp in waypoints:
            if current_time - wp.timestamp <= duration:
                trajectory_points.append({
                    'timestamp': wp.timestamp,
                    'world_position': {
                        'x': wp.world_x,
                        'y': wp.world_y,
                        'z': wp.world_z
                    }
                })

        return trajectory_points

    def get_performance_stats(self) -> Dict:
        """获取性能统计（兼容原接口）"""
        info = self.buffer.get_info()
        return {
            'total_points': info['waypoint_count'],
            'recent_points_count': len(self.buffer.get_waypoints(max_count=50)),
            'avg_processing_time_ms': 1.0,  # 简化
            'trajectory_length': info['total_length']
        }

    # 新增的便捷方法
    def get_control_waypoints(self, lookahead_distance: float = 30.0) -> List[TrajectoryWaypoint]:
        """获取控制用的前瞻轨迹点"""
        return self.buffer.get_waypoints_in_distance(lookahead_distance)

    def get_trajectory_curvature_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取轨迹的曲率分布（距离 vs 曲率）"""
        waypoints = self.buffer.get_waypoints()
        if not waypoints:
            return np.array([]), np.array([])

        distances = [wp.distance_from_start for wp in waypoints]
        curvatures = [wp.curvature for wp in waypoints]

        return np.array(distances), np.array(curvatures)

    def get_waypoint_at_distance(self, distance: float) -> Optional[TrajectoryWaypoint]:
        """获取指定距离处的轨迹点（插值计算）"""
        waypoints = self.buffer.get_waypoints()
        if not waypoints:
            return None

        # 找到距离最近的两个点进行插值
        for i in range(len(waypoints) - 1):
            if waypoints[i].distance_from_start <= distance <= waypoints[i + 1].distance_from_start:
                # 线性插值
                wp1, wp2 = waypoints[i], waypoints[i + 1]
                t = (distance - wp1.distance_from_start) / (wp2.distance_from_start - wp1.distance_from_start)

                return TrajectoryWaypoint(
                    world_x=wp1.world_x + t * (wp2.world_x - wp1.world_x),
                    world_y=wp1.world_y + t * (wp2.world_y - wp1.world_y),
                    world_z=wp1.world_z + t * (wp2.world_z - wp1.world_z),
                    tangent_angle=wp1.tangent_angle + t * (wp2.tangent_angle - wp1.tangent_angle),
                    curvature=wp1.curvature + t * (wp2.curvature - wp1.curvature),
                    speed=wp1.speed + t * (wp2.speed - wp1.speed),
                    distance_from_start=distance,
                    timestamp=time.time()
                )

        return None


# 向后兼容的别名
RealTimeTrajectoryManager = BSplineTrajectoryManager