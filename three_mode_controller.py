import math
from sppvt_longitudinal_control import sppvt_longitudinal_control, set_sppvt_parameters, reset_sppvt_controller


class ThreeModeController:
    """
    ACC三模式控制器：车距控制、时距控制、定速控制
    高内聚的独立模块，专门负责三模式控制逻辑
    """

    def __init__(self, V1_kmh=20, V2_kmh=30, V3_kmh=50, G1_m=5.0, G2_s=2.0, target_speed_kmh=30):
        """
        初始化三模式控制器

        参数:
        V1_kmh: 最低车速 (km/h)
        V2_kmh: 临界车速 (km/h)
        V3_kmh: 最高车速 (km/h)
        G1_m: 最小安全车距 (m)
        G2_s: 最小安全时距 (s)
        target_speed_kmh: 默认目标速度 (km/h)
        """
        # 三模式参数
        self.V1 = V1_kmh / 3.6  # 转换为 m/s
        self.V2 = V2_kmh / 3.6
        self.V3 = V3_kmh / 3.6
        self.G1 = G1_m
        self.G2 = G2_s

        # 默认目标速度
        self.target_speed = target_speed_kmh / 3.6

        # 当前状态
        self.current_mode = None
        self.prev_mode = None

        # 平滑过渡参数
        self.transition_zone = 2.8 / 3.6  # 2.8km/h过渡区间

        # 调试标志
        self.debug = False

    def set_parameters(self, V1_kmh=None, V2_kmh=None, V3_kmh=None, G1_m=None, G2_s=None, target_speed_kmh=None):
        """动态设置三模式参数"""
        if V1_kmh is not None:
            self.V1 = V1_kmh / 3.6
        if V2_kmh is not None:
            self.V2 = V2_kmh / 3.6
        if V3_kmh is not None:
            self.V3 = V3_kmh / 3.6
        if G1_m is not None:
            self.G1 = G1_m
        if G2_s is not None:
            self.G2 = G2_s
        if target_speed_kmh is not None:
            self.target_speed = target_speed_kmh / 3.6

        if self.debug:
            print(f"三模式参数更新: V1={self.V1 * 3.6:.1f}km/h, V2={self.V2 * 3.6:.1f}km/h, "
                  f"V3={self.V3 * 3.6:.1f}km/h, G1={self.G1:.1f}m, G2={self.G2:.1f}s, "
                  f"目标速度={self.target_speed * 3.6:.1f}km/h")

    def determine_control_mode(self, ego_speed):
        """
        根据车速确定控制模式

        参数:
        ego_speed: 自车速度 (m/s)

        返回:
        mode: 'DISABLED', 'DISTANCE', 'TIME', 'SPEED'
        """
        if ego_speed < self.V1:
            return 'DISABLED'
        elif ego_speed < self.V2:
            return 'DISTANCE'
        elif ego_speed <= self.V3:
            return 'TIME'
        else:
            return 'SPEED'

    def calculate_desired_distance(self, ego_speed):
        """
        计算期望跟车距离（平滑过渡版本）

        参数:
        ego_speed: 自车速度 (m/s)

        返回:
        desired_distance: 期望跟车距离 (m)
        control_mode: 控制模式字符串
        """
        if ego_speed < self.V1:
            return 0.0, 'DISABLED'

        # 车距到时距的平滑过渡
        if ego_speed < (self.V2 - self.transition_zone):
            # 纯车距控制
            desired_distance = self.G1
            control_mode = 'DISTANCE'

        elif ego_speed < (self.V2 + self.transition_zone):
            # 过渡区间：线性插值
            weight = (ego_speed - (self.V2 - self.transition_zone)) / (2 * self.transition_zone)
            weight = max(0.0, min(1.0, weight))

            distance_ref = self.G1
            time_ref = ego_speed * self.G2
            desired_distance = distance_ref * (1 - weight) + time_ref * weight
            control_mode = f'DISTANCE→TIME({weight:.2f})'

        elif ego_speed <= self.V3:
            # 纯时距控制
            desired_distance = ego_speed * self.G2
            control_mode = 'TIME'

        else:
            # 定速控制模式（参考距离）
            desired_distance = self.V3 * self.G2
            control_mode = 'SPEED'

        return desired_distance, control_mode

    def calculate_control_output(self, ego_speed, current_distance=None, target_speed=None):
        """
        计算三模式控制输出

        参数:
        ego_speed: 自车速度 (m/s)
        current_distance: 当前跟车距离 (m, 可选)
        target_speed: 目标速度 (m/s, 可选，默认使用V3)

        返回:
        control_output: 控制输出 (m/s²)
        control_info: 控制信息字典
        """
        # 确定控制模式
        mode = self.determine_control_mode(ego_speed)

        # 模式切换处理
        if mode != self.current_mode:
            if self.current_mode is not None:
                reset_sppvt_controller()  # 重置SPPVT状态
                if self.debug:
                    print(f"模式切换: {self.current_mode} → {mode} at {ego_speed * 3.6:.1f}km/h")
            self.prev_mode = self.current_mode
            self.current_mode = mode

        # 根据模式计算控制
        if mode == 'DISABLED':
            control_output = 0.0
            control_info = {
                'mode': mode,
                'error': 0.0,
                'reference': 0.0,
                'message': 'Speed too low for ACC'
            }

        elif mode in ['DISTANCE', 'TIME'] or 'DISTANCE→TIME' in mode:
            # 位置控制模式
            if current_distance is None:
                # === 修改：没有目标时切换到速度控制 ===
                if target_speed is None:
                    target_speed = self.target_speed if hasattr(self, 'target_speed') else self.V3

                speed_error = target_speed - ego_speed

                # 使用SPPVT速度控制
                set_sppvt_parameters(control_mode='speed')
                control_output = sppvt_longitudinal_control(speed_error, 0.05)

                control_info = {
                    'mode': f'{mode}_NO_TARGET',
                    'error': speed_error,
                    'reference': target_speed,
                    'current': ego_speed,
                    'message': f'No target detected, using speed control at {target_speed * 3.6:.1f}km/h'
                }
            else:
                # 有目标时正常距离控制
                desired_distance, _ = self.calculate_desired_distance(ego_speed)
                distance_error = desired_distance - current_distance

                # 使用SPPVT距离控制
                set_sppvt_parameters(control_mode='distance')
                control_output = sppvt_longitudinal_control(- distance_error, 0.05)

                control_info = {
                    'mode': mode,
                    'error': distance_error,
                    'reference': desired_distance,
                    'current': current_distance,
                    'message': f'{mode} control active'
                }

        elif mode == 'SPEED':
            # 速度控制模式
            if target_speed is None:
                target_speed = self.V3  # 默认限制在最高速度

            speed_error = target_speed - ego_speed

            # 使用SPPVT速度控制
            set_sppvt_parameters(control_mode='speed')
            control_output = sppvt_longitudinal_control(speed_error, 0.05)

            control_info = {
                'mode': mode,
                'error': speed_error,
                'reference': target_speed,
                'current': ego_speed,
                'message': f'Speed control: limiting to {target_speed * 3.6:.1f}km/h'
            }

        else:
            # 未知模式
            control_output = 0.0
            control_info = {
                'mode': 'UNKNOWN',
                'error': 0.0,
                'reference': 0.0,
                'message': 'Unknown control mode'
            }

        return control_output, control_info

    def get_status(self):
        """获取控制器状态信息"""
        return {
            'current_mode': self.current_mode,
            'prev_mode': self.prev_mode,
            'parameters': {
                'V1_kmh': self.V1 * 3.6,
                'V2_kmh': self.V2 * 3.6,
                'V3_kmh': self.V3 * 3.6,
                'G1_m': self.G1,
                'G2_s': self.G2
            }
        }

    def enable_debug(self, enable=True):
        """启用/禁用调试输出"""
        self.debug = enable


# 全局三模式控制器实例
_global_three_mode_controller = ThreeModeController(target_speed_kmh=30.0)


def calculate_three_mode_desired_distance(ego_speed_ms):
    """
    计算三模式期望跟车距离的全局函数

    参数:
    ego_speed_ms: 自车速度 (m/s)

    返回:
    desired_distance: 期望跟车距离 (m)
    control_mode: 控制模式字符串
    """
    return _global_three_mode_controller.calculate_desired_distance(ego_speed_ms)


def three_mode_control(ego_speed_ms, current_distance=None, target_speed_ms=None):
    """
    三模式控制的全局函数

    参数:
    ego_speed_ms: 自车速度 (m/s)
    current_distance: 当前跟车距离 (m, 可选)
    target_speed_ms: 目标速度 (m/s, 可选)

    返回:
    control_output: 控制输出 (m/s²)
    control_info: 控制信息字典
    """
    return _global_three_mode_controller.calculate_control_output(
        ego_speed_ms, current_distance, target_speed_ms)


def set_three_mode_parameters(V1_kmh=20, V2_kmh=40, V3_kmh=80, G1_m=5.0, G2_s=2.0, target_speed_kmh=30.0):
    """设置全局三模式参数"""
    _global_three_mode_controller.set_parameters(V1_kmh, V2_kmh, V3_kmh, G1_m, G2_s, target_speed_kmh)


def enable_three_mode_debug(enable=True):
    """启用三模式调试"""
    _global_three_mode_controller.enable_debug(enable)


def get_three_mode_status():
    """获取三模式状态"""
    return _global_three_mode_controller.get_status()