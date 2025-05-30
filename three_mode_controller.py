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

    def determine_control_mode(self, ego_speed, current_distance=None):  # 新增距离参数
        """
        根据车速和距离确定控制模式
        距离优先：当距离过近时，强制进入距离/时距控制
        """
        # === 新增：距离优先逻辑 ===
        if current_distance is not None:
            # 计算期望距离
            desired_distance_for_time = ego_speed * self.G2  # 时距控制的期望距离

            # 如果距离过近，强制进入距离控制（不管速度多高）
            if current_distance < self.G1:
                return 'DISTANCE'
            # 如果距离小于时距要求，强制进入时距控制
            elif current_distance < desired_distance_for_time:
                return 'TIME'

        # === 原有的速度判断逻辑 ===
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
        mode = self.determine_control_mode(ego_speed, current_distance)

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


def set_three_mode_parameters(V1_kmh=0, V2_kmh=30, V3_kmh=50, G1_m=5.0, G2_s=2.0, target_speed_kmh=30.0):
    """设置全局三模式参数"""
    _global_three_mode_controller.set_parameters(V1_kmh, V2_kmh, V3_kmh, G1_m, G2_s, target_speed_kmh)


def enable_three_mode_debug(enable=True):
    """启用三模式调试"""
    _global_three_mode_controller.enable_debug(enable)


def get_three_mode_status():
    """获取三模式状态"""
    return _global_three_mode_controller.get_status()

def three_mode_control_with_force_mode(ego_speed, current_distance, target_speed, force_mode=None):
        """
        带强制模式的三模式控制

        Args:
            ego_speed: 当前速度 (m/s)
            current_distance: 前车距离 (m)，None表示无前车
            target_speed: 目标速度 (m/s)
            force_mode: 强制模式 "distance"/"time_gap"/None

        Returns:
            (accel, control_info): 加速度和控制信息
        """

        if force_mode is None:
            # 正常模式：调用原有的three_mode_control
            return three_mode_control(ego_speed, current_distance, target_speed)

        if current_distance is None:
            # 无前车时不使用强制模式
            return three_mode_control(ego_speed, None, target_speed)

        # 获取当前三模式参数
        try:
            params = get_three_mode_status()
            V1_ms = params['V1_kmh'] / 3.6
            V2_ms = params['V2_kmh'] / 3.6
            V3_ms = params['V3_kmh'] / 3.6
            G1_m = params['G1_m']
            G2_s = params['G2_s']
        except:
            # 如果获取参数失败，使用默认值
            V1_ms = 10.0 / 3.6
            V2_ms = 30.0 / 3.6
            V3_ms = 50.0 / 3.6
            G1_m = 15.0
            G2_s = 2.0

        if force_mode == "distance":
            # 强制距离控制（第1阶段）
            # 检查是否满足距离控制条件
            if current_distance < G1_m:
                # 距离过近，需要减速
                distance_error = G1_m - current_distance
                # 简单的比例控制
                accel = -min(3.0, distance_error * 0.5)  # 最大减速3 m/s²

                control_info = {
                    'mode': 'FORCE_DISTANCE_CONTROL',
                    'message': f'强制距离控制: 距离{current_distance:.1f}m < G1({G1_m:.1f}m), 减速{abs(accel):.1f}m/s²',
                    'stage': 1,
                    'target_distance': G1_m,
                    'current_distance': current_distance
                }
            else:
                # 距离足够，轻微减速或保持
                accel = -0.5  # 轻微减速
                control_info = {
                    'mode': 'FORCE_DISTANCE_MAINTAIN',
                    'message': f'强制距离控制: 距离{current_distance:.1f}m >= G1({G1_m:.1f}m), 保持控制',
                    'stage': 1,
                    'target_distance': G1_m,
                    'current_distance': current_distance
                }

        elif force_mode == "time_gap":
            # 强制时距控制（第2阶段）
            desired_distance = ego_speed * G2_s + G1_m

            if current_distance < desired_distance:
                # 时距不足，需要减速
                distance_error = desired_distance - current_distance
                accel = -min(2.5, distance_error * 0.3)  # 最大减速2.5 m/s²

                control_info = {
                    'mode': 'FORCE_TIME_GAP_CONTROL',
                    'message': f'强制时距控制: 距离{current_distance:.1f}m < 需求{desired_distance:.1f}m, 减速{abs(accel):.1f}m/s²',
                    'stage': 2,
                    'desired_distance': desired_distance,
                    'current_distance': current_distance,
                    'time_gap': G2_s
                }
            else:
                # 时距足够，轻微减速或保持
                accel = -0.3  # 轻微减速
                control_info = {
                    'mode': 'FORCE_TIME_GAP_MAINTAIN',
                    'message': f'强制时距控制: 距离{current_distance:.1f}m >= 需求{desired_distance:.1f}m, 保持控制',
                    'stage': 2,
                    'desired_distance': desired_distance,
                    'current_distance': current_distance,
                    'time_gap': G2_s
                }

        else:
            # 未知的强制模式，回退到正常模式
            return three_mode_control(ego_speed, current_distance, target_speed)

        # 限制加速度范围
        accel = max(-4.0, min(2.0, accel))

        return accel, control_info

def get_force_mode_recommendation(ego_speed, current_distance, speed_increase_threshold=2.8):
        """
        根据当前状态推荐强制模式类型

        Args:
            ego_speed: 当前速度 (m/s)
            current_distance: 前车距离 (m)
            speed_increase_threshold: 速度增加阈值 (m/s)

        Returns:
            推荐的强制模式: "distance"/"time_gap"/None
        """

        if current_distance is None:
            return None

        try:
            params = get_three_mode_status()
            G1_m = params['G1_m']
            G2_s = params['G2_s']
        except:
            G1_m = 15.0
            G2_s = 2.0

        # 计算期望的时距距离
        desired_time_gap_distance = ego_speed * G2_s + G1_m

        # 如果距离很近（小于G1），推荐距离控制
        if current_distance < G1_m * 1.2:  # 给一点缓冲
            return "distance"

        # 如果距离在时距范围内，推荐时距控制
        elif current_distance < desired_time_gap_distance * 1.3:  # 给一点缓冲
            return "time_gap"

        # 距离较远，可能不需要强制控制
        else:
            return "time_gap"  # 保守起见，还是用时距控制