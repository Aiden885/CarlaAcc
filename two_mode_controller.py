import math
from realtime_sppvt_state_manager import RealtimeSPPVTStateManager

# 核心职责:
# 1.模式选择: 基于速度阈值的TIME / SPEED模式切换
# 2.算法调用: 调用SPPVT控制器
# 3.参数管理: 两模式参数设置和状态跟踪

class TwoModeController:
    """
    ACC两模式控制器：时距控制、定速控制
    高内聚的独立模块，专门负责两模式控制逻辑
    """

    def __init__(self, V_threshold_kmh=50, G2_s=2.0, target_speed_kmh=50.0):
        """
        初始化两模式控制器

        参数:
        V_threshold_kmh: 模式切换阈值速度 (km/h)
        G2_s: 时距参数 (s)
        target_speed_kmh: 默认目标速度 (km/h)
        """
        # 两模式参数
        self.V_threshold = V_threshold_kmh / 3.6  # 转换为 m/s
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

        # SPPVT Simulink管理器
        self.sppvt_manager = RealtimeSPPVTStateManager()
        self.control_mode_flag = 2  # 默认speed模式

    def set_parameters(self, V_threshold_kmh=None, G2_s=None, target_speed_kmh=None):
        """动态设置两模式参数"""
        if V_threshold_kmh is not None:
            self.V_threshold = V_threshold_kmh / 3.6
        if G2_s is not None:
            self.G2 = G2_s
        if target_speed_kmh is not None:
            self.target_speed = target_speed_kmh / 3.6

        if self.debug:
            print(f"两模式参数更新: V_threshold={self.V_threshold * 3.6:.1f}km/h, "
                  f"G2={self.G2:.1f}s, 目标速度={self.target_speed * 3.6:.1f}km/h")

    def determine_control_mode(self, ego_speed, current_distance=None):
        """
        根据车速确定控制模式
        两模式：时距控制、定速控制
        """
        # 根据速度阈值判断控制模式
        if ego_speed <= self.V_threshold:
            return 'TIME'
        else:
            return 'SPEED'

    def calculate_desired_time_gap(self, ego_speed):
        """
        计算期望时距和实际时距

        参数:
        ego_speed: 自车速度 (m/s)

        返回:
        desired_time_gap: 期望时距 (s)
        control_mode: 控制模式字符串
        """
        if ego_speed <= self.V_threshold:
            # 时距控制模式
            desired_time_gap = self.G2  # 期望时距固定为G2秒
            control_mode = 'TIME'
        else:
            # 定速控制模式（不使用时距）
            desired_time_gap = None  # 定速模式不需要时距
            control_mode = 'SPEED'

        return desired_time_gap, control_mode

    def calculate_control_output(self, ego_speed, current_distance=None, target_speed=None):
        """
        计算两模式控制输出

        参数:
        ego_speed: 自车速度 (m/s)
        current_distance: 当前跟车距离 (m, 可选)
        target_speed: 目标速度 (m/s, 可选)

        返回:
        control_output: 控制输出 (m/s²)
        control_info: 控制信息字典
        """
        # 确定控制模式
        mode = self.determine_control_mode(ego_speed, current_distance)

        # 模式切换处理
        if mode != self.current_mode:
            if self.current_mode is not None:
                # Simulink状态管理器自动处理状态重置
                if self.debug:
                    print(f"模式切换: {self.current_mode} → {mode} at {ego_speed * 3.6:.1f}km/h")
            self.prev_mode = self.current_mode
            self.current_mode = mode

        # 根据模式计算控制
        if mode == 'TIME':
            # 时距控制模式
            if current_distance is None:
                # 没有目标时切换到速度控制
                if target_speed is None:
                    target_speed = self.target_speed

                speed_error = target_speed - ego_speed

                # 使用Simulink SPPVT速度控制
                self.control_mode_flag = 2  # speed模式
                control_output = self.sppvt_manager.run_single_step_simulation(
                    control_error=speed_error,
                    ego_speed_ms=ego_speed,
                    control_mode_flag=self.control_mode_flag,
                    control_enabled=True
                )

                control_info = {
                    'mode': f'{mode}_NO_TARGET',
                    'error': speed_error,
                    'reference': target_speed,
                    'current': ego_speed,
                    'message': f'No target detected, using speed control at {target_speed * 3.6:.1f}km/h'
                }
            else:
                # 有目标时正常时距控制 - 修改为时间控制
                desired_time_gap = self.G2  # 期望时距(秒)

                # 计算实际时距，避免除零错误
                if ego_speed > 0.1:  # 低于0.36km/h视为静止
                    actual_time_gap = current_distance / ego_speed
                else:
                    actual_time_gap = float('inf')  # 静止时设为无穷大时距

                # 计算时间误差：期望时距 - 实际时距
                time_error = desired_time_gap - actual_time_gap

                # 使用Simulink SPPVT时间控制
                self.control_mode_flag = 1  # time模式
                control_output = self.sppvt_manager.run_single_step_simulation(
                    control_error=time_error,
                    ego_speed_ms=ego_speed,
                    control_mode_flag=self.control_mode_flag,
                    control_enabled=True
                )

                control_info = {
                    'mode': mode,
                    'error': time_error,
                    'reference': desired_time_gap,
                    'current': actual_time_gap,
                    'message': f'{mode} control: 期望时距{desired_time_gap:.1f}s, 实际时距{actual_time_gap:.1f}s'
                }

        elif mode == 'SPEED':
            # 速度控制模式
            if target_speed is None:
                target_speed = self.target_speed

            speed_error = target_speed - ego_speed

            # 使用Simulink SPPVT速度控制
            self.control_mode_flag = 2  # speed模式
            control_output = self.sppvt_manager.run_single_step_simulation(
                control_error=speed_error,
                ego_speed_ms=ego_speed,
                control_mode_flag=self.control_mode_flag,
                control_enabled=True
            )

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
                'V_threshold_kmh': self.V_threshold * 3.6,
                'G2_s': self.G2,
                'target_speed_kmh': self.target_speed * 3.6
            }
        }

    def enable_debug(self, enable=True):
        """启用/禁用调试输出"""
        self.debug = enable


# 全局两模式控制器实例
_global_two_mode_controller = TwoModeController(target_speed_kmh=50.0)


def calculate_two_mode_desired_distance(ego_speed_ms):
    """
    计算两模式期望跟车距离的全局函数（兼容性函数）
    注意：内部已改为时距控制，但保持接口兼容

    参数:
    ego_speed_ms: 自车速度 (m/s)

    返回:
    desired_distance: 期望跟车距离 (m) - 根据时距计算得出
    control_mode: 控制模式字符串
    """
    desired_time_gap, control_mode = _global_two_mode_controller.calculate_desired_time_gap(ego_speed_ms)
    if desired_time_gap is not None:
        # 将时距转换为距离以保持接口兼容
        desired_distance = ego_speed_ms * desired_time_gap
    else:
        # SPEED模式，使用阈值速度计算参考距离
        desired_distance = _global_two_mode_controller.V_threshold * _global_two_mode_controller.G2

    return desired_distance, control_mode


def two_mode_control(ego_speed_ms, current_distance=None, target_speed_ms=None):
    """
    两模式控制的全局函数

    参数:
    ego_speed_ms: 自车速度 (m/s)
    current_distance: 当前跟车距离 (m, 可选)
    target_speed_ms: 目标速度 (m/s, 可选)

    返回:
    control_output: 控制输出 (m/s²)
    control_info: 控制信息字典
    """
    return _global_two_mode_controller.calculate_control_output(
        ego_speed_ms, current_distance, target_speed_ms)


def set_two_mode_parameters(V_threshold_kmh=50, G2_s=2.0, target_speed_kmh=50.0):
    """设置全局两模式参数"""
    _global_two_mode_controller.set_parameters(V_threshold_kmh, G2_s, target_speed_kmh)


def enable_two_mode_debug(enable=True):
    """启用两模式调试"""
    _global_two_mode_controller.enable_debug(enable)


def get_two_mode_status():
    """获取两模式状态"""
    return _global_two_mode_controller.get_status()

def target_redetection_safety_control(ego_speed, current_distance, target_speed, force_mode=None):
        """
        前车重检测安全控制
        当前车丢失后重新检测到时，提供强制安全控制策略

        Args:
            ego_speed: 当前速度 (m/s)
            current_distance: 前车距离 (m)，None表示无前车
            target_speed: 目标速度 (m/s)
            force_mode: 强制模式 "time_gap"/None

        Returns:
            (accel, control_info): 加速度和控制信息
        """

        if force_mode is None:
            # 正常模式：调用两模式控制
            return two_mode_control(ego_speed, current_distance, target_speed)

        if current_distance is None:
            # 无前车时不使用强制模式
            return two_mode_control(ego_speed, None, target_speed)

        # 获取当前两模式参数
        try:
            params = get_two_mode_status()
            V_threshold_ms = params['parameters']['V_threshold_kmh'] / 3.6
            G2_s = params['parameters']['G2_s']
        except:
            # 如果获取参数失败，使用默认值
            V_threshold_ms = 50.0 / 3.6
            G2_s = 2.0

        if force_mode == "time_gap":
            # 强制时距控制
            desired_distance = ego_speed * G2_s

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
            return two_mode_control(ego_speed, current_distance, target_speed)

        # 限制加速度范围
        accel = max(-4.0, min(2.0, accel))

        return accel, control_info

def get_safety_control_mode_recommendation(ego_speed, current_distance, speed_increase_threshold=2.8):
        """
        根据当前状态推荐安全控制模式类型
        用于前车重检测时的强制控制策略

        Args:
            ego_speed: 当前速度 (m/s)
            current_distance: 前车距离 (m)
            speed_increase_threshold: 速度增加阈值 (m/s)

        Returns:
            推荐的安全控制模式: "time_gap"/None
        """

        if current_distance is None:
            return None

        try:
            params = get_two_mode_status()
            G2_s = params['parameters']['G2_s']
        except:
            G2_s = 2.0

        # 计算期望的时距距离
        desired_time_gap_distance = ego_speed * G2_s

        # 如果距离在时距范围内，推荐时距控制
        if current_distance < desired_time_gap_distance * 1.3:  # 给一点缓冲
            return "time_gap"

        # 距离较远，可能不需要强制控制
        else:
            return "time_gap"  # 保守起见，还是用时距控制


def enhanced_two_mode_control(ego_speed_ms, current_distance=None, target_speed_ms=None):
    """
    增强的两模式控制函数，为Simulink一体化接口提供标准化输出
    
    参数:
    ego_speed_ms: 自车速度 (m/s)
    current_distance: 当前跟车距离 (m, 可选)
    target_speed_ms: 目标速度 (m/s, 可选)
    
    返回:
    enhanced_output: 增强的控制信息字典，包含:
        - control_error: 控制误差 (float)
        - control_mode_flag: 控制模式标志 (int, 1=distance, 2=speed)
        - mode_description: 模式描述 (str)
        - reference_value: 参考值 (float)
        - current_value: 当前值 (float)
        - desired_distance: 期望距离 (float, 仅在时距模式下有效)
        - target_speed: 实际使用的目标速度 (float)
    """
    # 调用基础两模式控制
    control_output, basic_info = two_mode_control(ego_speed_ms, current_distance, target_speed_ms)
    
    # 获取当前两模式参数
    try:
        params = get_two_mode_status()
        V_threshold_ms = params['parameters']['V_threshold_kmh'] / 3.6
        G2_s = params['parameters']['G2_s']
        default_target_speed = params['parameters']['target_speed_kmh'] / 3.6
    except:
        # 使用默认参数
        V_threshold_ms = 50.0 / 3.6
        G2_s = 2.0
        default_target_speed = 50.0 / 3.6
    
    # 确定实际使用的目标速度
    if target_speed_ms is None:
        target_speed_ms = default_target_speed
    
    # 标准化输出
    enhanced_output = {
        'control_output': control_output,  # 基础控制输出 (m/s²)
        'control_error': basic_info['error'],  # 控制误差
        'reference_value': basic_info['reference'],  # 参考值
        'current_value': basic_info['current'],  # 当前值
        'target_speed': target_speed_ms,  # 实际使用的目标速度
        'mode_description': basic_info['mode'],  # 模式描述
    }
    
    # 根据模式设置标志和计算期望距离
    if 'TIME' in basic_info['mode']:
        enhanced_output['control_mode_flag'] = 1  # 时距模式
        enhanced_output['desired_distance'] = ego_speed_ms * G2_s
    elif 'SPEED' in basic_info['mode']:
        enhanced_output['control_mode_flag'] = 2  # 速度模式
        enhanced_output['desired_distance'] = V_threshold_ms * G2_s  # 参考距离
    else:
        enhanced_output['control_mode_flag'] = 2  # 默认速度模式
        enhanced_output['desired_distance'] = V_threshold_ms * G2_s
    
    # 添加调试信息
    enhanced_output['debug_info'] = {
        'ego_speed_kmh': ego_speed_ms * 3.6,
        'V_threshold_kmh': V_threshold_ms * 3.6,
        'G2_s': G2_s,
        'has_target': current_distance is not None,
        'current_distance': current_distance,
        'basic_message': basic_info['message']
    }
    
    return enhanced_output
