import numpy as np
import time


class SPPVTLongitudinalController:
    """
    SPPVT纵向控制器
    与PID控制器接口完全一致，可直接替换
    支持距离跟踪和速度跟踪两种模式
    """

    def __init__(self, control_mode='distance'):
        # SPPVT控制参数
        self.sppvt_kp = 1.0  # 比例控制系数
        self.sppvt_delta = 0.05  # 变目标阈值（接近0的正值）
        self.sppvt_eta = 0.2  # 定速控制精度
        self.sppvt_rho = 0.25  # 惩罚系数 (0 < ρ < 0.5)

        # 控制模式：'distance' 或 'speed'
        self.control_mode = control_mode

        # 输出限制
        self.max_accel = 2.0
        self.max_decel = -3.0

        # 控制状态
        self.stage = 1  # 当前阶式级数 i
        self.target_stages = {}  # 各级目标值 V_i

        # 历史状态（用于计算导数）
        self.prev_error = 0.0  # 上次误差值
        self.prev_velocity = 0.0  # 上次速度（一阶导数）
        self.prev_accel = 0.0  # 上次加速度（二阶导数）

        # 升级历史
        self.upgrade_count = 0
        self.last_upgrade_time = 0

        # 调试标志
        self.debug = False

    def set_sppvt_parameters(self, kp=None, delta=None, eta=None, rho=None, control_mode=None):
        """动态调整SPPVT参数"""
        if kp is not None:
            self.sppvt_kp = kp
        if delta is not None:
            self.sppvt_delta = delta
        if eta is not None:
            self.sppvt_eta = eta
        if rho is not None:
            self.sppvt_rho = max(0.01, min(0.49, rho))  # 限制在(0, 0.5)范围内
        if control_mode is not None:
            self.control_mode = control_mode

        if self.debug:
            print(f"SPPVT参数更新: Kp={self.sppvt_kp}, δ={self.sppvt_delta}, "
                  f"η={self.sppvt_eta}, ρ={self.sppvt_rho}, 模式={self.control_mode}")

    def compute_derivatives(self, error_value, dt):
        """
        计算误差的一阶和二阶导数

        参数:
        error_value: 当前误差值（距离误差或速度误差）
        dt: 时间步长

        返回:
        velocity: 一阶导数（误差变化速度）
        acceleration: 二阶导数（误差变化加速度）
        jerk: 三阶导数（误差变化加加速度）
        """
        if dt <= 0:
            dt = 0.05  # 默认控制周期

        # 计算一阶导数（误差变化速度）
        velocity = (error_value - self.prev_error) / dt

        # 计算二阶导数（误差变化加速度）
        acceleration = (velocity - self.prev_velocity) / dt

        # 计算三阶导数（误差变化加加速度）
        jerk = (acceleration - self.prev_accel) / dt

        # 更新历史状态
        self.prev_error = error_value
        self.prev_velocity = velocity
        self.prev_accel = acceleration

        return velocity, acceleration, jerk

    def check_upgrade_condition(self, jerk, acceleration, velocity, control_error):
        """
        检查升级条件 - 根据控制模式使用不同的升级条件

        距离跟踪模式: (加速度 < 0) & (速度 <= δ) & (控制差 > η)
        速度跟踪模式: (加加速度 < 0) & (加速度 <= δ) & (控制差 > η)

        参数:
        jerk: 三阶导数（加加速度）
        acceleration: 二阶导数（加速度）
        velocity: 一阶导数（速度）
        control_error: 当前控制差

        返回:
        should_upgrade: 是否应该升级
        """
        # 防止频繁升级，至少间隔0.5秒
        current_time = time.time()
        if current_time - self.last_upgrade_time < 0.5:
            return False

        # 根据控制模式选择升级条件
        if self.control_mode == 'distance':
            # 距离跟踪模式：(加速度 < 0) & (速度 <= δ) & (控制差 > η)
            condition1 = acceleration < 0  # 加速度 < 0
            condition2 = abs(velocity) <= self.sppvt_delta  # 速度 <= δ
            condition3 = abs(control_error) > self.sppvt_eta  # 控制差 > η

            condition_name = "加速度"
            condition_value = acceleration
        else:  # speed mode
            # 速度跟踪模式：(加加速度 < 0) & (加速度 <= δ) & (控制差 > η)
            condition1 = jerk < 0  # 加加速度 < 0
            condition2 = abs(acceleration) <= self.sppvt_delta  # 加速度 <= δ
            condition3 = abs(control_error) > self.sppvt_eta  # 控制差 > η

            condition_name = "加加速度"
            condition_value = jerk

        should_upgrade = condition1 and condition2 and condition3

        if self.debug and should_upgrade:
            print(f"升级条件满足({self.control_mode}模式): {condition_name}={condition_value:.3f}<0, "
                  f"第二条件={abs(velocity if self.control_mode == 'distance' else acceleration):.3f}<={self.sppvt_delta}, "
                  f"控制差={abs(control_error):.3f}>{self.sppvt_eta}")

        return should_upgrade

    def upgrade_stage(self, error_value):
        """
        执行阶段升级

        参数:
        error_value: 当前误差值
        """
        # 升级到下一级
        self.stage += 1
        self.upgrade_count += 1
        self.last_upgrade_time = time.time()

        # 获取上一级级差
        if (self.stage - 1) in self.target_stages:
            prev_offset = self.target_stages[self.stage - 1]
        else:
            prev_offset = 0.0  # 第一级级差为0

        # 根据控制模式计算新的级差
        if self.control_mode == 'distance':
            # 距离跟踪：增加负级差，让控制更积极（距离误差为负时，加上负级差让误差更负）
            new_offset = -(prev_offset + self.sppvt_rho * error_value)
        else:  # speed mode
            # 速度跟踪：增加正级差，让控制更积极（速度误差为正时，加上正级差让误差更正）
            new_offset = prev_offset + self.sppvt_rho * error_value

        self.target_stages[self.stage] = new_offset

        if self.debug:
            direction = "减少" if self.control_mode == 'distance' else "增加"
            print(f"升级到第{self.stage}级: {direction}级差 {prev_offset:.3f} → {new_offset:.3f}, "
                  f"增量={abs(new_offset - prev_offset):.3f}, 总升级次数={self.upgrade_count}")

    def sppvt_longitudinal_control(self, error_value, dt=0.05):
        """
        SPPVT纵向控制器主函数
        与pid_longitudinal_control接口完全一致

        参数:
        error_value: 误差值 (期望值 - 实际值)
                    - 距离模式：距离误差 (期望距离 - 实际距离)
                    - 速度模式：速度误差 (期望速度 - 实际速度)
        dt: 时间步长 (默认0.05s)

        返回:
        control_output: 控制输出（加速度）
        """
        # 如果是第一级且没有设置级差，则设置初始级差为0
        if self.stage == 1 and self.stage not in self.target_stages:
            self.target_stages[1] = 0.0  # 第一级级差为0

        # 获取当前级的级差
        current_stage_offset = self.target_stages.get(self.stage, 0.0)

        # 计算导数
        velocity, acceleration, jerk = self.compute_derivatives(error_value, dt)

        # 计算扩大后的控制误差：原误差 + 级差
        enhanced_error = error_value + current_stage_offset

        # 检查升级条件（传入所有导数和原始误差）
        if self.check_upgrade_condition(jerk, acceleration, velocity, error_value):
            self.upgrade_stage(error_value)
            # 更新当前级级差
            current_stage_offset = self.target_stages.get(self.stage, 0.0)
            enhanced_error = error_value + current_stage_offset

        # 计算控制输出
        # 使用比例控制，基于扩大后的误差
        control_output = self.sppvt_kp * enhanced_error

        # 限制输出范围
        control_output = np.clip(control_output, self.max_decel, self.max_accel)

        # 调试输出
        if self.debug:
            error_type = "距离误差" if self.control_mode == 'distance' else "速度误差"
            print(f"SPPVT Stage {self.stage}({self.control_mode}): "
                  f"原始{error_type}={error_value:.2f}, "
                  f"级差={current_stage_offset:.2f}, "
                  f"扩大误差={enhanced_error:.2f}, "
                  f"控制输出={control_output:.2f}m/s²")

        return control_output

        return control_output

    def reset(self):
        """重置控制器状态"""
        self.stage = 1
        self.target_stages = {}
        self.prev_error = 0.0
        self.prev_velocity = 0.0
        self.prev_accel = 0.0
        self.upgrade_count = 0
        self.last_upgrade_time = 0

        if self.debug:
            print(f"SPPVT控制器状态已重置({self.control_mode}模式)")

    def get_status(self):
        """获取控制器状态信息"""
        return {
            'stage': self.stage,
            'upgrade_count': self.upgrade_count,
            'target_stages': self.target_stages.copy(),
            'control_mode': self.control_mode,
            'parameters': {
                'kp': self.sppvt_kp,
                'delta': self.sppvt_delta,
                'eta': self.sppvt_eta,
                'rho': self.sppvt_rho
            }
        }


# 全局SPPVT控制器实例 - 默认距离跟踪模式
_global_sppvt_controller = SPPVTLongitudinalController(control_mode='distance')


def sppvt_longitudinal_control(error_value, dt=0.05):
    """
    SPPVT纵向控制器 - 与PID接口完全一致的函数
    可直接替换pid_longitudinal_control函数

    参数:
    error_value: 误差值 (期望值 - 实际值)
                - 距离模式：距离误差 (期望距离 - 实际距离)
                - 速度模式：速度误差 (期望速度 - 实际速度)
    dt: 时间步长 (默认0.05s)

    返回:
    control_output: 控制输出（加速度 m/s²）
    """
    return _global_sppvt_controller.sppvt_longitudinal_control(error_value, dt)


def set_sppvt_parameters(kp=None, delta=None, eta=None, rho=None, control_mode=None):
    """设置全局SPPVT参数"""
    _global_sppvt_controller.set_sppvt_parameters(kp, delta, eta, rho, control_mode)


def reset_sppvt_controller():
    """重置全局SPPVT控制器"""
    _global_sppvt_controller.reset()


def enable_sppvt_debug(enable=True):
    """启用/禁用SPPVT调试输出"""
    _global_sppvt_controller.debug = enable


def get_sppvt_status():
    """获取SPPVT控制器状态"""
    return _global_sppvt_controller.get_status()


# 使用示例说明
"""
使用方法：

1. 设置参数和模式：
   set_sppvt_parameters(kp=0.8, delta=0.05, eta=0.3, rho=0.25, control_mode='distance')

2. 在控制循环中使用（距离跟踪）：
   distance_error = desired_distance - current_distance  # 注意：期望值 - 实际值
   accel = sppvt_longitudinal_control(distance_error, dt)

3. 或使用完整参数（推荐）：
   distance_error = desired_distance - current_distance
   accel = sppvt_longitudinal_control(
       distance_error, dt, 
       current_actual_value=current_distance,
       current_desired_value=desired_distance
   )

升级条件：
- 距离跟踪模式: (加速度 < 0) & (速度 <= δ) & (控制差 > η)
- 速度跟踪模式: (加加速度 < 0) & (加速度 <= δ) & (控制差 > η)

SPPVT原理：
- 距离跟踪：逐步降低目标距离，增强控制作用
- 速度跟踪：逐步提高目标速度，增强控制作用
"""