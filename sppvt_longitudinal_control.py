import numpy as np
import time

# SPPVT控制器 - 纯Python实现
# 移除了所有MATLAB/Simulink依赖，以解决性能瓶颈

class SPPVTLongitudinalController:
    """
    SPPVT纵向控制器 - 纯Python实现
    与PID控制器接口完全一致，可直接替换
    支持距离跟踪和速度跟踪两种模式
    """

    def __init__(self, control_mode='distance'):
        # SPPVT控制参数
        self.sppvt_kp = 1.0  # 比例控制系数
        self.sppvt_delta = 0.05  # 变目标阈值（接近0的正值）
        self.sppvt_eta = 0.2  # 定速控制精度
        self.sppvt_rho = 0.25  # 惩罚系数 (0 < ρ < 0.5)

        # 控制模式：'distance'(处理距离误差) 或 'speed'
        self.control_mode = control_mode

        # 输出限制
        self.max_accel = 2.0
        self.max_decel = -3.0

        # 控制状态
        self.stage = 1  # 当前阶式级数 i
        self.target_stages = {}  # 各级目标值 V_i
        self.current_stage_offset = 0.0  # 当前级差

        # 历史状态（用于计算导数）
        self.prev_error = 0.0  # 上次误差值
        self.prev_velocity = 0.0  # 上次速度（一阶导数）
        self.prev_accel = 0.0  # 上次加速度（二阶导数）

        # 误差符号跟踪（用于检测符号变化）
        self.prev_error_sign = 0  # 上次误差符号：1(正), -1(负), 0(零)

        # 升级历史
        self.upgrade_count = 0
        self.last_upgrade_time = 0

        # 调试标志
        self.debug = True
        
        if self.debug:
            print("SPPVT控制器已初始化 (纯Python模式)")

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
        """
        if dt <= 0:
            dt = 0.05  # 默认控制周期

        velocity = (error_value - self.prev_error) / dt
        acceleration = (velocity - self.prev_velocity) / dt
        jerk = (acceleration - self.prev_accel) / dt

        return velocity, acceleration, jerk

    def check_upgrade_condition(self, jerk, acceleration, velocity, control_error):
        """
        检查升级条件
        """
        current_time = time.time()
        if current_time - self.last_upgrade_time < 0.5:
            return False

        if self.control_mode == 'distance':
            condition1 = acceleration < 0
            condition2 = abs(velocity) <= self.sppvt_delta
            condition3 = abs(control_error) > self.sppvt_eta
            condition_name = "加速度"
            condition_value = acceleration
        else:  # speed mode
            condition1 = jerk < 0
            condition2 = abs(acceleration) <= self.sppvt_delta
            condition3 = abs(control_error) > self.sppvt_eta
            condition_name = "加加速度"
            condition_value = jerk

        should_upgrade = condition1 and condition2 and condition3

        if self.debug and should_upgrade:
            print(f"升级条件满足({self.control_mode}模式): {condition_name}={condition_value:.3f}<0, "
                  f"第二条件={abs(velocity if self.control_mode == 'distance' else acceleration):.3f}<={self.sppvt_delta}, "
                  f"控制差={abs(control_error):.3f}>{self.sppvt_eta}")

        return should_upgrade

    def _check_error_sign_change(self, error_value):
        """
        检查误差符号变化，如果变化则重置状态
        """
        if abs(error_value) < 1e-6:
            current_sign = 0
        elif error_value > 0:
            current_sign = 1
        else:
            current_sign = -1

        sign_changed = False
        if self.prev_error_sign != 0 and current_sign != 0 and self.prev_error_sign != current_sign:
            sign_changed = True
            if self.debug:
                sign_names = {-1: "负", 0: "零", 1: "正"}
                print(f"检测到误差符号变化: {sign_names[self.prev_error_sign]} → {sign_names[current_sign]}, 重置到初始级")
            self.reset()

        if current_sign != 0:
            self.prev_error_sign = current_sign

        return sign_changed

    def upgrade_stage(self, error_value):
        """
        执行阶段升级
        """
        self.stage += 1
        self.upgrade_count += 1
        self.last_upgrade_time = time.time()

        prev_offset = self.target_stages.get(self.stage - 1, 0.0)

        if error_value > 0:
            new_offset = prev_offset + self.sppvt_rho * abs(error_value)
        else:
            new_offset = prev_offset - self.sppvt_rho * abs(error_value)

        self.target_stages[self.stage] = new_offset
        self.current_stage_offset = new_offset

        if self.debug:
            error_direction = "正" if error_value > 0 else "负"
            offset_direction = "正" if new_offset >= prev_offset else "负"
            print(f"升级到第{self.stage}级: {error_direction}误差({error_value:.3f}) → 增加{offset_direction}级差 {prev_offset:.3f} → {new_offset:.3f}")

    def sppvt_longitudinal_control(self, error_value, dt=0.05):
        """
        SPPVT纵向控制器主函数
        """
        self._check_error_sign_change(error_value)

        if self.stage == 1 and self.stage not in self.target_stages:
            self.target_stages[1] = 0.0
            self.current_stage_offset = 0.0

        # --- 纯Python实现 ---
        current_stage_offset = self.target_stages.get(self.stage, 0.0)
        self.current_stage_offset = current_stage_offset

        velocity, acceleration, jerk = self.compute_derivatives(error_value, dt)

        enhanced_error = error_value + current_stage_offset

        if self.check_upgrade_condition(jerk, acceleration, velocity, error_value):
            self.upgrade_stage(error_value)
            current_stage_offset = self.target_stages.get(self.stage, 0.0)
            self.current_stage_offset = current_stage_offset
            enhanced_error = error_value + current_stage_offset

        self.prev_error = error_value
        self.prev_velocity = velocity
        self.prev_accel = acceleration

        control_output = self.sppvt_kp * enhanced_error
        control_output = np.clip(control_output, self.max_decel, self.max_accel)

        if self.debug:
            error_type = "距离误差" if self.control_mode == 'distance' else "速度误差"
            error_unit = "m" if self.control_mode == 'distance' else "m/s"
            print(f"SPPVT Stage {self.stage}({self.control_mode})[Python]: "
                  f"原始{error_type}={error_value:.3f}{error_unit}, "
                  f"级差={current_stage_offset:.3f}{error_unit}, "
                  f"扩大误差={enhanced_error:.3f}{error_unit}, "
                  f"控制输出={control_output:.2f}m/s²")

        return control_output

    def reset(self):
        """重置控制器状态"""
        self.stage = 1
        self.target_stages = {}
        self.current_stage_offset = 0.0
        self.prev_error = 0.0
        self.prev_velocity = 0.0
        self.prev_accel = 0.0
        self.prev_error_sign = 0
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
            'matlab_engine_status': 'removed',
            'parameters': {
                'kp': self.sppvt_kp,
                'delta': self.sppvt_delta,
                'eta': self.sppvt_eta,
                'rho': self.sppvt_rho
            }
        }

# 全局SPPVT控制器实例
_global_sppvt_controller = SPPVTLongitudinalController(control_mode='distance')


def sppvt_longitudinal_control(error_value, dt=0.05):
    return _global_sppvt_controller.sppvt_longitudinal_control(error_value, dt)

def set_sppvt_parameters(kp=None, delta=None, eta=None, rho=None, control_mode=None):
    _global_sppvt_controller.set_sppvt_parameters(kp, delta, eta, rho, control_mode)

def reset_sppvt_controller():
    _global_sppvt_controller.reset()

def enable_sppvt_debug(enable=True):
    _global_sppvt_controller.debug = enable

def get_sppvt_status():
    return _global_sppvt_controller.get_status()
