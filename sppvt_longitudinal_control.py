import numpy as np
import time
import matlab.engine
import threading


class SPPVTLongitudinalController:
    """
    SPPVT纵向控制器 - 混合Python-Simulink实现
    与PID控制器接口完全一致，可直接替换
    支持距离跟踪和速度跟踪两种模式

    核心计算功能由Simulink模型提供，Python负责接口和状态管理
    """

    def __init__(self, control_mode='distance'):
        # SPPVT控制参数
        self.sppvt_kp = 1.0  # 比例控制系数
        self.sppvt_delta = 0.05  # 变目标阈值（接近0的正值）
        self.sppvt_eta = 0.2  # 定速控制精度
        self.sppvt_rho = 0.25  # 惩罚系数 (0 < ρ < 0.5)

        # 控制模式：'distance' 或 'speed'
        self.control_mode = control_mode
        self.control_mode_flag = 1.0 if control_mode == 'distance' else 2.0

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

        # 升级历史
        self.upgrade_count = 0
        self.last_upgrade_time = 0

        # 调试标志
        self.debug = True

        # MATLAB引擎和Simulink模型
        self.matlab_engine = None
        self.model_name = 'sppvt_control_model'
        self.engine_lock = threading.Lock()

        # 初始化MATLAB引擎
        self._init_matlab_engine()

    def _init_matlab_engine(self):
        """初始化MATLAB引擎和加载Simulink模型"""
        try:
            if self.debug:
                print("正在启动MATLAB引擎...")

            # 启动MATLAB引擎
            self.matlab_engine = matlab.engine.start_matlab()

            # 添加模型路径到MATLAB路径
            self.matlab_engine.addpath(self.matlab_engine.pwd(), nargout=0)

            # 检查模型文件是否存在
            model_exists = self.matlab_engine.exist(f'{self.model_name}.slx', 'file')
            if model_exists == 0:
                raise FileNotFoundError(f"找不到Simulink模型文件: {self.model_name}.slx")

            # 加载模型
            self.matlab_engine.load_system(self.model_name, nargout=0)

            # 配置模型参数
            self.matlab_engine.set_param(self.model_name, 'StopTime', '0.05', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'FixedStep', '0.001', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'LoadExternalInput', 'on', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'ExternalInput', '[external_input_data]', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'SaveOutput', 'on', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'OutputSaveName', 'yout', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'SaveFormat', 'StructureWithTime', nargout=0)

            if self.debug:
                print(f"MATLAB引擎启动成功，Simulink模型 {self.model_name} 已加载")

        except Exception as e:
            print(f"警告: MATLAB引擎初始化失败: {e}")
            print("将使用纯Python实现作为后备方案")
            self.matlab_engine = None

    def __del__(self):
        """析构函数，清理MATLAB引擎"""
        if self.matlab_engine is not None:
            try:
                self.matlab_engine.close_system(self.model_name, 0, nargout=0)
                self.matlab_engine.quit()
            except:
                pass

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
            self.control_mode_flag = 1.0 if control_mode == 'distance' else 2.0

        if self.debug:
            print(f"SPPVT参数更新: Kp={self.sppvt_kp}, δ={self.sppvt_delta}, "
                  f"η={self.sppvt_eta}, ρ={self.sppvt_rho}, 模式={self.control_mode}")

    def _run_simulink_model(self, error_value, dt):
        """
        运行Simulink模型进行核心计算

        返回:
        (control_output, velocity, acceleration, jerk, should_upgrade)
        """
        with self.engine_lock:
            try:
                # 创建时间向量
                time_points = np.linspace(0, 0.05, 51)

                # 准备输入数据
                input_values = [
                    float(error_value),
                    float(dt),
                    float(self.current_stage_offset),
                    float(self.sppvt_kp),
                    float(self.max_accel),
                    float(self.max_decel),
                    float(self.prev_error),
                    float(self.prev_velocity),
                    float(self.prev_accel),
                    float(self.sppvt_delta),
                    float(self.sppvt_eta),
                    float(self.control_mode_flag)
                ]

                # 创建输入数据矩阵
                input_data = []
                for t in time_points:
                    row = [float(t)] + input_values
                    input_data.append(row)

                # 转换为MATLAB数组
                matlab_input = matlab.double(input_data)

                # 将数据传入工作空间
                self.matlab_engine.workspace['external_input_data'] = matlab_input

                # 运行仿真
                self.matlab_engine.eval(f"simOut = sim('{self.model_name}');", nargout=0)

                # 检查simOut是否存在
                if not self.matlab_engine.exist('simOut', 'var'):
                    raise RuntimeError("仿真未能生成输出")

                # 从simOut获取yout - 使用eval方式访问
                self.matlab_engine.eval("yout = simOut.yout;", nargout=0)

                # 确认yout存在
                if not self.matlab_engine.exist('yout', 'var'):
                    # 尝试其他方式
                    self.matlab_engine.eval("yout = get(simOut, 'yout');", nargout=0)

                # 使用eval获取输出值
                control_output = float(self.matlab_engine.eval("yout.signals(1).values(end)"))
                velocity = float(self.matlab_engine.eval("yout.signals(2).values(end)"))
                acceleration = float(self.matlab_engine.eval("yout.signals(3).values(end)"))
                jerk = float(self.matlab_engine.eval("yout.signals(4).values(end)"))
                should_upgrade_value = float(self.matlab_engine.eval("yout.signals(5).values(end)"))
                should_upgrade = should_upgrade_value > 0.5

                if self.debug:
                    print(f"Simulink输出: control={control_output:.3f}, vel={velocity:.3f}, "
                          f"acc={acceleration:.3f}, jerk={jerk:.3f}, upgrade={should_upgrade}")

                return control_output, velocity, acceleration, jerk, should_upgrade

            except Exception as e:
                if self.debug:
                    print(f"Simulink模型运行失败: {e}")
                    import traceback
                    traceback.print_exc()
                # 返回None表示失败，将使用Python后备实现
                return None

    def compute_derivatives(self, error_value, dt):
        """
        计算误差的一阶和二阶导数（Python后备实现）

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

        return velocity, acceleration, jerk

    def check_upgrade_condition(self, jerk, acceleration, velocity, control_error):
        """
        检查升级条件（Python后备实现）

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
        self.current_stage_offset = new_offset

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
            self.current_stage_offset = 0.0

        # 尝试使用Simulink模型
        if self.matlab_engine is not None:
            result = self._run_simulink_model(error_value, dt)

            if result is not None:
                # Simulink运行成功
                control_output, velocity, acceleration, jerk, should_upgrade = result

                # 更新历史状态
                self.prev_error = error_value
                self.prev_velocity = velocity
                self.prev_accel = acceleration

                # 检查是否需要升级（使用Simulink的判断结果）
                if should_upgrade:
                    # 防止频繁升级
                    current_time = time.time()
                    if current_time - self.last_upgrade_time >= 0.5:
                        self.upgrade_stage(error_value)

                # 调试输出
                if self.debug:
                    error_type = "距离误差" if self.control_mode == 'distance' else "速度误差"
                    print(f"SPPVT Stage {self.stage}({self.control_mode})[Simulink]: "
                          f"原始{error_type}={error_value:.2f}, "
                          f"级差={self.current_stage_offset:.2f}, "
                          f"扩大误差={error_value + self.current_stage_offset:.2f}, "
                          f"控制输出={control_output:.2f}m/s²")

                return control_output

        # Simulink不可用或运行失败，使用Python后备实现
        # 获取当前级的级差
        current_stage_offset = self.target_stages.get(self.stage, 0.0)
        self.current_stage_offset = current_stage_offset

        # 计算导数
        velocity, acceleration, jerk = self.compute_derivatives(error_value, dt)

        # 计算扩大后的控制误差：原误差 + 级差
        enhanced_error = error_value + current_stage_offset

        # 检查升级条件（传入所有导数和原始误差）
        if self.check_upgrade_condition(jerk, acceleration, velocity, error_value):
            self.upgrade_stage(error_value)
            # 更新当前级级差
            current_stage_offset = self.target_stages.get(self.stage, 0.0)
            self.current_stage_offset = current_stage_offset
            enhanced_error = error_value + current_stage_offset

        # 更新历史状态
        self.prev_error = error_value
        self.prev_velocity = velocity
        self.prev_accel = acceleration

        # 计算控制输出
        # 使用比例控制，基于扩大后的误差
        control_output = self.sppvt_kp * enhanced_error

        # 限制输出范围
        control_output = np.clip(control_output, self.max_decel, self.max_accel)

        # 调试输出
        if self.debug:
            error_type = "距离误差" if self.control_mode == 'distance' else "速度误差"
            print(f"SPPVT Stage {self.stage}({self.control_mode})[Python]: "
                  f"原始{error_type}={error_value:.2f}, "
                  f"级差={current_stage_offset:.2f}, "
                  f"扩大误差={enhanced_error:.2f}, "
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
            'matlab_engine_status': 'active' if self.matlab_engine is not None else 'inactive',
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

混合实现说明：
- 优先使用Simulink模型进行核心计算（导数计算、升级判断、控制输出）
- 如果MATLAB引擎不可用，自动切换到纯Python实现
- 所有外部接口保持不变，确保向后兼容
"""