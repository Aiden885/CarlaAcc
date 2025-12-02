"""
ACC Simulink决策管理器
使用Simulink实现决策状态机，保持与acc_controller.py完全兼容的接口
"""
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional

try:
    import matlab  # type: ignore
except ImportError:
    matlab = None

from matlab_engine_factory import get_matlab_engine


class SimulinkACCDecisionManager:
    """
    Simulink版本的ACC决策管理器
    - 与ACCController接口完全兼容（鸭子类型）
    - 核心状态机由Simulink实现
    - 参数调整和安全逻辑保留在Python
    """

    def __init__(self, matlab_engine=None, debug: bool = False,
                 max_target_speed_kmh: float = 150.0):
        self.debug = debug
        self.max_target_speed_kmh = max_target_speed_kmh

        # MATLAB Engine管理
        self.matlab_engine = matlab_engine or get_matlab_engine(force=False)
        self.model_name = 'acc_decision_core'
        self.model_loaded = False
        self.fast_restart_enabled = False
        self.use_rapid_accelerator = True  # 使用Rapid Accelerator以获得最快速度
        self.rapid_accel_built = False

        # ACC状态定义（与ACCController保持一致）
        self.STATES = {
            'ACTIVE_CONTROL': 0,           # S0: 在控状态
            'ADAPTIVE_HISTORY_STANDBY': 1, # S1: 适速有史待命
            'ADAPTIVE_NO_HISTORY_STANDBY': 2, # S2: 适速无史待命
            'LOW_SPEED': 3                 # S3: 低速状态
        }

        # 决策定义（与ACCController保持一致）
        self.DECISIONS = {
            'DECREASE_SPEED': 1,           # R1: 速度降低
            'INCREASE_SPEED': 2,           # R2: 速度增加
            'DECREASE_DISTANCE': 3,        # R3: 时距降低
            'INCREASE_DISTANCE': 4,        # R4: 时距增加
            'ACTIVATE_CURRENT_SPEED': 5,   # R5: 无继控制
            'ACTIVATE_INHERITED_SPEED': 6, # R6: 继承控制
            'TORQUE_ARBITRATION': 7,       # R7: 扭矩仲裁
            'SYSTEM_STANDBY': 8            # R8: 系统待命
        }

        # Python端管理的状态（持久化）
        self.current_state = self.STATES['ADAPTIVE_NO_HISTORY_STANDBY']
        self.has_history = False
        self.last_active_decision = self.DECISIONS['SYSTEM_STANDBY']

        # Python端管理的参数
        self.params = {
            'V_target_kmh': 50.0,
            'V_min_kmh': 30.0,
            'G2_s': 2.0
        }

        # 调试计数器
        self.debug_counter = 0

    def reset(self):
        """重置ACC状态"""
        self.current_state = self.STATES['ADAPTIVE_NO_HISTORY_STANDBY']
        self.has_history = False
        self.last_active_decision = self.DECISIONS['SYSTEM_STANDBY']

        if self.debug:
            print("🔄 ACC控制器已重置 (Simulink版本)")

    def _ensure_model_loaded(self):
        """确保Simulink模型已加载"""
        if self.matlab_engine is None:
            self.matlab_engine = get_matlab_engine()
            if self.matlab_engine is None:
                raise RuntimeError("❌ MATLAB Engine未初始化")

        if not self.model_loaded:
            try:
                # 加载查找表数据到工作区
                self.matlab_engine.eval("load('decision_lookup_data.mat')", nargout=0)

                # 检查模型是否已经加载
                is_loaded = self.matlab_engine.bdIsLoaded(self.model_name, nargout=1)

                if not is_loaded:
                    # 加载模型
                    self.matlab_engine.load_system(self.model_name, nargout=0)
                else:
                    if self.debug:
                        print(f"✅ 模型已在工作区: {self.model_name}")

                # 配置仿真参数（快速单步仿真）
                self.matlab_engine.set_param(self.model_name, 'StopTime', '0.05', nargout=0)
                self.matlab_engine.set_param(self.model_name, 'Solver', 'FixedStepDiscrete', nargout=0)
                self.matlab_engine.set_param(self.model_name, 'FixedStep', '0.05', nargout=0)

                # 启用Accelerator模式加速仿真
                # 参考: https://www.mathworks.com/help/simulink/ug/how-the-acceleration-modes-work.html
                try:
                    self.matlab_engine.set_param(self.model_name, 'SimulationMode', 'accelerator', nargout=0)
                    # 打开Fast Restart以避免重复初始化/编译
                    self.matlab_engine.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
                    if self.debug:
                        print("✅ 已启用Accelerator模式 + Fast Restart")
                except Exception as e:
                    if self.debug:
                        print(f"⚠️ 无法启用Accelerator/FastRestart: {e}, 使用Normal模式")

                self.model_loaded = True

                if self.debug:
                    print(f"✅ Simulink决策模型已就绪: {self.model_name}")
                    print(f"✅ 决策查找表数据已加载")

            except Exception as e:
                raise RuntimeError(f"❌ 加载Simulink决策模型失败: {e}")

    def validate_and_process_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入验证和预处理 - 与ACCController接口兼容

        Args:
            raw_input: 原始输入数据

        Returns:
            Dict: 验证后的输入数据
        """
        # 复制并验证数值范围
        validated = raw_input.copy()

        # 核心数值验证
        validated['ego_speed_kmh'] = np.clip(raw_input['ego_speed_kmh'], 0, 200)
        validated['ego_speed_ms'] = np.clip(raw_input['ego_speed_ms'], 0, 60)
        validated['V_target_kmh'] = np.clip(raw_input['V_target_kmh'], 30, self.max_target_speed_kmh)
        validated['V_min_kmh'] = np.clip(raw_input['V_min_kmh'], 20, 50)
        validated['G2_s'] = np.clip(raw_input['G2_s'], 1.0, 8.0)

        # 确保参数一致性
        if validated['V_min_kmh'] > validated['V_target_kmh']:
            validated['V_min_kmh'] = validated['V_target_kmh'] - 10

        # 处理NaN和无效值
        for key in ['ego_speed_kmh', 'ego_speed_ms', 'control_error']:
            if not np.isfinite(validated.get(key, 0)):
                validated[key] = 0.0

        return validated

    def process_keyboard_command(self, command_type: int, ego_speed_kmh: float) -> Tuple[bool, int, Dict[str, float]]:
        """
        处理键盘指令 - 与ACCController接口兼容

        Args:
            command_type: 指令类型 (0=NONE, 1-7=I0-I6)
            ego_speed_kmh: 当前车速

        Returns:
            Tuple[control_enabled, decision, updated_params]
        """
        # 1. Python端参数调整（保留在Python，与原acc_controller.py逻辑一致）
        updated_params = self._adjust_parameters(command_type, ego_speed_kmh)

        # 2. Python端低速检测（安全相关，保留在Python）
        self._handle_low_speed_transition(ego_speed_kmh)

        # 3. 调用Simulink状态机
        control_enabled, decision = self._run_simulink_decision(command_type)

        return control_enabled, decision, updated_params

    def _adjust_parameters(self, command_type: int, ego_speed_kmh: float) -> Dict[str, float]:
        """
        参数调整 - 保留在Python（与原acc_controller.py逻辑一致）
        处理E/Q/T/R键的参数调整
        """
        updated_params = {}

        if command_type in [1, 2, 3, 4]:  # 参数调整指令
            speed_step = 5.0
            time_gap_step = 0.2
            in_active_control = (self.current_state == self.STATES['ACTIVE_CONTROL'])

            if command_type == 1:  # E键: 降速 or 当速启控
                if in_active_control:
                    new_target = max(self.params['V_min_kmh'], self.params['V_target_kmh'] - speed_step)
                    updated_params['V_target_kmh'] = new_target
                    if self.debug:
                        print(f"⌨️ E键降速: {self.params['V_target_kmh']:.1f} → {new_target:.1f} km/h")
                # 未在控时，E键用于启控，不调整速度

            elif command_type == 2:  # Q键: 增速 or 继承启控
                if in_active_control:
                    new_target = min(self.max_target_speed_kmh, self.params['V_target_kmh'] + speed_step)
                    updated_params['V_target_kmh'] = new_target
                    if self.debug:
                        print(f"⌨️ Q键增速: {self.params['V_target_kmh']:.1f} → {new_target:.1f} km/h")
                # 未在控时，Q键用于启控，不调整速度

            elif command_type == 3:  # T键: 降距
                new_gap = max(1.0, self.params['G2_s'] - time_gap_step)
                updated_params['G2_s'] = new_gap
                if self.debug:
                    print(f"⌨️ T键降距: {self.params['G2_s']:.1f} → {new_gap:.1f} s")

            elif command_type == 4:  # R键: 增距
                new_gap = min(5.0, self.params['G2_s'] + time_gap_step)
                updated_params['G2_s'] = new_gap
                if self.debug:
                    print(f"⌨️ R键增距: {self.params['G2_s']:.1f} → {new_gap:.1f} s")

        # 更新参数
        self.params.update(updated_params)

        return updated_params

    def _handle_low_speed_transition(self, ego_speed_kmh: float):
        """
        低速检测 - 保留在Python（安全相关）
        """
        if ego_speed_kmh < self.params['V_min_kmh']:
            if self.current_state != self.STATES['LOW_SPEED']:
                self.current_state = self.STATES['LOW_SPEED']
                if self.debug:
                    print(f"🚗 车速过低({ego_speed_kmh:.1f} < {self.params['V_min_kmh']:.1f})，转入S3低速状态")
        else:
            # 从S3恢复
            if self.current_state == self.STATES['LOW_SPEED']:
                if self.has_history:
                    self.current_state = self.STATES['ADAPTIVE_HISTORY_STANDBY']
                    if self.debug:
                        print("🚗 车速恢复，转入S1有史待命")
                else:
                    self.current_state = self.STATES['ADAPTIVE_NO_HISTORY_STANDBY']
                    if self.debug:
                        print("🚗 车速恢复，转入S2无史待命")

    def _run_simulink_decision(self, command_type: int) -> Tuple[bool, int]:
        """
        运行Simulink决策模型 - 核心状态机查表

        Returns:
            Tuple[control_enabled, decision]
        """
        self._ensure_model_loaded()

        if matlab is None:
            raise RuntimeError("matlab python package is required for Simulink decision manager.")

        # 准备输入（4个输入端口）
        inputs = [
            float(self.current_state),           # In1: current_state (0-3)
            float(int(command_type)),            # In2: command_type (0-7)
            float(1 if self.has_history else 0), # In3: has_history (0/1)
            float(self.last_active_decision)     # In4: last_active_decision (1-8)
        ]

        # 构建外部输入（时间 + 信号值）
        dt = 0.05
        ext_input = [
            [0.0] + inputs,
            [dt] + inputs,
        ]
        matlab_ext_input = matlab.double(ext_input)

        # 运行仿真
        start_time = time.time()
        try:
            self.matlab_engine.workspace['ext_input'] = matlab_ext_input

            # 启用Fast Restart以加速重复仿真
            # 参考: https://www.mathworks.com/help/simulink/ug/fast-restart-workflow.html
            if not self.fast_restart_enabled:
                # 首次仿真:启用Fast Restart
                self.matlab_engine.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
                self.fast_restart_enabled = True
                if self.debug:
                    print("✅ 已启用Fast Restart模式")

            sim_in = self.matlab_engine.eval(f"Simulink.SimulationInput('{self.model_name}')", nargout=1)
            sim_in = self.matlab_engine.setExternalInput(sim_in, 'ext_input', nargout=1)

            # 使用Fast Restart加速仿真
            # 参考: https://www.mathworks.com/help/simulink/ug/write-matlab-scripts-to-run-fast-restart-simulations.html
            sim_in = self.matlab_engine.setModelParameter(sim_in, 'FastRestart', 'on', nargout=1)

            sim_out = self.matlab_engine.sim(sim_in, nargout=1)

            elapsed_ms = (time.time() - start_time) * 1000.0

            # 提取输出（5个输出端口）
            # 使用花括号{}访问Dataset - 在Python中需要通过eval或subsref
            self.matlab_engine.workspace['sim_out'] = sim_out

            # 方法: 直接在MATLAB中使用{}索引提取数据
            # Dataset使用花括号访问,参考: https://www.mathworks.com/help/simulink/slref/simulink.simulationdata.dataset.get.html
            next_state = int(round(float(self.matlab_engine.eval('sim_out.yout{1}.Values.Data(end)', nargout=1))))
            decision = int(round(float(self.matlab_engine.eval('sim_out.yout{2}.Values.Data(end)', nargout=1))))
            control_enabled = bool(int(round(float(self.matlab_engine.eval('sim_out.yout{3}.Values.Data(end)', nargout=1)))))
            next_has_history = int(round(float(self.matlab_engine.eval('sim_out.yout{4}.Values.Data(end)', nargout=1))))
            next_last_decision = int(round(float(self.matlab_engine.eval('sim_out.yout{5}.Values.Data(end)', nargout=1))))

            # 更新Python端状态
            self.current_state = next_state
            self.has_history = bool(next_has_history)
            self.last_active_decision = next_last_decision

            if self.debug:
                print(f"🔧 Simulink决策: S{self.current_state}, R{decision}, "
                      f"enabled={control_enabled}, history={self.has_history} ({elapsed_ms:.1f}ms)")

            return control_enabled, decision

        except Exception as e:
            if self.debug:
                print(f"❌ Simulink仿真失败: {e}")

            # 重置到安全状态,避免状态不一致
            # 如果有历史,退到S1(有史待命),否则退到S2(无史待命)
            if self.has_history:
                self.current_state = self.STATES['ADAPTIVE_HISTORY_STANDBY']
            else:
                self.current_state = self.STATES['ADAPTIVE_NO_HISTORY_STANDBY']

            # last_active_decision保持不变(保留用户最后的有效决策)

            if self.debug:
                print(f"🔄 状态已重置: S{self.current_state}, history={self.has_history}, "
                      f"last_decision=R{self.last_active_decision}")

            # 回退到待命状态
            return False, self.DECISIONS['SYSTEM_STANDBY']

    def get_state_info(self) -> Dict[str, Any]:
        """获取当前状态信息 - 与ACCController接口兼容"""
        state_names = {v: k for k, v in self.STATES.items()}
        decision_names = {v: k for k, v in self.DECISIONS.items()}

        return {
            'current_state': self.current_state,
            'state_name': state_names.get(self.current_state, 'UNKNOWN'),
            'has_history': self.has_history,
            'last_active_decision': self.last_active_decision,
            'last_decision_name': decision_names.get(self.last_active_decision, 'UNKNOWN'),
            'params': self.params.copy()
        }

    def update_debug_counter(self) -> int:
        """更新调试计数器并返回调试代码 - 与ACCController接口兼容"""
        self.debug_counter += 1

        if self.debug_counter % 20 == 0:
            # 生成复合调试代码
            debug_code = 1000 + self.current_state * 100 + self.last_active_decision * 10
            if self.debug:
                print(f"🔍 ACC Debug (Simulink): State=S{self.current_state}, "
                      f"Decision=R{self.last_active_decision}, Code={debug_code}")
            return debug_code
        else:
            return self.debug_counter

    def cleanup(self):
        """清理资源"""
        if self.matlab_engine and self.model_loaded:
            try:
                # 禁用Fast Restart
                if self.fast_restart_enabled:
                    self.matlab_engine.set_param(self.model_name, 'FastRestart', 'off', nargout=0)
                    self.fast_restart_enabled = False
                    if self.debug:
                        print("✅ Fast Restart已禁用")

                self.matlab_engine.close_system(self.model_name, 0, nargout=0)
                if self.debug:
                    print(f"✅ Simulink决策模型已关闭: {self.model_name}")
            except Exception:
                pass
