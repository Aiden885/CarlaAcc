"""
ACC Simulink决策管理器
使用UDP通信与Simulink决策模型交互，保持与acc_controller.py完全兼容的接口
"""
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional

from simulink_udp_interface import SimulinkUDPClient


class SimulinkACCDecisionManager:
    """
    Simulink版本的ACC决策管理器
    - 与ACCController接口完全兼容（鸭子类型）
    - 核心状态机由Simulink实现
    - 参数调整和安全逻辑保留在Python
    """

    def __init__(self, matlab_engine=None, debug: bool = False,
                 max_target_speed_kmh: float = 150.0):
        """
        初始化决策UDP客户端

        Args:
            matlab_engine: 保留参数用于向后兼容，实际不再使用
            debug: 是否打印调试信息
            max_target_speed_kmh: 最大目标速度
        """
        self.debug = debug
        self.max_target_speed_kmh = max_target_speed_kmh
        self.model_name = 'acc_decision_core'

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

        # 创建UDP客户端替代MATLAB Engine
        # 端口配置：25000（Python→Simulink），25001（Simulink→Python）
        # 输入：4个double（current_state, command_type, has_history, last_active_decision）
        # 输出：5个double（next_state, decision, control_enabled, next_has_history, next_last_decision）

        # ⚠️ 重要：初始状态必须是S2（无史待命），不能是S0（在控状态）
        initial_decision_state = [
            float(self.current_state),        # 2.0 = S2（无史待命）
            0.0,                              # command_type = NONE
            float(1 if self.has_history else 0),  # has_history = False
            float(self.last_active_decision)  # 8.0 = R8（系统待命）
        ]

        self.udp_client = SimulinkUDPClient(
            send_port=25000,
            recv_port=25001,
            num_inputs=4,
            num_outputs=5,
            timeout=2.0,  # ⚠️ 增加到2秒，解决阻塞模式下顺序调用的超时问题
            debug=debug,
            send_initial_packet=True,
            initial_values=initial_decision_state,  # 使用正确的初始状态
            local_send_port=9090  # 绑定固定源端口，匹配Simulink Remote port配置
        )

        if self.debug:
            print(f"✅ 决策UDP客户端已初始化: {self.model_name}")
            print(f"   初始状态: S{int(self.current_state)} (待命), R{int(self.last_active_decision)} (待命)")

    def reset(self):
        """重置ACC状态"""
        self.current_state = self.STATES['ADAPTIVE_NO_HISTORY_STANDBY']
        self.has_history = False
        self.last_active_decision = self.DECISIONS['SYSTEM_STANDBY']

        if self.debug:
            print("🔄 ACC控制器已重置 (Simulink版本)")

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
        通过UDP与Simulink决策模型通信 - 核心状态机查表

        输入（4个double）：
            In1: current_state (0-3)
            In2: command_type (0-7)
            In3: has_history (0/1)
            In4: last_active_decision (1-8)

        输出（5个double）：
            Out1: next_state
            Out2: decision
            Out3: control_enabled
            Out4: next_has_history
            Out5: next_last_decision

        Returns:
            Tuple[control_enabled, decision]
        """
        # 准备输入（4个输入端口）
        inputs = [
            float(self.current_state),           # In1: current_state (0-3)
            float(int(command_type)),            # In2: command_type (0-7)
            float(1 if self.has_history else 0), # In3: has_history (0/1)
            float(self.last_active_decision)     # In4: last_active_decision (1-8)
        ]

        # 通过UDP通信
        start_time = time.time()
        try:
            # ⚠️ 双调用解决Simulink UDP延迟问题：
            # 第一次调用返回的是上一帧的结果，丢弃
            # 第二次调用才是当前输入对应的结果
            self.udp_client.call(inputs)  # 预热调用，丢弃结果
            outputs = self.udp_client.call(inputs)  # 实际调用，使用结果

            elapsed_ms = (time.time() - start_time) * 1000.0

            # 解析输出（5个输出端口）
            next_state = int(round(outputs[0]))
            decision = int(round(outputs[1]))
            control_enabled = bool(int(round(outputs[2])))
            next_has_history = int(round(outputs[3]))
            next_last_decision = int(round(outputs[4]))

            # 更新Python端状态
            self.current_state = next_state
            self.has_history = bool(next_has_history)
            self.last_active_decision = next_last_decision

            if self.debug:
                print(f"🔧 决策UDP: S{self.current_state}, R{decision}, "
                      f"enabled={control_enabled}, history={self.has_history} ({elapsed_ms:.1f}ms)")

            return control_enabled, decision

        except Exception as e:
            if self.debug:
                print(f"❌ UDP通信失败: {e}")

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
        """关闭UDP连接"""
        try:
            self.udp_client.cleanup()
        except Exception as e:
            if self.debug:
                print(f"⚠️ 决策UDP清理失败: {e}")
