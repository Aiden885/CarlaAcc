"""
统一的Simulink集成管理器
替代原来的双UDP架构（acc_decision_simulink_manager + sppvt_manager_simulink）
使用单一UDP接口与acc_integrated_model.slx通信

设计理念：
- Python维护所有状态（Decision状态、SPPVT状态）
- Simulink只做纯函数式计算（每帧输入→计算→输出）
- 避免因果环：使用上一帧的control_enabled来控制当前帧的SPPVT输入
"""
from __future__ import annotations

import time
import numpy as np
from typing import Dict, Any, Optional

from simulink_udp_interface import SimulinkUDPClient
from acc_config import ACCConfig


class IntegratedSimulinkManager:
    """
    统一的Simulink集成管理器

    与acc_integrated_model.slx通信：
    - 输入：9个double (Decision 4个 + SPPVT 5个)
    - 输出：10个double (Decision 5个 + SPPVT 5个)

    UDP配置：
    - Python发送 → 27000 (Simulink接收), 源端口: 9090
    - Python接收 ← 27001 (Simulink发送)
    """

    # ACC状态定义（与原Decision模块保持一致）
    STATES = {
        'ACTIVE_CONTROL': 0,           # S0: 在控状态
        'ADAPTIVE_HISTORY_STANDBY': 1, # S1: 适速有史待命
        'ADAPTIVE_NO_HISTORY_STANDBY': 2, # S2: 适速无史待命
        'LOW_SPEED': 3                 # S3: 低速状态
    }

    # 决策定义（与原Decision模块保持一致）
    DECISIONS = {
        'DECREASE_SPEED': 1,           # R1: 速度降低
        'INCREASE_SPEED': 2,           # R2: 速度增加
        'DECREASE_DISTANCE': 3,        # R3: 时距降低
        'INCREASE_DISTANCE': 4,        # R4: 时距增加
        'ACTIVATE_CURRENT_SPEED': 5,   # R5: 无继控制
        'ACTIVATE_INHERITED_SPEED': 6, # R6: 继承控制
        'TORQUE_ARBITRATION': 7,       # R7: 扭矩仲裁
        'SYSTEM_STANDBY': 8            # R8: 系统待命
    }

    def __init__(self, debug: bool = False, max_target_speed_kmh: Optional[float] = None, config: Optional[ACCConfig] = None):
        """
        初始化统一管理器

        Args:
            debug: 是否打印调试信息
            max_target_speed_kmh: 最大目标速度（None 时使用配置）
            config: ACC配置对象（未提供时使用默认配置）
        """
        self.debug = debug
        self.config = config or ACCConfig()
        self.max_target_speed_kmh = max_target_speed_kmh if max_target_speed_kmh is not None else self.config.max_target_speed_kmh
        self.sppvt_rho = getattr(self.config, 'integrated_sppvt_rho', 0.1)  # stage offset累积系数
        self.upgrade_cooldown_frames = getattr(self.config, 'sppvt_upgrade_cooldown', 2)  # 升级冷却周期

        # ============ Decision状态（Python端维护） ============
        self.decision_state = {
            'current_state': self.STATES['ADAPTIVE_NO_HISTORY_STANDBY'],
            'has_history': False,
            'last_active_decision': self.DECISIONS['SYSTEM_STANDBY']
        }

        # ============ SPPVT状态（Python端维护） ============
        self.sppvt_state = {
            'stage_offset': 0.0,
            'stage': 1.0,
            'error_sign': 0.0,
            'upgrade_count': 0.0,
            'prev_error': 0.0,
            'prev_velocity': 0.0,
            'prev_accel': 0.0,
            'upgrade_cooldown': 0,  # 升级冷却计数器
            'prev_G2_s': 2.0,       # 记录上一帧的G2参数（用于突变检测）
            'prev_error_abs': 0.0   # 记录上一帧的误差绝对值（用于突变检测）
        }

        # ============ ACC参数（Python端管理，传递给Simulink） ============
        self.params = {
            'V_target_kmh': 50.0,
            'V_min_kmh': 30.0,
            'G2_s': 2.0
        }

        # ============ 控制状态 ============
        self._last_control_enabled = False  # 上一帧的control_enabled（用于当前帧SPPVT）
        self.torque_arbitration_active = False

        # ============ 性能统计 ============
        self.call_count = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0

        # ============ 创建UDP客户端 ============
        self.udp_client = SimulinkUDPClient(
            send_port=self.config.integrated_udp_send_port,
            recv_port=self.config.integrated_udp_recv_port,
            num_inputs=9,   # Decision 4个 + SPPVT 5个
            num_outputs=10, # Decision 5个 + SPPVT 5个
            timeout=self.config.integrated_udp_timeout,
            debug=debug,
            local_send_port=self.config.integrated_udp_local_send_port,  # 绑定固定源端口
            send_initial_packet=True,
            initial_values=self._get_initial_inputs()
        )

        if self.debug:
            print("✅ 统一Simulink管理器已初始化")
            print(f"   UDP: 发送→{self.config.integrated_udp_send_port}(源端口{self.config.integrated_udp_local_send_port}), 接收←{self.config.integrated_udp_recv_port}")
            print(f"   输入: 9个double, 输出: 10个double")

    def _get_initial_inputs(self) -> list:
        """获取初始输入值（用于首次UDP握手）"""
        return [
            # Decision输入 (1-4)
            float(self.decision_state['current_state']),        # 2.0 (S2无史待命)
            0.0,                                                 # command_type = NONE
            float(1 if self.decision_state['has_history'] else 0),  # 0.0
            float(self.decision_state['last_active_decision']), # 8.0 (R8待命)

            # SPPVT输入 (5-9)
            0.0,  # error_value
            0.0,  # stage_offset
            0.0,  # prev_error
            0.0,  # prev_velocity
            0.0   # prev_accel
        ]

    # ================================================================
    # 公共API：与ACCControlFacade接口兼容
    # ================================================================

    def process_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理一个控制周期（主入口）

        替代原来的并发调用decision和sppvt，现在统一调用一次Simulink

        Args:
            input_data: 输入数据字典，包含：
                - ego_speed_kmh: 自车速度
                - command_type: 键盘指令
                - control_error: 控制误差
                - control_mode_flag: 控制模式
                - V_target_kmh, V_min_kmh, G2_s: ACC参数
                - manual_throttle_active: 手动油门激活
                - 等

        Returns:
            统一输出字典，与原ACCControlFacade输出格式兼容
        """
        start_time = time.time()
        self.call_count += 1

        # 1. 数据清洗和验证
        sanitized_input = self._sanitize_input(input_data)

        # 2. Python端参数调整和低速检测
        self._update_parameters(sanitized_input)
        self._handle_low_speed_transition(sanitized_input['ego_speed_kmh'])

        # 3. 准备Simulink输入
        simulink_inputs = self._prepare_simulink_inputs(sanitized_input)

        # 4. 调用Simulink（单次UDP通信）
        simulink_outputs = self._call_simulink(simulink_inputs)

        # 5. 解析Simulink输出并更新Python端状态
        integrated_output = self._process_simulink_outputs(
            simulink_outputs,
            sanitized_input
        )

        # 6. 性能统计
        duration = time.time() - start_time
        self.last_processing_time = duration
        self.total_processing_time += duration

        if self.debug and self.call_count % 20 == 0:
            print(f"[IntegratedManager] call={self.call_count}, time={duration*1000:.2f}ms, "
                  f"state=S{self.decision_state['current_state']}, "
                  f"decision=R{integrated_output['current_decision']}")

        return integrated_output

    def process_decision_and_control(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """兼容性别名，供acc_updated.py调用"""
        return self.process_cycle(input_data)

    # ================================================================
    # 内部方法：数据准备和处理
    # ================================================================

    def _sanitize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """数据清洗和默认值填充"""
        sanitized = {}

        for key, value in input_data.items():
            if isinstance(value, (float, int)):
                sanitized[key] = 0.0 if value is None or not np.isfinite(value) else float(value)
            else:
                sanitized[key] = value

        # 默认值
        defaults = {
            'ego_speed_kmh': 0.0,
            'ego_speed_ms': 0.0,
            'control_error': 0.0,
            'control_mode_flag': 1,
            'command_type': 0,
            'command_active': False,
            'manual_throttle_active': False,
            'V_target_kmh': self.params['V_target_kmh'],
            'V_min_kmh': self.params['V_min_kmh'],
            'G2_s': self.params['G2_s'],
        }

        for key, default in defaults.items():
            sanitized.setdefault(key, default)

        return sanitized

    def _update_parameters(self, input_data: Dict[str, Any]):
        """
        Python端参数调整（E/Q/R/T键的参数修改）

        与原acc_decision_simulink_manager的逻辑保持一致
        """
        command_type = input_data.get('command_type', 0)
        ego_speed_kmh = input_data.get('ego_speed_kmh', 0.0)

        if command_type in [1, 2, 3, 4]:  # 参数调整指令
            speed_step = 5.0
            time_gap_step = 0.2
            in_active_control = (self.decision_state['current_state'] == self.STATES['ACTIVE_CONTROL'])

            if command_type == 1:  # E键: 降速 or 当速启控
                if in_active_control:
                    new_target = max(self.params['V_min_kmh'], self.params['V_target_kmh'] - speed_step)
                    self.params['V_target_kmh'] = new_target
                    if self.debug:
                        print(f"⌨️ E键降速: → {new_target:.1f} km/h")

            elif command_type == 2:  # Q键: 增速 or 继承启控
                if in_active_control:
                    new_target = min(self.max_target_speed_kmh, self.params['V_target_kmh'] + speed_step)
                    self.params['V_target_kmh'] = new_target
                    if self.debug:
                        print(f"⌨️ Q键增速: → {new_target:.1f} km/h")

            elif command_type == 3:  # T键: 降距
                new_gap = max(1.0, self.params['G2_s'] - time_gap_step)
                self.params['G2_s'] = new_gap
                if self.debug:
                    print(f"⌨️ T键降距: → {new_gap:.1f} s")

            elif command_type == 4:  # R键: 增距
                new_gap = min(5.0, self.params['G2_s'] + time_gap_step)
                self.params['G2_s'] = new_gap
                if self.debug:
                    print(f"⌨️ R键增距: → {new_gap:.1f} s")

    def _handle_low_speed_transition(self, ego_speed_kmh: float):
        """
        Python端低速检测（安全相关，保留在Python）
        """
        if ego_speed_kmh < self.params['V_min_kmh']:
            if self.decision_state['current_state'] != self.STATES['LOW_SPEED']:
                self.decision_state['current_state'] = self.STATES['LOW_SPEED']
                if self.debug:
                    print(f"🚗 低速状态: {ego_speed_kmh:.1f} < {self.params['V_min_kmh']:.1f} km/h")
        else:
            # 从S3恢复
            if self.decision_state['current_state'] == self.STATES['LOW_SPEED']:
                if self.decision_state['has_history']:
                    self.decision_state['current_state'] = self.STATES['ADAPTIVE_HISTORY_STANDBY']
                    if self.debug:
                        print("🚗 恢复到S1有史待命")
                else:
                    self.decision_state['current_state'] = self.STATES['ADAPTIVE_NO_HISTORY_STANDBY']
                    if self.debug:
                        print("🚗 恢复到S2无史待命")

    def _prepare_simulink_inputs(self, input_data: Dict[str, Any]) -> list:
        """
        准备Simulink输入（9个double）

        关键：使用上一帧的control_enabled来决定是否传递control_error给SPPVT
        """
        # 根据上一帧的control_enabled过滤error_value
        if self._last_control_enabled:
            error_value = input_data['control_error']
        else:
            error_value = 0.0  # SPPVT不工作

        inputs = [
            # Decision输入 (1-4)
            float(self.decision_state['current_state']),
            float(input_data['command_type']),
            float(1 if self.decision_state['has_history'] else 0),
            float(self.decision_state['last_active_decision']),

            # SPPVT输入 (5-9)
            error_value,
            self.sppvt_state['stage_offset'],
            self.sppvt_state['prev_error'],
            self.sppvt_state['prev_velocity'],
            self.sppvt_state['prev_accel']
        ]

        return inputs

    def _call_simulink(self, inputs: list) -> list:
        """
        调用Simulink模型（单次UDP通信）

        Returns:
            10个double的输出
        """
        start_time = time.time()

        try:
            # 双调用策略：解决Simulink UDP延迟问题
            self.udp_client.call(inputs)  # 预热调用，丢弃结果
            outputs = self.udp_client.call(inputs)  # 实际调用

            elapsed_ms = (time.time() - start_time) * 1000.0

            if self.debug and self.call_count % 20 == 0:
                print(f"✅ Simulink UDP: {elapsed_ms:.1f}ms")

            return outputs

        except Exception as e:
            if self.debug:
                print(f"❌ Simulink UDP失败: {e}")

            # 返回安全的默认输出
            return [
                float(self.decision_state['current_state']),  # next_state
                float(self.DECISIONS['SYSTEM_STANDBY']),      # decision
                0.0,  # control_enabled
                float(1 if self.decision_state['has_history'] else 0),  # next_has_history
                float(self.decision_state['last_active_decision']),     # next_last_decision
                0.0, 0.0, 0.0, 0.0, 0.0  # SPPVT输出全零
            ]

    def _process_simulink_outputs(self, outputs: list, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析Simulink输出并更新Python端状态

        Args:
            outputs: Simulink的10个输出
            input_data: 原始输入数据（用于构建完整输出）

        Returns:
            与ACCControlFacade兼容的输出字典
        """
        # 解析Decision输出 (1-5)
        next_state = int(round(outputs[0]))
        current_decision = int(round(outputs[1]))
        control_enabled = bool(int(round(outputs[2])))
        next_has_history = bool(int(round(outputs[3])))
        next_last_decision = int(round(outputs[4]))

        # 解析SPPVT输出 (6-10)
        sppvt_control_output = outputs[5]
        sppvt_velocity_output = outputs[6]
        sppvt_acceleration_output = outputs[7]
        sppvt_jerk_output = outputs[8]
        sppvt_should_upgrade = outputs[9]

        # 更新Decision状态
        self.decision_state['current_state'] = next_state
        self.decision_state['has_history'] = next_has_history
        self.decision_state['last_active_decision'] = next_last_decision

        # === SPPVT阶段管理（Python侧维护）===
        current_error = input_data['control_error']
        current_error_abs = abs(current_error)

        # ============================================================
        # 🆕 SPPVT阶段重置条件检测（在符号翻转检测之前）
        # ============================================================
        should_reset = False
        reset_reason = ""

        # 条件1: G2参数突变检测
        current_G2_s = self.params['G2_s']
        prev_G2_s = self.sppvt_state.get('prev_G2_s', current_G2_s)
        G2_change = abs(current_G2_s - prev_G2_s)

        G2_CHANGE_THRESHOLD = 0.3  # 时距变化阈值：0.3秒

        if G2_change > G2_CHANGE_THRESHOLD:
            should_reset = True
            reset_reason = f"G2突变: {prev_G2_s:.1f}s → {current_G2_s:.1f}s (Δ{G2_change:.1f}s)"

        # 条件2: 时距误差突变检测（仅在TIME模式下）
        if not should_reset and input_data.get('control_mode_flag') == 1:  # TIME模式
            prev_error_abs = self.sppvt_state.get('prev_error_abs', 0.0)

            ERROR_JUMP_RATIO = 5.0  # 误差跳变倍数阈值

            if prev_error_abs > 0.0:  # 避免除零
                error_ratio = current_error_abs / prev_error_abs

                if error_ratio > ERROR_JUMP_RATIO:
                    should_reset = True
                    reset_reason = f"误差突变: {prev_error_abs:.2f}s → {current_error_abs:.2f}s (×{error_ratio:.1f}倍)"

        # 执行重置
        if should_reset:
            self.sppvt_state['stage'] = 1.0
            self.sppvt_state['stage_offset'] = 0.0
            self.sppvt_state['upgrade_count'] = 0.0
            self.sppvt_state['upgrade_cooldown'] = 0

            if self.debug:
                print(f"\n🔄 [SPPVT重置] 帧#{self.call_count} | {reset_reason} | Stage→1")

        # 更新历史记录
        self.sppvt_state['prev_G2_s'] = current_G2_s
        self.sppvt_state['prev_error_abs'] = current_error_abs

        # ============================================================
        # 原有的符号翻转检测逻辑
        # ============================================================
        if abs(current_error) < 1e-6:
            current_sign = 0
        elif current_error > 0:
            current_sign = 1
        else:
            current_sign = -1

        prev_sign = int(self.sppvt_state['error_sign'])
        sign_changed = (prev_sign != 0 and current_sign != 0 and prev_sign != current_sign)

        if sign_changed:
            # 符号翻转：重置阶段累计
            self.sppvt_state['stage'] = 1.0
            self.sppvt_state['stage_offset'] = 0.0
            self.sppvt_state['upgrade_count'] = 0.0
            self.sppvt_state['upgrade_cooldown'] = 0  # 重置冷却
        elif sppvt_should_upgrade > 0.5 and self.sppvt_state['upgrade_cooldown'] == 0:
            # 升级触发（仅在冷却结束后）：阶段+1并累加offset
            old_stage = int(self.sppvt_state['stage'])
            self.sppvt_state['stage'] += 1.0
            self.sppvt_state['upgrade_count'] += 1.0
            offset_delta = self.sppvt_rho * abs(current_error)
            if current_error > 0:
                self.sppvt_state['stage_offset'] += offset_delta
            else:
                self.sppvt_state['stage_offset'] -= offset_delta
            self.sppvt_state['stage_offset'] = max(-100.0, min(100.0, self.sppvt_state['stage_offset']))

            # 设置冷却期
            self.sppvt_state['upgrade_cooldown'] = self.upgrade_cooldown_frames

            # 调试输出
            if self.debug:
                print(f"\n🔼 [SPPVT升级] 帧#{self.call_count} | Stage {old_stage} → {int(self.sppvt_state['stage'])} | "
                      f"冷却:{self.upgrade_cooldown_frames}帧")

        # 每帧减少冷却计数器
        if self.sppvt_state['upgrade_cooldown'] > 0:
            self.sppvt_state['upgrade_cooldown'] -= 1

        # 更新误差符号历史
        if current_sign != 0:
            self.sppvt_state['error_sign'] = float(current_sign)

        # 更新SPPVT输入状态（使用当前帧的输出作为下一帧的输入）
        self.sppvt_state['prev_error'] = current_error
        self.sppvt_state['prev_velocity'] = sppvt_velocity_output
        self.sppvt_state['prev_accel'] = sppvt_acceleration_output

        # 更新control_enabled（供下一帧使用）
        self._last_control_enabled = control_enabled

        # 更新扭矩仲裁标志
        self.torque_arbitration_active = (current_decision == self.DECISIONS['TORQUE_ARBITRATION'])

        # 构建输出字典（与ACCControlFacade格式兼容）
        integrated_output = {
            # Decision输出
            'control_enabled': control_enabled,
            'current_state': next_state,
            'current_decision': current_decision,
            'torque_arbitration_active': self.torque_arbitration_active,
            'updated_V_target_kmh': self.params['V_target_kmh'],
            'updated_G2_s': self.params['G2_s'],
            'next_state': next_state,
            'next_has_history': next_has_history,
            'next_last_active_decision': next_last_decision,

            # SPPVT输出
            'sppvt_control_output': sppvt_control_output,
            'sppvt_velocity_output': sppvt_velocity_output,
            'sppvt_acceleration_output': sppvt_acceleration_output,
            'sppvt_jerk_output': sppvt_jerk_output,
            'sppvt_stage_output': self.sppvt_state['stage'],
            'sppvt_status_output': sppvt_should_upgrade,

            # SPPVT状态（供调试）
            'new_stage_offset': self.sppvt_state['stage_offset'],
            'new_stage': self.sppvt_state['stage'],
            'new_error_sign': self.sppvt_state['error_sign'],
            'new_upgrade_count': self.sppvt_state['upgrade_count'],
            'new_control_error': current_error,
            'new_error_derivative': sppvt_velocity_output,
            'new_error_second_derivative': sppvt_acceleration_output,
            'target_accel': sppvt_control_output,

            # 调试信息
            'debug_message': 0,
            'simulation_time_ms': self.last_processing_time * 1000,
        }

        return integrated_output

    # ================================================================
    # 兼容性接口：与原管理器接口保持一致
    # ================================================================

    def reset(self):
        """重置所有状态"""
        self.decision_state = {
            'current_state': self.STATES['ADAPTIVE_NO_HISTORY_STANDBY'],
            'has_history': False,
            'last_active_decision': self.DECISIONS['SYSTEM_STANDBY']
        }

        self.sppvt_state = {
            'stage_offset': 0.0,
            'stage': 1.0,
            'error_sign': 0.0,
            'upgrade_count': 0.0,
            'prev_error': 0.0,
            'prev_velocity': 0.0,
            'prev_accel': 0.0,
            'upgrade_cooldown': 0,  # 升级冷却计数器
            'prev_G2_s': self.params['G2_s'],     # 重置历史G2
            'prev_error_abs': 0.0                  # 重置历史误差
        }

        self._last_control_enabled = False
        self.torque_arbitration_active = False
        self.call_count = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0

        if self.debug:
            print("🔄 统一管理器已重置")

    def cleanup(self):
        """清理资源"""
        try:
            self.udp_client.cleanup()
            if self.debug:
                print("✅ 统一管理器已清理")
        except Exception as e:
            if self.debug:
                print(f"⚠️ 清理失败: {e}")

    # ================================================================
    # 兼容属性：供acc_updated.py的显示代码使用
    # ================================================================

    @property
    def current_state(self):
        """兼容属性：返回当前状态（枚举类型）"""
        from acc_controller import ACCState
        mapping = {
            0: ACCState.ACTIVE_CONTROL,
            1: ACCState.ADAPTIVE_HISTORY_STANDBY,
            2: ACCState.ADAPTIVE_NO_HISTORY_STANDBY,
            3: ACCState.LOW_SPEED,
        }
        return mapping.get(self.decision_state['current_state'], ACCState.ADAPTIVE_NO_HISTORY_STANDBY)
