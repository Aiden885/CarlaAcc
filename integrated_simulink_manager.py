"""
统一的Simulink集成管理器
替代原来的双UDP架构（acc_decision_simulink_manager + sppvt_manager_simulink）
使用单一UDP接口与acc_integrated_model.slx通信

设计理念（更新版 - 基于 SIMULINK_SPPVT_UPDATE_PLAN.md）：
- Python 计算 control_error_signed, Y0, reset_flag
- Simulink 维护 SPPVT 状态（stage, stage_offset, cooldown, error_sign）
- 控制方程：Y(k) = Kp * (error + stage_offset) + Y0
- control_output 直接输出 N·m，Python 不再做缩放
"""
from __future__ import annotations

import os
import time
import csv
import numpy as np
from typing import Dict, Any, Optional

from simulink_udp_interface import SimulinkUDPClient
from acc_config import ACCConfig


class IntegratedSimulinkManager:
    """
    统一的Simulink集成管理器

    与acc_integrated_model.slx通信（新接口）：
    - 输入：7个double (Decision 4个 + control_error_signed, Y0, reset_flag)
    - 输出：6个double (Decision 5个 + control_output)

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

        # ============ Decision状态（Python端维护） ============
        self.decision_state = {
            'current_state': self.STATES['ADAPTIVE_NO_HISTORY_STANDBY'],
            'has_history': False,
            'last_active_decision': self.DECISIONS['SYSTEM_STANDBY']
        }

        # ============ ACC参数（Python端管理，传递给Simulink） ============
        self.params = {
            'V_target_kmh': 50.0,
            'V_min_kmh': 30.0,
            'G2_s': 2.0
        }

        # ============ 控制状态 ============
        self._last_control_enabled = False  # 上一帧的control_enabled（用于reset_flag计算）
        self.torque_arbitration_active = False

        # ============ reset_flag 检测状态 ============
        self._prev_G2_s = self.params['G2_s']  # 上一帧G2（用于突变检测）
        self._prev_error_abs = 0.0              # 上一帧误差绝对值（用于突变检测）
        self._reset_pending = False             # 待发送的重置标志（边沿检测触发后设置）

        # ============ Y0 计算物理参数 ============
        # 从 acc_config.py 和 torque_to_throttle_converter.py 获取
        self._vehicle_mass = 2370.0  # kg (CARLA Audi e-tron)
        self._wheel_radius = 0.37    # m
        self._gear_ratio = 9.204     # 总传动比
        self._rolling_resistance_coeff = getattr(self.config, 'rolling_resistance_coeff', 0.012)
        self._aero_cdA = getattr(self.config, 'aero_cdA', 0.74)
        self._air_density = getattr(self.config, 'air_density_kg_m3', 1.225)
        self._gravity = 9.81  # m/s²

        # ============ 性能统计 ============
        self.call_count = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        self._last_cycle_ts = None
        self._last_udp_ms = 0.0

        # ============ IO Trace (optional) ============
        env_trace = bool(int(os.environ.get('ACC_SIMULINK_TRACE', '0')))
        self._trace_enabled = bool(getattr(self.config, 'enable_simulink_trace', False)) or env_trace
        self._trace_path = os.environ.get(
            'ACC_SIMULINK_TRACE_PATH',
            getattr(self.config, 'simulink_trace_path', 'simulink_io_trace.csv')
        )
        self._trace_flush_every = int(os.environ.get(
            'ACC_SIMULINK_TRACE_FLUSH',
            getattr(self.config, 'simulink_trace_flush_every', 50)
        ))
        self._trace_count = 0
        self._trace_file = None
        self._trace_writer = None
        self._last_trace_inputs = {}
        self._instance_id = id(self)

        # ============ 创建UDP客户端 ============
        # 注意：send_initial_packet=False 是关键！
        # 如果设为 True，初始包的响应不会被消费，导致后续所有响应错位一帧，引发状态振荡
        self.udp_client = SimulinkUDPClient(
            send_port=self.config.integrated_udp_send_port,
            recv_port=self.config.integrated_udp_recv_port,
            num_inputs=7,   # Decision 4个 + control_error_signed, Y0, reset_flag
            num_outputs=6,  # Decision 5个 + control_output
            timeout=self.config.integrated_udp_timeout,
            debug=debug,
            local_send_port=self.config.integrated_udp_local_send_port,
            send_initial_packet=False,  # 修复振荡问题：不发送初始包
        )

        if self.debug:
            print("✅ 统一Simulink管理器已初始化（新架构）")
            print(f"   UDP: 发送→{self.config.integrated_udp_send_port}(源端口{self.config.integrated_udp_local_send_port}), 接收←{self.config.integrated_udp_recv_port}")
            print(f"   输入: 7个double (Decision 4 + error/Y0/reset)")
            print(f"   输出: 6个double (Decision 5 + control_output)")
        if self._trace_enabled:
            self._init_trace()

    def _get_initial_inputs(self) -> list:
        """获取初始输入值（用于首次UDP握手）"""
        return [
            # Decision输入 (1-4)
            float(self.decision_state['current_state']),        # 2.0 (S2无史待命)
            0.0,                                                 # command_type = NONE
            float(1 if self.decision_state['has_history'] else 0),  # 0.0
            float(self.decision_state['last_active_decision']), # 8.0 (R8待命)

            # SPPVT输入 (5-7)
            0.0,  # control_error_signed
            0.0,  # Y0 (初始扭矩)
            1.0   # reset_flag = 1 (初始化时复位)
        ]

    def _calculate_Y0(self, speed_ms: float) -> float:
        """
        计算维持当前速度所需的基础扭矩 Y0

        基于阻力平衡：
        - 滚动阻力: F_roll = μ * m * g
        - 空气阻力: F_aero = 0.5 * ρ * CdA * v²
        - 总阻力: F_total = F_roll + F_aero
        - 发动机扭矩: Y0 = F_total * wheel_radius / gear_ratio

        Args:
            speed_ms: 当前车速 (m/s)

        Returns:
            维持速度所需的发动机扭矩 (N·m)
        """
        # 滚动阻力
        F_roll = self._rolling_resistance_coeff * self._vehicle_mass * self._gravity

        # 空气阻力
        F_aero = 0.5 * self._air_density * self._aero_cdA * (speed_ms ** 2)

        # 总阻力
        F_total = F_roll + F_aero

        # 转换为发动机扭矩
        # 车轮扭矩 = F * r，发动机扭矩 = 车轮扭矩 / 传动比
        Y0 = F_total * self._wheel_radius / self._gear_ratio

        return Y0

    def _calculate_reset_flag(self, control_enabled: bool, current_error: float) -> bool:
        """
        计算 reset_flag

        触发条件：
        1. control_enabled 从 True 变为 False（边沿检测，通过 _reset_pending 标志实现）
        2. G2 参数突变（变化超过阈值）
        3. 时距误差突变（TIME模式下）

        注意：条件1 的边沿检测在 _process_simulink_outputs 中完成，
             这里只检查 _reset_pending 标志

        Args:
            control_enabled: 上一帧的 control_enabled（来自 _last_control_enabled）
            current_error: 当前帧的控制误差

        Returns:
            是否需要复位 SPPVT 状态
        """
        # 条件1: control_enabled 从 True 变为 False（边沿检测）
        # 注意：边沿检测在 _process_simulink_outputs 中完成，设置 _reset_pending
        if self._reset_pending:
            if self.debug:
                print(f"🔄 [reset_flag] 控制失效边沿触发重置")
            self._reset_pending = False  # 清除标志，只触发一次
            return True

        # 条件2: G2 参数突变
        current_G2_s = self.params['G2_s']
        G2_change = abs(current_G2_s - self._prev_G2_s)
        G2_CHANGE_THRESHOLD = 0.3  # 时距变化阈值：0.3秒

        if G2_change > G2_CHANGE_THRESHOLD:
            if self.debug:
                print(f"🔄 [reset_flag] G2突变: {self._prev_G2_s:.1f}s → {current_G2_s:.1f}s")
            return True

        # 条件3: 误差突变
        current_error_abs = abs(current_error)
        ERROR_JUMP_RATIO = 5.0  # 误差跳变倍数阈值

        if self._prev_error_abs > 0.01:  # 避免除零和小误差噪声
            error_ratio = current_error_abs / self._prev_error_abs
            if error_ratio > ERROR_JUMP_RATIO:
                if self.debug:
                    print(f"🔄 [reset_flag] 误差突变: {self._prev_error_abs:.2f} → {current_error_abs:.2f} (×{error_ratio:.1f})")
                return True

        return False

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
        准备Simulink输入（7个double）

        新接口：
        1-4: Decision 输入 (current_state, command_type, has_history, last_active_decision)
        5: control_error_signed（Python已做符号约定）
        6: Y0（维持当前速度所需扭矩）
        7: reset_flag（是否需要复位SPPVT状态）
        """
        # 获取控制误差（TIME模式下需要反号：正误差代表过近，应输出制动）
        control_error_signed = input_data['control_error']
        if int(input_data.get('control_mode_flag', 1)) == 1:
            control_error_signed = -control_error_signed

        # 计算 Y0（基于当前车速）
        speed_ms = input_data.get('ego_speed_ms', input_data.get('ego_speed_kmh', 0.0) / 3.6)
        Y0 = self._calculate_Y0(speed_ms)

        # 计算 reset_flag
        reset_flag = self._calculate_reset_flag(self._last_control_enabled, control_error_signed)

        # 更新历史记录（用于下一帧的reset_flag计算）
        self._prev_G2_s = self.params['G2_s']
        self._prev_error_abs = abs(control_error_signed)

        # Save for trace (actual values sent to Simulink)
        self._last_trace_inputs = {
            'current_state': float(self.decision_state['current_state']),
            'command_type': float(input_data['command_type']),
            'has_history': float(1 if self.decision_state['has_history'] else 0),
            'last_active_decision': float(self.decision_state['last_active_decision']),
            'control_error_signed': control_error_signed,
            'Y0': Y0,
            'reset_flag': float(1 if reset_flag else 0),
        }

        inputs = [
            # Decision输入 (1-4)
            float(self.decision_state['current_state']),
            float(input_data['command_type']),
            float(1 if self.decision_state['has_history'] else 0),
            float(self.decision_state['last_active_decision']),

            # SPPVT输入 (5-7)
            control_error_signed,
            Y0,
            float(1 if reset_flag else 0)
        ]

        return inputs

    def _call_simulink(self, inputs: list) -> list:
        """
        调用Simulink模型（单次UDP通信）

        Returns:
            6个double的输出 (Decision 5个 + control_output)
        """
        start_time = time.time()

        try:
            # 单次调用（移除了导致缓冲区错位的双调用策略）
            outputs = self.udp_client.call(inputs)

            elapsed_ms = (time.time() - start_time) * 1000.0
            self._last_udp_ms = elapsed_ms

            if self.debug and self.call_count % 20 == 0:
                print(f"✅ Simulink UDP: {elapsed_ms:.1f}ms")

            return outputs

        except Exception as e:
            if self.debug:
                print(f"❌ Simulink UDP失败: {e}")

            # 返回安全的默认输出（6个值）
            return [
                float(self.decision_state['current_state']),  # next_state
                float(self.DECISIONS['SYSTEM_STANDBY']),      # decision
                0.0,  # control_enabled
                float(1 if self.decision_state['has_history'] else 0),  # next_has_history
                float(self.decision_state['last_active_decision']),     # next_last_decision
                0.0   # control_output (N·m)
            ]

    def _process_simulink_outputs(self, outputs: list, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析Simulink输出并更新Python端状态

        新架构（6个输出）：
        1. next_state
        2. decision
        3. control_enabled
        4. next_has_history
        5. next_last_decision
        6. control_output (N·m，直接使用，无需缩放)

        Args:
            outputs: Simulink的6个输出
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

        # 解析SPPVT输出 (6) - 现在只有一个：control_output (N·m)
        control_output_nm = outputs[5]

        # 更新Decision状态
        self.decision_state['current_state'] = next_state
        self.decision_state['has_history'] = next_has_history
        self.decision_state['last_active_decision'] = next_last_decision

        # 边沿检测：control_enabled 从 True 变为 False 时，设置 _reset_pending
        if self._last_control_enabled and not control_enabled:
            self._reset_pending = True
            if self.debug:
                print(f"🔄 [边沿检测] control_enabled 下降沿: True → False, 设置 _reset_pending")

        # 更新control_enabled（供下一帧边沿检测使用）
        self._last_control_enabled = control_enabled

        # 更新扭矩仲裁标志
        self.torque_arbitration_active = (current_decision == self.DECISIONS['TORQUE_ARBITRATION'])

        # 当前帧输入信息（用于输出和调试）
        current_error = input_data['control_error']
        speed_ms = input_data.get('ego_speed_ms', input_data.get('ego_speed_kmh', 0.0) / 3.6)
        Y0 = self._calculate_Y0(speed_ms)

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

            # SPPVT输出（简化版）
            'control_output': control_output_nm,  # 直接扭矩输出 (N·m)
            'Y0': Y0,  # 基础扭矩（供调试）

            # 兼容旧接口（部分字段保留，值可能为0）
            'sppvt_control_output': control_output_nm,  # 兼容旧字段名
            'target_torque': control_output_nm,         # 兼容旧字段名
            'new_control_error': current_error,

            # 调试信息
            'debug_message': 0,
            'simulation_time_ms': self.last_processing_time * 1000,
        }

        # Trace IO
        if self._trace_enabled:
            self._trace_io(input_data, integrated_output)

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

        # reset_flag 检测状态
        self._prev_G2_s = self.params['G2_s']
        self._prev_error_abs = 0.0
        self._reset_pending = False

        self._last_control_enabled = False
        self.torque_arbitration_active = False
        self.call_count = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        self._last_cycle_ts = None

        if self.debug:
            print("🔄 统一管理器已重置（新架构）")

    def reset_sppvt_state(self, reason: str = ""):
        """
        重置SPPVT相关状态

        新架构下，SPPVT 状态由 Simulink 维护。
        这个方法现在只重置 Python 端的 reset_flag 检测状态，
        并在下一帧通过 reset_flag=1 通知 Simulink 重置。
        """
        # 重置 reset_flag 检测状态
        self._prev_G2_s = self.params['G2_s']
        self._prev_error_abs = 0.0

        # 强制下一帧发送 reset_flag=1（通过设置 _reset_pending 标志）
        self._reset_pending = True

        if self.debug:
            msg = f"🔄 SPPVT重置请求（将通过reset_flag通知Simulink）"
            if reason:
                msg += f" | {reason}"
            print(msg)

    def cleanup(self):
        """清理资源"""
        try:
            self.udp_client.cleanup()
            if self._trace_file:
                self._trace_file.flush()
                self._trace_file.close()
            if self.debug:
                print("✅ 统一管理器已清理")
        except Exception as e:
            if self.debug:
                print(f"⚠️ 清理失败: {e}")

    # ================================================================
    # IO Trace helpers
    # ================================================================

    def _init_trace(self):
        try:
            is_new = not os.path.exists(self._trace_path)
            self._trace_file = open(self._trace_path, 'a', newline='')
            self._trace_writer = csv.writer(self._trace_file)
            if is_new:
                self._trace_writer.writerow([
                    'ts_wall',
                    'pid',
                    'instance_id',
                    'frame_id',
                    'cycle_dt_ms',
                    'udp_ms',
                    'acc_system_enabled',
                    'current_state_in',
                    'command_type_in',
                    'has_history_in',
                    'last_active_decision_in',
                    'control_error_signed_in',
                    'Y0_in',
                    'reset_flag_in',
                    'control_mode_flag',
                    'next_state_out',
                    'decision_out',
                    'control_enabled_out',
                    'next_has_history_out',
                    'next_last_decision_out',
                    'control_output_out'
                ])
        except Exception as e:
            if self.debug:
                print(f"⚠️ Trace init failed: {e}")
            self._trace_enabled = False

    def _trace_io(self, input_data: Dict[str, Any], output_data: Dict[str, Any]):
        try:
            now = time.time()
            if self._last_cycle_ts is None:
                cycle_dt_ms = 0.0
            else:
                cycle_dt_ms = (now - self._last_cycle_ts) * 1000.0
            self._last_cycle_ts = now

            frame_id = input_data.get('frame_id', input_data.get('frame_count', -1))
            control_error_signed = self._last_trace_inputs.get('control_error_signed', 0.0)
            Y0 = self._last_trace_inputs.get('Y0', 0.0)
            reset_flag = self._last_trace_inputs.get('reset_flag', 0.0)

            row = [
                f"{now:.6f}",
                os.getpid(),
                self._instance_id,
                frame_id,
                f"{cycle_dt_ms:.3f}",
                f"{self._last_udp_ms:.3f}",
                int(bool(input_data.get('acc_system_enabled', False))),
                self._last_trace_inputs.get('current_state', self.decision_state['current_state']),
                self._last_trace_inputs.get('command_type', input_data.get('command_type', 0)),
                self._last_trace_inputs.get('has_history', 0),
                self._last_trace_inputs.get('last_active_decision', self.decision_state['last_active_decision']),
                f"{control_error_signed:.6f}",
                f"{Y0:.6f}",
                reset_flag,
                input_data.get('control_mode_flag', 1),
                output_data.get('current_state', -1),
                output_data.get('current_decision', -1),
                int(bool(output_data.get('control_enabled', False))),
                int(bool(output_data.get('next_has_history', False))),
                output_data.get('next_last_active_decision', -1),
                f"{output_data.get('control_output', 0.0):.6f}",
            ]
            self._trace_writer.writerow(row)
            self._trace_count += 1
            if self._trace_count % self._trace_flush_every == 0:
                self._trace_file.flush()
        except Exception as e:
            if self.debug:
                print(f"⚠️ Trace write failed: {e}")
            self._trace_enabled = False

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
