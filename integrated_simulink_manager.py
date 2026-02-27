"""
统一的Simulink集成管理器
替代原来的双UDP架构（acc_decision_simulink_manager + sppvt_manager_simulink）
使用单一UDP接口与acc_integrated_model.slx通信

简化架构：
- Python 只传入 5 个值：command_type, ego_speed_ms, vehicle_distance, target_speed_ms, current_engine_torque
- Simulink 内部维护 Decision 状态（通过 Unit Delay 自回环）
- Simulink 内部管理 G2_s 参数（G2_Manager 子系统）
- Simulink 内部检测 reset_flag（Reset_Flag_Detector 子系统）
- Simulink 内部完成扭矩仲裁（Torque_Arbitration 子系统，R7时取max）
- Simulink 返回 2 个值：control_enabled, final_output（已含仲裁结果）
- Python 端并行维护 G2_s 副本用于 HUD 显示
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

    与acc_integrated_model.slx通信（简化接口）：
    - 输入：5个double (command_type, ego_speed_ms, vehicle_distance, target_speed_ms, current_engine_torque)
    - 输出：2个double (control_enabled, final_output)

    UDP配置：
    - Python发送 → 27000 (Simulink接收), 源端口: 9090
    - Python接收 ← 27001 (Simulink发送)
    """

    # ACC状态定义（保留用于兼容显示）
    STATES = {
        'ACTIVE_CONTROL': 0,
        'ADAPTIVE_HISTORY_STANDBY': 1,
        'ADAPTIVE_NO_HISTORY_STANDBY': 2,
        'LOW_SPEED': 3
    }

    def __init__(self, debug: bool = False, max_target_speed_kmh: Optional[float] = None, config: Optional[ACCConfig] = None):
        self.debug = debug
        self.config = config or ACCConfig()
        self.max_target_speed_kmh = max_target_speed_kmh if max_target_speed_kmh is not None else self.config.max_target_speed_kmh

        # ============ ACC参数（Python端本地副本，用于显示和V_target管理） ============
        self.params = {
            'V_target_kmh': 50.0,
            'V_min_kmh': 30.0,
            'G2_s': 4.0  # 与 Simulink G2_Manager 初始值一致
        }

        # ============ 控制状态 ============
        self._last_control_enabled = False  # 用于 V_target 的 E/Q 键门控

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
        self.udp_client = SimulinkUDPClient(
            send_port=self.config.integrated_udp_send_port,
            recv_port=self.config.integrated_udp_recv_port,
            num_inputs=5,   # command_type, ego_speed_ms, vehicle_distance, target_speed_ms, current_engine_torque
            num_outputs=2,  # control_enabled, final_output
            timeout=self.config.integrated_udp_timeout,
            debug=debug,
            local_send_port=self.config.integrated_udp_local_send_port,
            send_initial_packet=False,
        )

        if self.debug:
            print("✅ 统一Simulink管理器已初始化（简化架构）")
            print(f"   UDP: 发送→{self.config.integrated_udp_send_port}(源端口{self.config.integrated_udp_local_send_port}), 接收←{self.config.integrated_udp_recv_port}")
            print(f"   输入: 5个double (cmd, speed, dist, target_spd, engine_torque)")
            print(f"   输出: 2个double (control_enabled, final_output)")
        if self._trace_enabled:
            self._init_trace()

    def _get_initial_inputs(self) -> list:
        """获取初始输入值（用于首次UDP握手）"""
        return [
            0.0,     # command_type = NONE
            0.0,     # ego_speed_ms
            9999.0,  # vehicle_distance (无目标)
            0.0,     # target_speed_ms
            0.0,     # current_engine_torque
        ]

    # ================================================================
    # 公共API：与ACCControlFacade接口兼容
    # ================================================================

    def process_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理一个控制周期（主入口）

        Args:
            input_data: 输入数据字典

        Returns:
            统一输出字典，与原ACCControlFacade输出格式兼容
        """
        start_time = time.time()
        self.call_count += 1

        # 1. 数据清洗和验证
        sanitized_input = self._sanitize_input(input_data)

        # 2. Python端参数调整（V_target 的 E/Q 键，G2 的 T/R 键本地副本）
        self._update_parameters(sanitized_input)

        # 3. 准备Simulink输入（5个值）
        simulink_inputs = self._prepare_simulink_inputs(sanitized_input)

        # 4. 调用Simulink（单次UDP通信）
        simulink_outputs = self._call_simulink(simulink_inputs)

        # 5. 解析Simulink输出
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
                  f"ctrl={'ON' if self._last_control_enabled else 'OFF'}")

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

        defaults = {
            'ego_speed_kmh': 0.0,
            'ego_speed_ms': 0.0,
            'control_error': 0.0,
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
        Python端参数调整

        E/Q键：调整 V_target（仅在 control_enabled=True 即在控时生效）
        T/R键：调整 G2_s 本地副本（Simulink 端独立维护自己的 G2_s）
        """
        command_type = input_data.get('command_type', 0)

        if command_type in [1, 2, 3, 4]:
            speed_step = 5.0
            time_gap_step = 0.2
            # 用 control_enabled 代替 current_state == S0 判断是否在控
            in_active_control = self._last_control_enabled

            if command_type == 1:  # E键: 降速
                if in_active_control:
                    new_target = max(self.params['V_min_kmh'], self.params['V_target_kmh'] - speed_step)
                    self.params['V_target_kmh'] = new_target
                    if self.debug:
                        print(f"⌨️ E键降速: → {new_target:.1f} km/h")

            elif command_type == 2:  # Q键: 增速
                if in_active_control:
                    new_target = min(self.max_target_speed_kmh, self.params['V_target_kmh'] + speed_step)
                    self.params['V_target_kmh'] = new_target
                    if self.debug:
                        print(f"⌨️ Q键增速: → {new_target:.1f} km/h")

            elif command_type == 3:  # T键: 降距（本地副本，Simulink独立处理）
                new_gap = max(1.0, self.params['G2_s'] - time_gap_step)
                self.params['G2_s'] = new_gap
                if self.debug:
                    print(f"⌨️ T键降距: → {new_gap:.1f} s")

            elif command_type == 4:  # R键: 增距（本地副本，Simulink独立处理）
                new_gap = min(5.0, self.params['G2_s'] + time_gap_step)
                self.params['G2_s'] = new_gap
                if self.debug:
                    print(f"⌨️ R键增距: → {new_gap:.1f} s")

    def _prepare_simulink_inputs(self, input_data: Dict[str, Any]) -> list:
        """
        准备Simulink输入（5个double）

        1: command_type     - 键盘指令（0=无, 1=E, 2=Q, 3=T, 4=R, 5=W, 6=S, 7=取消）
        2: ego_speed_ms     - 自车速度 (m/s)
        3: vehicle_distance - 两车距离 (m)，无目标时 9999
        4: target_speed_ms  - 前车速度 (m/s)，显示用
        5: current_engine_torque - 当前油门对应的发动机扭矩 (N·m)
                                   Simulink Y0_Latch 在启控瞬间锁存为 Y0
                                   Simulink Torque_Arbitration 在 R7 时与 control_output 取 max
        """
        command_type = float(input_data.get('command_type', 0))
        speed_ms = input_data.get('ego_speed_ms', input_data.get('ego_speed_kmh', 0.0) / 3.6)
        vehicle_distance = input_data.get('vehicle_distance', 9999.0)
        target_speed_ms = float(input_data.get('target_speed_ms', 0.0))
        current_engine_torque = float(input_data.get('current_engine_torque_nm', 0.0))

        # Save for trace
        self._last_trace_inputs = {
            'command_type': command_type,
            'ego_speed_ms': speed_ms,
            'vehicle_distance': vehicle_distance,
            'target_speed_ms': target_speed_ms,
            'current_engine_torque': current_engine_torque,
        }

        return [
            command_type,
            speed_ms,
            vehicle_distance,
            target_speed_ms,
            current_engine_torque,
        ]

    def _call_simulink(self, inputs: list) -> list:
        """
        调用Simulink模型（单次UDP通信）

        Returns:
            2个double: [control_enabled, final_output]
        """
        start_time = time.time()

        try:
            outputs = self.udp_client.call(inputs)

            elapsed_ms = (time.time() - start_time) * 1000.0
            self._last_udp_ms = elapsed_ms

            if self.debug and self.call_count % 20 == 0:
                print(f"✅ Simulink UDP: {elapsed_ms:.1f}ms")

            return outputs

        except Exception as e:
            if self.debug:
                print(f"❌ Simulink UDP失败: {e}")

            # 安全默认输出
            return [0.0, 0.0]  # control_enabled=False, final_output=0

    def _process_simulink_outputs(self, outputs: list, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析Simulink输出

        简化架构（2个输出）：
        1. control_enabled  - ACC是否在控 (0/1)
        2. final_output     - 最终控制扭矩 (N·m，已含R7仲裁)
        """
        control_enabled = bool(int(round(outputs[0])))
        control_output_nm = outputs[1]

        # 更新 control_enabled（供 V_target E/Q 键门控使用）
        self._last_control_enabled = control_enabled

        # 当前帧输入信息（用于输出显示）
        current_error = input_data.get('control_error', 0.0)
        current_engine_torque = float(input_data.get('current_engine_torque_nm', 0.0))

        # 构建输出字典（保持与旧接口兼容，缺失字段给安全默认值）
        integrated_output = {
            # 核心输出（来自 Simulink）
            'control_enabled': control_enabled,
            'control_output': control_output_nm,

            # 兼容旧接口（Simulink不再输出这些，给默认值供 output_formatter 安全访问）
            'current_state': 0 if control_enabled else 2,  # S0 在控 / S2 待命（近似）
            'current_decision': 0,          # 不再可知
            'torque_arbitration_active': False,  # 已在 Simulink 内完成
            'next_has_history': False,
            'next_last_active_decision': 0,

            # Python端参数
            'updated_V_target_kmh': self.params['V_target_kmh'],
            'updated_G2_s': self.params['G2_s'],

            # 兼容旧字段名
            'sppvt_control_output': control_output_nm,
            'target_torque': control_output_nm,
            'new_control_error': current_error,
            'current_engine_torque': current_engine_torque,

            # 调试
            'debug_message': 0,
            'simulation_time_ms': self.last_processing_time * 1000,
        }

        # Trace IO
        if self._trace_enabled:
            self._trace_io(input_data, integrated_output)

        return integrated_output

    # ================================================================
    # 兼容性接口
    # ================================================================

    def reset(self):
        """重置所有状态"""
        self._last_control_enabled = False
        self.call_count = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        self._last_cycle_ts = None

        if self.debug:
            print("🔄 统一管理器已重置（简化架构）")

    def reset_sppvt_state(self, reason: str = ""):
        """
        重置SPPVT相关状态

        简化架构下，reset_flag 由 Simulink Reset_Flag_Detector 自行检测
        （control_enabled下降沿 + G2突变），Python 端不再主动发送 reset_flag。
        此方法保留接口兼容性。

        # [保留] 未来如需恢复 Python 端 reset 触发（如目标丢失、模式切换），
        # 可在此添加逻辑，并在输入向量中增加 reset_flag 位。
        """
        if self.debug:
            msg = f"🔄 SPPVT重置请求（简化架构下由Simulink自行检测）"
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
    # IO Trace
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
                    # 输入 (5)
                    'command_type_in',
                    'ego_speed_ms_in',
                    'vehicle_distance_in',
                    'target_speed_ms_in',
                    'current_engine_torque_in',
                    # 输出 (2)
                    'control_enabled_out',
                    'final_output_out',
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
            ti = self._last_trace_inputs

            row = [
                f"{now:.6f}",
                os.getpid(),
                self._instance_id,
                frame_id,
                f"{cycle_dt_ms:.3f}",
                f"{self._last_udp_ms:.3f}",
                int(bool(input_data.get('acc_system_enabled', False))),
                # 输入
                ti.get('command_type', 0),
                f"{ti.get('ego_speed_ms', 0.0):.6f}",
                f"{ti.get('vehicle_distance', 9999.0):.6f}",
                f"{ti.get('target_speed_ms', 0.0):.6f}",
                f"{ti.get('current_engine_torque', 0.0):.6f}",
                # 输出
                int(bool(output_data.get('control_enabled', False))),
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
        """兼容属性：根据 control_enabled 近似返回状态"""
        from acc_controller import ACCState
        if self._last_control_enabled:
            return ACCState.ACTIVE_CONTROL
        else:
            return ACCState.ADAPTIVE_NO_HISTORY_STANDBY
