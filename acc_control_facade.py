"""
Unified façade for ACC decision + SPPVT control.
Allows selecting between hybrid (Python decision + Simulink control),
pure Python fallback, or legacy Simulink pipelines while exposing a
consistent process_cycle interface.
"""
from __future__ import annotations

import time
import threading
from typing import Dict, Optional

import numpy as np

from acc_config import ACCConfig
from acc_controller import ACCState
from sppvt_manager_simulink import SimulinkSPPVTManager
from acc_decision_simulink_manager import SimulinkACCDecisionManager


class ACCControlFacade:
    MODES = ('hybrid',)

    def __init__(self, config: Optional[ACCConfig] = None, mode: str = 'hybrid',
                 debug: bool = False, matlab_engine=None, model_name: str = 'sppvt_control_model'):
        """
        初始化ACC控制外观（决策+控制）

        Args:
            config: ACC配置对象
            mode: 控制模式（当前仅支持'hybrid'）
            debug: 是否打印调试信息
            matlab_engine: 保留参数用于向后兼容，实际不再使用（现使用UDP通信）
            model_name: 保留参数用于向后兼容
        """
        if mode not in self.MODES:
            raise ValueError(f"Unsupported control mode '{mode}'.")

        self.config = config or ACCConfig()
        self.mode = mode
        self.debug = debug

        # 决策模块 - 使用UDP与Simulink通信
        self.acc_controller = SimulinkACCDecisionManager(
            matlab_engine=matlab_engine,  # 保留参数用于向后兼容
            debug=debug,
            max_target_speed_kmh=self.config.max_target_speed_kmh
        )

        if self.debug:
            print("✅ 使用Simulink决策模块（UDP通信）")

        # 初始化决策参数与配置保持一致
        try:
            self.acc_controller.params.update(self.config.acc_params)
        except Exception:
            pass

        # SPPVT控制模块 - 使用UDP与Simulink通信
        self.sppvt_manager = SimulinkSPPVTManager(
            matlab_engine=matlab_engine,  # 保留参数用于向后兼容
            model_name=model_name,
            debug=debug
        )

        self.torque_arbitration_active = False
        self.call_count = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        self._last_decision_output = None

        # 用于并发调用的上一帧control_enabled（解决阻塞模式顺序调用问题）
        self._last_control_enabled = False

    # ------------------------------------------------------------------ public API
    def process_cycle(self, input_data: Dict) -> Dict:
        """
        并发调用decision和sppvt，解决阻塞模式下顺序调用的超时问题

        策略：
        1. 使用上一帧的control_enabled来调用sppvt（避免依赖）
        2. decision和sppvt并发执行（threading）
        3. 更新control_enabled供下一帧使用

        这模仿了test_udp_concurrent.py的成功模式
        """
        start = time.time()
        self.call_count += 1

        sanitized_input = self._sanitize_input(input_data)

        # ⚠️ 并发调用：使用threading同时调用decision和sppvt
        # 这解决了阻塞模式下顺序调用导致的超时问题
        decision_output = [None]
        sppvt_output = [None]
        decision_error = [None]
        sppvt_error = [None]

        def call_decision():
            try:
                decision_output[0] = self._process_acc_decision(sanitized_input)
            except Exception as e:
                decision_error[0] = e
                if self.debug:
                    print(f"❌ Decision线程异常: {e}")

        def call_sppvt():
            try:
                # 使用上一帧的control_enabled（延迟一帧，但避免依赖）
                sppvt_output[0] = self.sppvt_manager.process_from_values(
                    control_enabled=self._last_control_enabled,
                    control_error=sanitized_input.get('control_error', 0.0),
                    control_mode_flag=sanitized_input.get('control_mode_flag', 1)
                )
            except Exception as e:
                sppvt_error[0] = e
                if self.debug:
                    print(f"❌ SPPVT线程异常: {e}")

        # 创建并启动两个线程
        t1 = threading.Thread(target=call_decision)
        t2 = threading.Thread(target=call_sppvt)

        t1.start()
        t2.start()

        # 等待两个线程完成
        t1.join()
        t2.join()

        # 检查异常
        if decision_error[0]:
            raise decision_error[0]
        if sppvt_error[0]:
            raise sppvt_error[0]

        # 获取结果
        decision_output = decision_output[0]
        sppvt_output = sppvt_output[0]

        # 更新control_enabled供下一帧使用
        self._last_control_enabled = decision_output['control_enabled']

        integrated = self._integrate_outputs(decision_output, sppvt_output)
        self._last_decision_output = decision_output

        duration = time.time() - start
        self.last_processing_time = duration
        self.total_processing_time += duration

        if self.debug and self.call_count % 20 == 0:
            print(f"[ACCControlFacade] call={self.call_count}, time={duration*1000:.2f}ms, "
                  f"state={decision_output['current_state']}, decision={decision_output['current_decision']}")

        return integrated

    def process_decision_and_control(self, input_data: Dict) -> Dict:
        """Compatibility alias for legacy callers."""
        return self.process_cycle(input_data)

    def cleanup(self):
        if hasattr(self.acc_controller, 'cleanup'):
            self.acc_controller.cleanup()
        if hasattr(self.sppvt_manager, 'cleanup'):
            self.sppvt_manager.cleanup()

    def reset(self):
        """Reset decision/controller state and performance counters."""
        if hasattr(self.acc_controller, 'reset'):
            self.acc_controller.reset()
        if hasattr(self.sppvt_manager, 'reset'):
            self.sppvt_manager.reset()
        self.torque_arbitration_active = False
        self.call_count = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        self._last_decision_output = None
        self._last_control_enabled = False

    # ------------------------------------------------------------------ helpers
    def _sanitize_input(self, input_data: Dict) -> Dict:
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
            'control_mode_flag': 1,
            'command_type': 0,
            'command_active': False,
            'manual_throttle_active': False,
            'V_target_kmh': self.config.acc_params['V_target_kmh'],
            'V_min_kmh': self.config.acc_params['V_min_kmh'],
            'G2_s': self.config.acc_params['G2_s'],
            'timestamp': time.time(),
        }
        for key, default in defaults.items():
            sanitized.setdefault(key, default)
        return sanitized

    def _process_acc_decision(self, validated_input: Dict) -> Dict:
        validated_data = self.acc_controller.validate_and_process_input(validated_input)
        control_enabled, current_decision, _ = self.acc_controller.process_keyboard_command(
            command_type=validated_data.get('command_type', 0),
            ego_speed_kmh=validated_data['ego_speed_kmh']
        )
        state_info = self.acc_controller.get_state_info()
        decision_output = {
            'control_enabled': control_enabled,
            'current_state': state_info['current_state'],
            'current_decision': current_decision,
            'torque_arbitration_active': (current_decision == 7),
            'updated_V_target_kmh': state_info['params']['V_target_kmh'],
            'updated_G2_s': state_info['params']['G2_s'],
            'debug_message': state_info.get('debug_counter', 0),
            'next_state': state_info['current_state'],
            'next_has_history': state_info['has_history'],
            'next_last_active_decision': state_info['last_active_decision'],
            'command_description': f"R{current_decision}",
        }
        self.torque_arbitration_active = decision_output['torque_arbitration_active']
        return decision_output

    def _integrate_outputs(self, decision_output: Dict, sppvt_output: Dict) -> Dict:
        integrated = {
            'control_enabled': decision_output['control_enabled'],
            'current_state': decision_output['current_state'],
            'current_decision': decision_output['current_decision'],
            'torque_arbitration_active': decision_output['torque_arbitration_active'],
            'updated_V_target_kmh': decision_output['updated_V_target_kmh'],
            'updated_G2_s': decision_output['updated_G2_s'],
            'sppvt_control_output': sppvt_output['sppvt_control_output'],
            'sppvt_velocity_output': sppvt_output['sppvt_velocity_output'],
            'sppvt_acceleration_output': sppvt_output['sppvt_acceleration_output'],
            'sppvt_jerk_output': sppvt_output.get('sppvt_jerk_output', 0.0),
            'sppvt_stage_output': sppvt_output['sppvt_stage_output'],
            'sppvt_status_output': sppvt_output['sppvt_status_output'],
            'debug_message': decision_output['debug_message'],
            'next_state': decision_output['next_state'],
            'next_has_history': decision_output['next_has_history'],
            'next_last_active_decision': decision_output['next_last_active_decision'],
            'new_stage_offset': sppvt_output['new_stage_offset'],
            'new_stage': sppvt_output['new_stage'],
            'new_error_sign': sppvt_output['new_error_sign'],
            'new_upgrade_count': sppvt_output['new_upgrade_count'],
            'new_control_error': sppvt_output['new_control_error'],
            'new_error_derivative': sppvt_output['new_error_derivative'],
            'new_error_second_derivative': sppvt_output['new_error_second_derivative'],
            'target_accel': sppvt_output['target_accel'],
            'simulation_time_ms': sppvt_output.get('simulation_time_ms', 0.0),
        }
        return integrated

    # ------------------------------------------------------------------ compatibility helpers
    @property
    def current_state(self):
        if self._last_decision_output is None:
            return ACCState.ADAPTIVE_NO_HISTORY_STANDBY

        state = self._last_decision_output.get('current_state', ACCState.ADAPTIVE_NO_HISTORY_STANDBY.value)
        if isinstance(state, ACCState):
            return state

        # 从整数值映射到枚举
        mapping = {
            0: ACCState.ACTIVE_CONTROL,
            1: ACCState.ADAPTIVE_HISTORY_STANDBY,
            2: ACCState.ADAPTIVE_NO_HISTORY_STANDBY,
            3: ACCState.LOW_SPEED,
        }
        return mapping.get(state, ACCState.ADAPTIVE_NO_HISTORY_STANDBY)
