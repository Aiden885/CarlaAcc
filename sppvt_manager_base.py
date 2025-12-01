"""
Shared SPPVT manager utilities.
Provides the common state machine, parameter handling, and result assembly
so that different backends (pure Python vs MATLAB/Simulink) can reuse logic.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


class BaseSPPVTManager:
    """Shared functionality for all SPPVT managers."""

    def __init__(self, params: Dict[str, float] | None = None, debug: bool = False):
        self.debug = debug
        self.params = {
            'dt': 0.05,
            'kp': 1.0,
            'max_accel': 2.0,
            'max_decel': -3.0,
            'delta': 0.05,
            'eta': 0.2,
            'sppvt_rho': 0.1,
        }
        if params:
            self.params.update(params)

        self.call_count = 0
        self.reset_sppvt_state()

    def reset_sppvt_state(self):
        self.sppvt_state = {
            'stage_offset': 0.0,
            'stage': 1.0,
            'error_sign': 0.0,
            'upgrade_count': 0.0,
            'control_error': 0.0,
            'error_derivative': 0.0,
            'error_second_derivative': 0.0,
        }

    # ------------------------------------------------------------------ helpers
    def _prepare_inputs(self, control_enabled: bool, control_error: float,
                        control_mode_flag: int) -> Tuple[float, List[float]]:
        """
        准备Simulink模型的实时输入变量（参数通过Constant模块提供）

        Simulink输入端口映射：
            In1: error_value - 控制误差 (m)
            In2: current_stage_offset - 级差状态累积值
            In3: prev_error - 上一帧控制误差
            In4: prev_velocity - 上一帧误差导数 (m/s)
            In5: prev_accel - 上一帧误差二阶导 (m/s²)
            In6: control_mode_flag - 控制模式 (1=TIME, 2=SPEED)

        Constant模块参数（不在输入列表中）：
            SPPVT_dt, SPPVT_kp, SPPVT_max_accel, SPPVT_max_decel,
            SPPVT_delta, SPPVT_eta
        """
        error_value = control_error if control_enabled else 0.0
        inputs = [
            error_value,                                    # In1: 控制误差
            self.sppvt_state['stage_offset'],               # In2: current_stage_offset
            self.sppvt_state['control_error'],              # In3: prev_error
            self.sppvt_state['error_derivative'],           # In4: prev_velocity
            self.sppvt_state['error_second_derivative'],    # In5: prev_accel
            float(control_mode_flag),                       # In6: control_mode_flag
        ]
        return error_value, inputs

    def _update_stage_state(self, error_value: float, sppvt_outputs: List[float]) -> Dict[str, float]:
        if abs(error_value) < 1e-6:
            current_sign = 0
        elif error_value > 0:
            current_sign = 1
        else:
            current_sign = -1

        prev_error_sign = int(self.sppvt_state['error_sign'])
        sign_changed = (
            prev_error_sign != 0
            and current_sign != 0
            and prev_error_sign != current_sign
        )

        if sign_changed:
            self.sppvt_state['stage'] = 1.0
            self.sppvt_state['stage_offset'] = 0.0
            self.sppvt_state['upgrade_count'] = 0.0
        else:
            acceleration = sppvt_outputs[2] if len(sppvt_outputs) > 2 else 0.0
            velocity = abs(sppvt_outputs[1]) if len(sppvt_outputs) > 1 else 0.0
            should_upgrade = (
                acceleration < 0
                and velocity <= self.params['delta']
                and abs(error_value) > self.params['eta']
            )
            if should_upgrade:
                self.sppvt_state['stage'] += 1.0
                self.sppvt_state['upgrade_count'] += 1.0
                offset_delta = self.params['sppvt_rho'] * abs(error_value)
                if error_value > 0:
                    self.sppvt_state['stage_offset'] += offset_delta
                else:
                    self.sppvt_state['stage_offset'] -= offset_delta
                self.sppvt_state['stage_offset'] = max(
                    -100.0, min(100.0, self.sppvt_state['stage_offset'])
                )

        if current_sign != 0:
            self.sppvt_state['error_sign'] = float(current_sign)

        self.sppvt_state['control_error'] = error_value
        if len(sppvt_outputs) > 1:
            self.sppvt_state['error_derivative'] = sppvt_outputs[1]
        if len(sppvt_outputs) > 2:
            self.sppvt_state['error_second_derivative'] = sppvt_outputs[2]

        return {
            'new_stage_offset': self.sppvt_state['stage_offset'],
            'new_stage': self.sppvt_state['stage'],
            'new_error_sign': self.sppvt_state['error_sign'],
            'new_upgrade_count': self.sppvt_state['upgrade_count'],
            'new_control_error': self.sppvt_state['control_error'],
            'new_error_derivative': self.sppvt_state['error_derivative'],
            'new_error_second_derivative': self.sppvt_state['error_second_derivative'],
            'sign_changed': sign_changed,
        }

    def _build_result(self, sppvt_outputs: List[float], stage_update: Dict[str, float],
                      simulation_time_ms: float = 0.0) -> Dict[str, float]:
        # Ensure list has at least 5 entries
        padded = list(sppvt_outputs) + [0.0] * (5 - len(sppvt_outputs))
        control, velocity, acceleration, jerk, status = padded[:5]

        result = {
            'sppvt_control_output': control,
            'sppvt_velocity_output': velocity,
            'sppvt_acceleration_output': acceleration,
            'sppvt_jerk_output': jerk,
            'sppvt_stage_output': stage_update['new_stage'],
            'sppvt_status_output': status,
            'new_stage_offset': stage_update['new_stage_offset'],
            'new_stage': stage_update['new_stage'],
            'new_error_sign': stage_update['new_error_sign'],
            'new_upgrade_count': stage_update['new_upgrade_count'],
            'new_control_error': stage_update['new_control_error'],
            'new_error_derivative': stage_update['new_error_derivative'],
            'new_error_second_derivative': stage_update['new_error_second_derivative'],
            'target_accel': control,
            'simulation_time_ms': simulation_time_ms,
        }
        return result

    # ------------------------------------------------------------------ interface
    def process_from_values(self, control_enabled: bool, control_error: float,
                            control_mode_flag: int) -> Dict[str, float]:
        """Template method. Sub-classes should override."""
        raise NotImplementedError

    def process_sppvt_control(self, decision_output=None, validated_input=None,
                              control_enabled: bool | None = None,
                              control_error: float | None = None,
                              control_mode_flag: int | None = None) -> Dict[str, float]:
        """
        Compatibility wrapper to support both legacy callers that pass decision_output
        + validated_input and the new façade that sends raw values.
        """
        if decision_output is not None and validated_input is not None:
            control_enabled = decision_output.get('control_enabled', False)
            control_error = validated_input.get('control_error', 0.0)
            control_mode_flag = validated_input.get('control_mode_flag', 1)

        if control_enabled is None or control_error is None or control_mode_flag is None:
            raise ValueError("process_sppvt_control requires either legacy inputs "
                             "or explicit control values.")

        return self.process_from_values(control_enabled, control_error, control_mode_flag)

    def get_sppvt_state(self) -> Dict[str, float]:
        return self.sppvt_state.copy()
