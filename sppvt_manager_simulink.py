"""
SimulinkSPPVTManager - SPPVT控制器的Simulink实现

该管理器使用UDP通信与Simulink模型交互，替代MATLAB Engine。
继承自BaseSPPVTManager，共享状态管理逻辑。
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

from simulink_udp_interface import SimulinkUDPClient
from sppvt_manager_base import BaseSPPVTManager


class SimulinkSPPVTManager(BaseSPPVTManager):
    def __init__(self, matlab_engine=None, model_name: str = 'sppvt_control_model',
                 params: Dict[str, float] | None = None, debug: bool = False):
        """
        初始化SPPVT UDP客户端

        Args:
            matlab_engine: 保留参数用于向后兼容，实际不再使用
            model_name: 保留参数用于向后兼容
            params: SPPVT参数字典
            debug: 是否打印调试信息
        """
        super().__init__(params=params, debug=debug)
        self.model_name = model_name

        # 创建UDP客户端替代MATLAB Engine
        # 端口配置：26000（Python→Simulink），26001（Simulink→Python）
        # 输入：5个double（error, stage_offset, prev_error, prev_vel, prev_accel）
        # 输出：5个double（control, velocity, accel, jerk, should_upgrade）
        self.udp_client = SimulinkUDPClient(
            send_port=26000,
            recv_port=26001,
            num_inputs=5,   # ⚠️ 用户已从模型中删除control_mode_flag，从6个改为5个
            num_outputs=5,
            timeout=2.0,  # ⚠️ 增加到2秒，解决阻塞模式下顺序调用的超时问题
            debug=debug,
            local_send_port=9091  # 绑定固定源端口，匹配Simulink Remote port配置
        )

        if self.debug:
            print(f"✅ SPPVT UDP客户端已初始化: {self.model_name}")

    # ------------------------------------------------------------------ input preparation
    def _prepare_inputs(self, control_enabled: bool, control_error: float,
                        control_mode_flag: int) -> Tuple[float, List[float]]:
        """
        准备Simulink UDP输入（5个double）

        ⚠️ 注意：用户已从Simulink模型中删除control_mode_flag输入
        现在只有5个输入，不再是6个

        Simulink输入端口映射（基于test_sppvt_udp.py）：
            In1: error_value - 控制误差 (m)
            In2: current_stage_offset - 级差状态累积值
            In3: prev_error - 上一帧控制误差
            In4: prev_velocity - 上一帧误差导数 (m/s)
            In5: prev_accel - 上一帧误差二阶导 (m/s²)

        参数通过Simulink Constant模块提供，不在UDP输入中：
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
            # ⚠️ 不再包含control_mode_flag（In6已删除）
        ]
        return error_value, inputs

    # ------------------------------------------------------------------ processing
    def process_from_values(self, control_enabled: bool, control_error: float,
                            control_mode_flag: int) -> Dict[str, float]:
        self.call_count += 1

        error_value, inputs = self._prepare_inputs(control_enabled, control_error, control_mode_flag)
        outputs, sim_time_ms = self._run_simulink(inputs)
        stage_update = self._update_stage_state(error_value, outputs)
        return self._build_result(outputs, stage_update, simulation_time_ms=sim_time_ms)

    def _run_simulink(self, inputs: List[float]) -> tuple[list[float], float]:
        """
        通过UDP发送输入到Simulink，接收输出

        输入（5个double）：
            In1: error_value, In2: current_stage_offset, In3: prev_error,
            In4: prev_velocity, In5: prev_accel

        输出（5个double）：
            Out1: control_output, Out2: velocity_output, Out3: acceleration_output,
            Out4: jerk_output, Out5: should_upgrade

        参数通过Simulink Constant模块提供：
            SPPVT_dt, SPPVT_kp, SPPVT_max_accel, SPPVT_max_decel,
            SPPVT_delta, SPPVT_eta
        """
        start_time = time.time()

        # ⚠️ 双调用解决Simulink UDP延迟问题：
        # 第一次调用返回的是上一帧的结果，丢弃
        # 第二次调用才是当前输入对应的结果
        self.udp_client.call(inputs)  # 预热调用，丢弃结果
        outputs = self.udp_client.call(inputs)  # 实际调用，使用结果

        elapsed_ms = (time.time() - start_time) * 1000.0

        if self.debug and self.call_count % 20 == 0:
            print(f"✅ SPPVT UDP输出: control={outputs[0]:.4f}, velocity={outputs[1]:.4f}, "
                  f"accel={outputs[2]:.4f}, jerk={outputs[3]:.4f}, should_upgrade={outputs[4]:.0f}")

        return outputs, elapsed_ms

    def cleanup(self):
        """关闭UDP连接"""
        try:
            self.udp_client.cleanup()
        except Exception as e:
            if self.debug:
                print(f"⚠️ SPPVT UDP清理失败: {e}")


# Backwards compatibility
SPPVTManager = SimulinkSPPVTManager
