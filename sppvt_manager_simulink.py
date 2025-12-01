"""
SimulinkSPPVTManager - SPPVT控制器的Simulink实现

该管理器使用MATLAB Simulink引擎来执行SPPVT控制算法。
继承自BaseSPPVTManager，共享状态管理逻辑。
"""
from __future__ import annotations

import time
from typing import Dict, List

try:
    import matlab  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    matlab = None

from matlab_engine_factory import get_matlab_engine
from sppvt_manager_base import BaseSPPVTManager


class SimulinkSPPVTManager(BaseSPPVTManager):
    def __init__(self, matlab_engine=None, model_name: str = 'sppvt_control_model',
                 params: Dict[str, float] | None = None, debug: bool = False):
        super().__init__(params=params, debug=debug)
        self.model_name = model_name
        self.matlab_engine = matlab_engine or get_matlab_engine(force=False)
        self.model_loaded = False

    # ------------------------------------------------------------------ engine management
    def initialize_matlab_engine(self):
        if self.matlab_engine is None:
            self.matlab_engine = get_matlab_engine()
        return self.matlab_engine is not None

    def _ensure_model_loaded(self):
        if self.matlab_engine is None:
            if not self.initialize_matlab_engine():
                raise RuntimeError("MATLAB engine is not available.")

        if not self.model_loaded:
            # 加载模型
            self.matlab_engine.load_system(self.model_name, nargout=0)

            # 同步Python参数到Simulink Constant模块
            self._sync_params_to_constant_blocks()

            # 配置仿真参数
            self.matlab_engine.set_param(self.model_name, 'SimulationMode', 'normal', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'StopTime', str(self.params['dt']), nargout=0)
            self.matlab_engine.set_param(self.model_name, 'SaveOutput', 'on', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'OutputSaveName', 'yout', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'SaveFormat', 'Structure', nargout=0)
            self.model_loaded = True

    def _sync_params_to_constant_blocks(self):
        """
        将Python端的参数同步到Simulink模型中的Constant模块

        基于检测结果，模型中有6个Constant模块：
            SPPVT_dt, SPPVT_kp, SPPVT_max_accel, SPPVT_max_decel,
            SPPVT_delta, SPPVT_eta

        参考：https://stackoverflow.com/questions/64285280
        """
        if self.matlab_engine is None:
            raise RuntimeError("MATLAB engine not initialized")

        # 定义参数到Constant模块的映射
        # 格式：'Constant模块名': (Python参数名, 描述)
        param_blocks = {
            'SPPVT_dt': ('dt', '控制周期(s)'),
            'SPPVT_kp': ('kp', '比例系数'),
            'SPPVT_max_accel': ('max_accel', '最大加速度(m/s²)'),
            'SPPVT_max_decel': ('max_decel', '最大减速度(m/s²)'),
            'SPPVT_delta': ('delta', '速度阈值'),
            'SPPVT_eta': ('eta', '误差阈值'),
        }

        for block_name, (param_name, description) in param_blocks.items():
            param_value = self.params[param_name]

            # 构建完整的块路径
            block_path = f'{self.model_name}/{block_name}'

            # 设置Constant模块的Value参数（值必须转为字符串）
            try:
                self.matlab_engine.set_param(
                    block_path,
                    'Value',
                    str(param_value),  # 必须是字符串格式
                    nargout=0
                )

                if self.debug:
                    print(f"✅ {description}: {block_name} = {param_value}")

            except Exception as e:
                print(f"❌ 设置{block_name}失败: {e}")
                # 尝试验证块是否存在
                try:
                    current_value = self.matlab_engine.get_param(block_path, 'Value', nargout=1)
                    print(f"   当前值: {current_value}")
                except:
                    print(f"   块路径可能不存在: {block_path}")

    # ------------------------------------------------------------------ processing
    def process_from_values(self, control_enabled: bool, control_error: float,
                            control_mode_flag: int) -> Dict[str, float]:
        self.call_count += 1

        error_value, inputs = self._prepare_inputs(control_enabled, control_error, control_mode_flag)
        outputs, sim_time_ms = self._run_simulink(inputs)
        stage_update = self._update_stage_state(error_value, outputs)
        return self._build_result(outputs, stage_update, simulation_time_ms=sim_time_ms)

    def _run_simulink(self, inputs: List[float]) -> tuple[list[float], float]:
        self._ensure_model_loaded()
        if matlab is None:
            raise RuntimeError("matlab python package is required for Simulink SPPVT manager.")

        # Build external input matrix (time + 6 real-time signals)
        # 输入信号（基于检测结果）：
        #   In1: error_value, In2: current_stage_offset, In3: prev_error,
        #   In4: prev_velocity, In5: prev_accel, In6: control_mode_flag
        # 参数（通过Constant模块提供）：
        #   SPPVT_dt, SPPVT_kp, SPPVT_max_accel, SPPVT_max_decel,
        #   SPPVT_delta, SPPVT_eta
        dt = self.params['dt']
        ext_input = [
            [0.0] + inputs,
            [dt] + inputs,
        ]
        matlab_ext_input = matlab.double(ext_input) if matlab else None

        # Push to workspace and configure SimulationInput
        start_time = time.time()
        self.matlab_engine.workspace['ext_input'] = matlab_ext_input
        sim_in = self.matlab_engine.eval(f"Simulink.SimulationInput('{self.model_name}')", nargout=1)
        sim_in = self.matlab_engine.setExternalInput(sim_in, 'ext_input', nargout=1)
        sim_out = self.matlab_engine.sim(sim_in, nargout=1)
        elapsed_ms = (time.time() - start_time) * 1000.0

        # Extract outputs (control, velocity, acceleration, jerk, should_upgrade, extra)
        self.matlab_engine.workspace['sim_out'] = sim_out
        control_output = float(self.matlab_engine.eval('sim_out.yout.signals(1).values(end)', nargout=1))
        velocity_output = float(self.matlab_engine.eval('sim_out.yout.signals(2).values(end)', nargout=1))
        acceleration_output = float(self.matlab_engine.eval('sim_out.yout.signals(3).values(end)', nargout=1))
        jerk_output = float(self.matlab_engine.eval('sim_out.yout.signals(4).values(end)', nargout=1))
        should_upgrade_flag = float(self.matlab_engine.eval('sim_out.yout.signals(5).values(end)', nargout=1))
        # Some models output com1-3; reuse com1 as status indicator when present
        status_output = should_upgrade_flag
        if self.debug:
            print(f"✅ Simulink outputs: control={control_output:.4f}, velocity={velocity_output:.4f}, "
                  f"accel={acceleration_output:.4f}, jerk={jerk_output:.4f}")

        outputs = [control_output, velocity_output, acceleration_output, jerk_output, status_output]
        return outputs, elapsed_ms

    def cleanup(self):
        if self.matlab_engine and self.model_loaded:
            try:
                self.matlab_engine.close_system(self.model_name, 0, nargout=0)
            except Exception:
                pass


# Backwards compatibility
SPPVTManager = SimulinkSPPVTManager
