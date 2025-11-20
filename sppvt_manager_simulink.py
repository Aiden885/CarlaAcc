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
            self.matlab_engine.load_system(self.model_name, nargout=0)
            self.matlab_engine.set_param(self.model_name, 'SimulationMode', 'normal', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'StopTime', str(self.params['dt']), nargout=0)
            self.matlab_engine.set_param(self.model_name, 'SaveOutput', 'on', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'OutputSaveName', 'yout', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'SaveFormat', 'Structure', nargout=0)
            self.model_loaded = True

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

        # Build external input matrix (time + signals)
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
