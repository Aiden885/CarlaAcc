"""
统一的ACC控制外观（支持集成模式）

支持集成模式（单次UDP同时完成Decision+SPPVT），已弃用旧hybrid双UDP架构
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from acc_config import ACCConfig
from acc_controller import ACCState


class ACCControlFacade:
    MODES = ('integrated',)

    def __init__(self, config: Optional[ACCConfig] = None, mode: str = 'integrated',
                 debug: bool = False, matlab_engine=None, model_name: str = 'sppvt_control_model'):
        """
        初始化ACC控制外观（决策+控制）

        Args:
            config: ACC配置对象
            mode: 控制模式（仅保留'integrated'）
            debug: 是否打印调试信息
            matlab_engine: 保留参数用于向后兼容
            model_name: 保留参数用于向后兼容
        """
        if mode not in self.MODES:
            raise ValueError(f"Unsupported control mode '{mode}'. Choose from {self.MODES}")

        self.config = config or ACCConfig()
        self.mode = mode
        self.debug = debug

        from integrated_simulink_manager import IntegratedSimulinkManager

        self.manager = IntegratedSimulinkManager(
            debug=debug,
            max_target_speed_kmh=self.config.max_target_speed_kmh,
            config=self.config
        )

        # 初始化参数
        self.manager.params.update(self.config.acc_params)

        if self.debug:
            print("✅ 使用统一Simulink管理器 (integrated模式)")

        # 通用状态
        self.torque_arbitration_active = False
        self.call_count = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0

    # ================================================================
    # 公共API
    # ================================================================

    def process_cycle(self, input_data: Dict) -> Dict:
        """
        处理一个控制周期（主入口）

        根据mode调用不同的实现
        """
        start = time.time()
        self.call_count += 1

        # 统一管理器：单次调用
        result = self.manager.process_cycle(input_data)

        # 更新性能统计
        duration = time.time() - start
        self.last_processing_time = duration
        self.total_processing_time += duration

        return result

    def process_decision_and_control(self, input_data: Dict) -> Dict:
        """兼容性别名"""
        return self.process_cycle(input_data)

    def cleanup(self):
        """清理资源"""
        self.manager.cleanup()

    def reset(self):
        """重置状态"""
        self.manager.reset()

        self.torque_arbitration_active = False
        self.call_count = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0

    def reset_sppvt_state(self, reason: str = ""):
        """仅重置SPPVT状态（不影响Decision状态）"""
        self.manager.reset_sppvt_state(reason=reason)

    # ================================================================
    # 兼容属性
    # ================================================================

    @property
    def current_state(self):
        """兼容属性：返回当前状态"""
        return self.manager.current_state
