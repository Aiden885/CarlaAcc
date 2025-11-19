"""
混合式ACC控制器
Python决策 + Simulink SPPVT控制的统一接口
替代原有的acc_decision_sppvt_interface.py
"""

import time
import numpy as np
from typing import Dict, Optional
from acc_controller import ACCController
from sppvt_manager_python import SPPVTManager


class ACCHybridController:
    """
    混合式ACC控制器
    - Python端：决策逻辑、输入验证、状态管理
    - Simulink端：SPPVT控制算法
    """
    
    def __init__(self, debug=False, matlab_engine=None, model_name='sppvt_control_model',
                 max_target_speed_kmh: float = 150.0):
        # 核心组件
        self.acc_controller = ACCController(max_target_speed_kmh=max_target_speed_kmh)
        self.sppvt_manager = SPPVTManager(matlab_engine, model_name)
        
        # 配置
        self.debug = debug
        self.torque_arbitration_active = False
        
        # 性能统计
        self.call_count = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        
    def process_decision_and_control(self, input_data: Dict) -> Dict:
        """
        统一的决策和控制处理接口
        
        Args:
            input_data (Dict): 输入数据，包含车辆状态、指令等
            
        Returns:
            Dict: 统一的输出，包含决策结果和SPPVT控制输出
        """
        start_time = time.time()
        self.call_count += 1
        
        try:
            # 1. 数据验证和清洗
            validated_input = self._sanitize_input(input_data)
            
            # 2. ACC决策处理
            decision_output = self._process_acc_decision(validated_input)
            
            # 3. SPPVT控制处理
            sppvt_output = self.sppvt_manager.process_sppvt_control(decision_output, validated_input)
            
            # 4. 整合输出
            unified_output = self._integrate_outputs(decision_output, sppvt_output, validated_input)
            
            # 5. 性能统计
            processing_time = time.time() - start_time
            self.last_processing_time = processing_time
            self.total_processing_time += processing_time
            
            # 6. 调试输出
            if self.debug:
                self._print_debug_info(validated_input, unified_output, processing_time)
            
            return unified_output
            
        except Exception as e:
            print(f"❌ 混合控制器处理失败: {e}")
            # 返回安全的默认输出
            return self._get_safe_default_output()
    
    def _sanitize_input(self, input_data: Dict) -> Dict:
        """数据清洗：将Inf/NaN替换为默认值"""
        def sanitize_value(value, default=0.0):
            if value is None or not np.isfinite(value):
                return default
            return value
        
        sanitized = {}
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                sanitized[key] = sanitize_value(value, 0.0)
            else:
                sanitized[key] = value
        
        # 确保必需字段存在
        required_defaults = {
            'ego_speed_kmh': 0.0,
            'ego_speed_ms': 0.0,
            'command_type': 0,
            'command_active': False,
            'manual_throttle_active': False,
            'control_error': 0.0,
            'control_mode_flag': 1,
            'V_target_kmh': 50.0,
            'V_min_kmh': 30.0,
            'G2_s': 2.0,
            'timestamp': time.time()
        }
        
        for key, default in required_defaults.items():
            if key not in sanitized:
                sanitized[key] = default
        
        return sanitized
    
    def _process_acc_decision(self, validated_input: Dict) -> Dict:
        """处理ACC决策逻辑"""
        # 1. 验证和处理输入
        validated_data = self.acc_controller.validate_and_process_input(validated_input)
        
        # 2. 处理键盘指令
        control_enabled, current_decision, param_updates = self.acc_controller.process_keyboard_command(
            command_type=validated_data.get('command_type', 0),
            ego_speed_kmh=validated_data['ego_speed_kmh']
        )
        
        # 3. 获取状态信息
        state_info = self.acc_controller.get_state_info()
        
        # 4. 构造决策输出
        decision_output = {
            'control_enabled': control_enabled,
            'current_state': state_info['current_state'],
            'current_decision': current_decision,
            'torque_arbitration_active': (current_decision == 7),  # R7: 扭矩仲裁
            'updated_V_target_kmh': state_info['params']['V_target_kmh'],
            'updated_G2_s': state_info['params']['G2_s'],
            'debug_message': state_info.get('debug_counter', 0),
            'next_state': state_info['current_state'],
            'next_has_history': state_info['has_history'],
            'next_last_active_decision': state_info['last_active_decision'],
            'command_description': f"R{current_decision}: {self._get_decision_name(current_decision)}"
        }
        
        return decision_output
    
    def _get_decision_name(self, decision: int) -> str:
        """获取决策名称"""
        decision_names = {
            1: "适速减速", 2: "适速增速", 3: "时距增长", 4: "时距减少", 
            5: "距离模式", 6: "速度模式", 7: "扭矩仲裁", 8: "系统待命"
        }
        return decision_names.get(decision, "未知决策")
    
    def _integrate_outputs(self, decision_output: Dict, sppvt_output: Dict, input_data: Dict) -> Dict:
        """整合决策输出和SPPVT输出"""
        # 基于原有的DecisionSPPVTOutputExtended总线结构
        integrated = {
            # 1-6: 决策系统信息
            'control_enabled': decision_output['control_enabled'],
            'current_state': decision_output['current_state'], 
            'current_decision': decision_output['current_decision'],
            'torque_arbitration_active': decision_output['torque_arbitration_active'],
            'updated_V_target_kmh': decision_output['updated_V_target_kmh'],
            'updated_G2_s': decision_output['updated_G2_s'],
            
            # 7-11: SPPVT控制输出
            'sppvt_control_output': sppvt_output['sppvt_control_output'],
            'sppvt_velocity_output': sppvt_output['sppvt_velocity_output'], 
            'sppvt_acceleration_output': sppvt_output['sppvt_acceleration_output'],
            'sppvt_stage_output': sppvt_output['sppvt_stage_output'],
            'sppvt_status_output': sppvt_output['sppvt_status_output'],
            
            # 12: 调试信息
            'debug_message': decision_output['debug_message'],
            
            # 13-15: 决策状态输出
            'next_state': decision_output['next_state'],
            'next_has_history': decision_output['next_has_history'],
            'next_last_active_decision': decision_output['next_last_active_decision'],
            
            # 16-22: SPPVT状态输出
            'new_stage_offset': sppvt_output['new_stage_offset'],
            'new_stage': sppvt_output['new_stage'],
            'new_error_sign': sppvt_output['new_error_sign'],
            'new_upgrade_count': sppvt_output['new_upgrade_count'],
            'new_control_error': sppvt_output['new_control_error'],
            'new_error_derivative': sppvt_output['new_error_derivative'],
            'new_error_second_derivative': sppvt_output['new_error_second_derivative'],
            
            # 额外的便利字段
            'target_accel': sppvt_output['target_accel'],  # 主控制输出
            'command_description': decision_output.get('command_description', None),
            'simulation_time_ms': sppvt_output.get('simulation_time_ms', 0.0)
        }
        
        # 更新扭矩仲裁状态
        self.torque_arbitration_active = decision_output['torque_arbitration_active']
        
        return integrated
    
    def _get_safe_default_output(self) -> Dict:
        """获取安全的默认输出"""
        return {
            'control_enabled': False,
            'current_state': 2,  # S2: 适速无史待命
            'current_decision': 8,  # R8: 系统待命
            'torque_arbitration_active': False,
            'updated_V_target_kmh': 50.0,
            'updated_G2_s': 2.0,
            'sppvt_control_output': 0.0,
            'sppvt_velocity_output': 0.0,
            'sppvt_acceleration_output': 0.0,
            'sppvt_stage_output': 1.0,
            'sppvt_status_output': 0.0,
            'debug_message': 9999,  # 错误代码
            'next_state': 2,
            'next_has_history': False,
            'next_last_active_decision': 8,
            'new_stage_offset': 0.0,
            'new_stage': 1.0,
            'new_error_sign': 0.0,
            'new_upgrade_count': 0.0,
            'new_control_error': 0.0,
            'new_error_derivative': 0.0,
            'new_error_second_derivative': 0.0,
            'target_accel': 0.0,
            'command_description': None,
            'simulation_time_ms': 0.0
        }
    
    def _print_debug_info(self, input_data: Dict, output: Dict, processing_time: float):
        """打印调试信息"""
        if self.call_count % 20 == 0:  # 每20次调用输出一次
            print(f"\n🔧 混合控制器调试信息 (第{self.call_count}次调用)")
            print(f"   输入: 车速={input_data['ego_speed_kmh']:.1f}km/h, "
                  f"指令={input_data['command_type']}, 误差={input_data['control_error']:.3f}")
            print(f"   决策: 状态=S{output['current_state']}, 决策=R{output['current_decision']}, "
                  f"控制={'ON' if output['control_enabled'] else 'OFF'}")
            print(f"   SPPVT: 阶段={output['sppvt_stage_output']:.0f}, "
                  f"输出={output['sppvt_control_output']:.3f}, 级差={output['new_stage_offset']:.3f}")
            print(f"   性能: 处理时间={processing_time*1000:.1f}ms, "
                  f"平均={self.total_processing_time/self.call_count*1000:.1f}ms")
    
    def reset(self):
        """重置控制器状态"""
        self.acc_controller.reset()
        self.sppvt_manager.reset_sppvt_state()
        self.torque_arbitration_active = False
        self.call_count = 0
        self.total_processing_time = 0.0
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if self.call_count > 0:
            avg_time = self.total_processing_time / self.call_count
        else:
            avg_time = 0.0
            
        return {
            'call_count': self.call_count,
            'total_time': self.total_processing_time,
            'average_time_ms': avg_time * 1000,
            'last_time_ms': self.last_processing_time * 1000
        }
    
    def init_two_mode_controller(self):
        """初始化两模式控制器（兼容性接口）"""
        # 保持与原接口的兼容性
        pass
    
    def cleanup(self):
        """清理资源"""
        self.sppvt_manager.cleanup()
    
    # 兼容性属性和方法
    @property  
    def debug(self):
        return self._debug
    
    @debug.setter
    def debug(self, value):
        self._debug = value
        self.acc_controller.debug = value if hasattr(self.acc_controller, 'debug') else False
        self.sppvt_manager.debug = value

    @property
    def current_state(self):
        """返回当前ACC状态（兼容旧接口）"""
        # 返回ACCState枚举，而不是整数
        from acc_decision import ACCState
        state_mapping = {
            0: ACCState.IN_CONTROL,                    # S0: 在控
            1: ACCState.ADAPTIVE_HISTORY_STANDBY,      # S1: 适速有史待命
            2: ACCState.ADAPTIVE_NO_HISTORY_STANDBY,   # S2: 适速无史待命
            3: ACCState.LOW_SPEED                      # S3: 低速状态
        }
        return state_mapping.get(self.acc_controller.current_state, ACCState.ADAPTIVE_NO_HISTORY_STANDBY)
