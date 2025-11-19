"""
统一ACC接口 - 简化设计
整合ACC控制器和SPPVT管理器，提供简洁的外部接口
替代原有的复杂总线结构
"""
import numpy as np
from typing import Dict, Any, Optional
from acc_controller import ACCController
from sppvt_manager import SPPVTManager


class UnifiedACCInterface:
    """
    统一ACC接口
    - 整合ACCController和SPPVTManager
    - 提供简化的输入输出接口
    - 替代原有的复杂Simulink总线结构
    - 高度封装，低耦合设计
    """
    
    def __init__(self, sppvt_model_path: str = 'sppvt_control_model', debug: bool = False,
                 max_target_speed_kmh: float = 150.0):
        self.debug = debug
        
        # 初始化子模块
        self.acc_controller = ACCController(debug=debug, max_target_speed_kmh=max_target_speed_kmh)
        self.sppvt_manager = SPPVTManager(model_path=sppvt_model_path, debug=debug)
        
        # 统一状态管理
        self.system_enabled = False
        self.torque_arbitration_active = False
        
        if debug:
            print("✅ 统一ACC接口初始化完成")
    
    def reset(self):
        """重置整个ACC系统"""
        self.acc_controller.reset()
        self.sppvt_manager.reset()
        self.system_enabled = False
        self.torque_arbitration_active = False
        
        if self.debug:
            print("🔄 统一ACC系统已重置")
    
    def process_control_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理一个控制周期 - 主要接口
        
        Args:
            input_data: 输入数据，包含：
                - ego_speed_kmh: 自车速度
                - ego_speed_ms: 自车速度(m/s)  
                - control_error: 控制误差
                - control_mode_flag: 控制模式标志
                - command_type: 键盘指令类型
                - manual_throttle_active: 手动油门是否激活
                - V_target_kmh: 目标速度
                - V_min_kmh: 最小速度
                - G2_s: 时距参数
                - timestamp: 时间戳
                
        Returns:
            Dict: 处理结果，包含控制输出和状态信息
        """
        try:
            # 1. 输入验证和预处理
            validated_input = self.acc_controller.validate_and_process_input(input_data)
            
            # 2. ACC决策处理
            control_enabled, current_decision, param_updates = self.acc_controller.process_keyboard_command(
                command_type=validated_input.get('command_type', 0),
                ego_speed_kmh=validated_input['ego_speed_kmh']
            )
            
            # 3. 更新系统状态
            if control_enabled:
                self.system_enabled = True
            elif current_decision == 8:  # R8: 系统待命
                self.system_enabled = False
                
            # 4. 扭矩仲裁处理
            self.torque_arbitration_active = (current_decision == 7)  # R7: 扭矩仲裁
            
            # 5. SPPVT控制处理（只在系统使能且控制激活时）
            sppvt_result = {'sppvt_control_output': 0.0, 'sppvt_velocity_output': 0.0, 'sppvt_acceleration_output': 0.0}
            
            if self.system_enabled and control_enabled:
                sppvt_result = self.sppvt_manager.process_sppvt_control(
                    control_enabled=True,
                    control_error=validated_input.get('control_error', 0.0),
                    control_mode_flag=validated_input.get('control_mode_flag', 1)
                )
            else:
                # 控制未激活时，仍调用SPPVT但误差为0
                sppvt_result = self.sppvt_manager.process_sppvt_control(
                    control_enabled=False,
                    control_error=0.0,
                    control_mode_flag=validated_input.get('control_mode_flag', 1)
                )
            
            # 6. 获取状态信息
            acc_state = self.acc_controller.get_state_info()
            sppvt_state = self.sppvt_manager.get_sppvt_state()
            
            # 7. 构造统一输出
            result = {
                # 控制状态
                'control_enabled': control_enabled and self.system_enabled,
                'current_state': acc_state['current_state'],
                'current_decision': current_decision,
                'torque_arbitration_active': self.torque_arbitration_active,
                
                # 更新后的参数
                'updated_V_target_kmh': acc_state['params']['V_target_kmh'],
                'updated_G2_s': acc_state['params']['G2_s'],
                
                # SPPVT输出
                'sppvt_control_output': sppvt_result.get('sppvt_control_output', 0.0),
                'sppvt_velocity_output': sppvt_result.get('sppvt_velocity_output', 0.0),
                'sppvt_acceleration_output': sppvt_result.get('sppvt_acceleration_output', 0.0),
                'sppvt_stage_output': sppvt_result.get('sppvt_stage_output', 1.0),
                'sppvt_status_output': sppvt_result.get('sppvt_status_output', 0.0),
                
                # 调试信息
                'debug_message': self.acc_controller.update_debug_counter(),
                
                # 状态信息（用于状态持久化，如果需要的话）
                'acc_state': acc_state,
                'sppvt_state': sppvt_state,
                
                # 系统状态
                'system_enabled': self.system_enabled
            }
            
            # 调试输出
            if self.debug and hasattr(self.acc_controller, 'debug_counter') and self.acc_controller.debug_counter % 20 == 0:
                print(f"🎮 统一接口第{self.acc_controller.debug_counter}次调用:")
                print(f"   系统使能: {self.system_enabled}")
                print(f"   控制激活: {control_enabled}")
                print(f"   当前状态: S{acc_state['current_state']} ({acc_state['state_name']})")
                print(f"   当前决策: R{current_decision}")
                print(f"   SPPVT输出: {sppvt_result.get('sppvt_control_output', 0.0):.3f}")
                
            return result
            
        except Exception as e:
            if self.debug:
                print(f"❌ 统一接口处理失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 返回安全的默认值
            return self._get_safe_default_output()
    
    def _get_safe_default_output(self) -> Dict[str, Any]:
        """获取安全的默认输出"""
        return {
            'control_enabled': False,
            'current_state': 2,  # S2: 无史待命
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
            'system_enabled': False
        }
    
    def get_current_parameters(self) -> Dict[str, float]:
        """获取当前ACC参数"""
        return self.acc_controller.params.copy()
    
    def get_status_info(self) -> Dict[str, Any]:
        """获取详细状态信息"""
        acc_state = self.acc_controller.get_state_info()
        sppvt_state = self.sppvt_manager.get_sppvt_state()
        
        return {
            'system_enabled': self.system_enabled,
            'acc_state': acc_state,
            'sppvt_state': sppvt_state,
            'torque_arbitration_active': self.torque_arbitration_active
        }
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'sppvt_manager'):
            self.sppvt_manager.cleanup()
        
        if self.debug:
            print("🔧 统一ACC接口资源已清理")


# 兼容性接口，替代原有的ACCDecisionSPPVTInterface
class ACCDecisionSPPVTInterface(UnifiedACCInterface):
    """
    兼容性接口类
    保持与原有代码的接口兼容性
    """
    
    def __init__(self, debug: bool = False, use_realtime_sppvt: bool = True,
                 max_target_speed_kmh: float = 150.0):
        # 忽略use_realtime_sppvt参数，因为新架构总是实时的
        super().__init__(debug=debug, max_target_speed_kmh=max_target_speed_kmh)
        
        # 兼容性属性
        self.torque_arbitration_active = False
        
    def process_decision_and_control(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """兼容性方法，映射到新的接口"""
        return self.process_control_cycle(input_data)
    
    def init_two_mode_controller(self):
        """兼容性方法，新架构中无需此初始化"""
        if self.debug:
            print("ℹ️ init_two_mode_controller: 新架构中无需此步骤")
        pass
