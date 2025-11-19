"""
SPPVT管理器 - 高内聚设计
整合SPPVT状态管理 + Simulink调用 + 阶段管理
"""
import numpy as np
import matlab
import matlab.engine
from typing import Dict, Any, Tuple, Optional, List


class SPPVTManager:
    """
    SPPVT完整管理器
    - 管理SPPVT状态（级差、阶段、升级计数等）
    - 封装Simulink调用逻辑
    - 整合Stage_Manager功能
    - 提供简化的外部接口
    """
    
    def __init__(self, model_path: str = 'sppvt_control_model', debug: bool = False):
        self.model_path = model_path
        self.debug = debug
        
        # SPPVT状态
        self.sppvt_state = {
            'stage_offset': 0.0,        # 级差值
            'stage': 1.0,               # 当前阶段
            'error_sign': 0.0,          # 误差符号
            'upgrade_count': 0.0,       # 升级计数
            'control_error': 0.0,       # 控制误差
            'error_derivative': 0.0,    # 控制误差导数
            'error_second_derivative': 0.0  # 控制误差二阶导数
        }
        
        # MATLAB引擎初始化
        self.matlab_engine = None
        self._init_matlab_engine()
        
        # 固定参数
        self.sppvt_params = {
            'dt': 0.05,           # 时间步长
            'kp': 1.0,            # 比例系数
            'max_accel': 2.0,     # 最大加速度
            'max_decel': -3.0,    # 最大减速度
            'delta': 0.05,        # SPPVT delta参数
            'eta': 0.2,           # SPPVT eta参数
            'sppvt_rho': 0.1      # 级差计算系数
        }
        
        # 调试计数器
        self.call_count = 0
        
    def _init_matlab_engine(self):
        """初始化MATLAB引擎"""
        try:
            if self.debug:
                print("🔧 启动MATLAB引擎...")
            self.matlab_engine = matlab.engine.start_matlab()
            
            # 设置工作目录
            import os
            current_dir = os.getcwd()
            self.matlab_engine.cd(current_dir)
            
            if self.debug:
                print(f"✅ MATLAB引擎启动成功，工作目录: {current_dir}")
                
        except Exception as e:
            print(f"❌ MATLAB引擎启动失败: {e}")
            self.matlab_engine = None
            
    def reset(self):
        """重置SPPVT状态"""
        self.sppvt_state = {
            'stage_offset': 0.0,
            'stage': 1.0,
            'error_sign': 0.0,
            'upgrade_count': 0.0,
            'control_error': 0.0,
            'error_derivative': 0.0,
            'error_second_derivative': 0.0
        }
        self.call_count = 0
        
        if self.debug:
            print("🔄 SPPVT状态已重置")
    
    def _prepare_sppvt_inputs(self, control_enabled: bool, control_error: float, control_mode_flag: int) -> List[float]:
        """
        准备SPPVT控制器输入 - 整合Adapter功能
        
        Args:
            control_enabled: 控制是否激活
            control_error: 控制误差
            control_mode_flag: 控制模式标志
            
        Returns:
            List[float]: 12个SPPVT输入信号
        """
        # 确定实际误差值
        if control_enabled:
            error_value = control_error
        else:
            error_value = 0.0
            
        # 构造12个输入信号（对应sppvt_adapter_code.m的输出）
        inputs = [
            error_value,                                    # [1] 控制误差
            self.sppvt_params['dt'],                       # [2] 时间步长
            self.sppvt_state['stage_offset'],              # [3] 级差
            self.sppvt_params['kp'],                       # [4] 比例系数
            self.sppvt_params['max_accel'],                # [5] 最大加速度
            self.sppvt_params['max_decel'],                # [6] 最大减速度
            self.sppvt_state['control_error'],             # [7] 上次误差
            self.sppvt_state['error_derivative'],          # [8] 上次导数
            self.sppvt_state['error_second_derivative'],   # [9] 上次二阶导数
            self.sppvt_params['delta'],                    # [10] delta参数
            self.sppvt_params['eta'],                      # [11] eta参数
            float(control_mode_flag)                       # [12] 控制模式标志
        ]
        
        if self.debug and abs(error_value) > 0.01:
            print(f"🎯 SPPVT输入: Error={error_value:.3f}, Stage={self.sppvt_state['stage']:.1f}, "
                  f"Offset={self.sppvt_state['stage_offset']:.3f}, Mode={control_mode_flag}")
                  
        return inputs
    
    def _update_stage_manager(self, error_value: float, sppvt_outputs: List[float]) -> Dict[str, float]:
        """
        更新阶段管理器状态 - 整合Stage_Manager功能
        
        Args:
            error_value: 当前控制误差
            sppvt_outputs: SPPVT控制器输出
            
        Returns:
            Dict: 更新后的阶段状态
        """
        # 计算当前误差符号
        if abs(error_value) < 1e-6:
            current_sign = 0
        elif error_value > 0:
            current_sign = 1
        else:
            current_sign = -1
            
        # 误差符号变化检测
        prev_error_sign = int(self.sppvt_state['error_sign'])
        sign_changed = False
        
        if (prev_error_sign != 0 and current_sign != 0 and 
            prev_error_sign != current_sign):
            # 符号变化：重置状态
            sign_changed = True
            self.sppvt_state['stage'] = 1.0
            self.sppvt_state['stage_offset'] = 0.0
            self.sppvt_state['upgrade_count'] = 0.0
            
            if self.debug:
                print(f"🔄 误差符号变化 ({prev_error_sign} → {current_sign})，重置SPPVT状态")
        else:
            # 升级条件检查
            # 假设从SPPVT输出中提取升级条件信息
            # sppvt_outputs: [control, velocity, acceleration, jerk, status]
            if len(sppvt_outputs) >= 3:
                acceleration = sppvt_outputs[2]
                velocity = abs(sppvt_outputs[1])
                
                # 升级条件: (acceleration < 0) && (|velocity| <= delta) && (|error| > eta)
                should_upgrade = (acceleration < 0 and 
                                velocity <= self.sppvt_params['delta'] and 
                                abs(error_value) > self.sppvt_params['eta'])
                
                if should_upgrade and not sign_changed:
                    # 阶段升级
                    self.sppvt_state['stage'] += 1
                    self.sppvt_state['upgrade_count'] += 1
                    
                    # 更新级差
                    if error_value > 0:
                        self.sppvt_state['stage_offset'] += self.sppvt_params['sppvt_rho'] * abs(error_value)
                    else:
                        self.sppvt_state['stage_offset'] -= self.sppvt_params['sppvt_rho'] * abs(error_value)
                    
                    # 限制级差范围
                    self.sppvt_state['stage_offset'] = np.clip(self.sppvt_state['stage_offset'], -100.0, 100.0)
                    
                    if self.debug:
                        print(f"📈 SPPVT阶段升级: Stage={self.sppvt_state['stage']:.0f}, "
                              f"Offset={self.sppvt_state['stage_offset']:.3f}")
        
        # 更新误差符号历史
        if current_sign != 0:
            self.sppvt_state['error_sign'] = float(current_sign)
            
        # 更新误差状态
        self.sppvt_state['control_error'] = error_value
        if len(sppvt_outputs) >= 3:
            self.sppvt_state['error_derivative'] = sppvt_outputs[1]      # velocity作为一阶导数
            self.sppvt_state['error_second_derivative'] = sppvt_outputs[2]  # acceleration作为二阶导数
        
        return {
            'new_stage_offset': self.sppvt_state['stage_offset'],
            'new_stage': self.sppvt_state['stage'],
            'new_error_sign': self.sppvt_state['error_sign'],
            'new_upgrade_count': self.sppvt_state['upgrade_count'],
            'new_control_error': self.sppvt_state['control_error'],
            'new_error_derivative': self.sppvt_state['error_derivative'],
            'new_error_second_derivative': self.sppvt_state['error_second_derivative'],
            'sign_changed': sign_changed
        }
    
    def process_sppvt_control(self, control_enabled: bool, control_error: float, 
                             control_mode_flag: int) -> Dict[str, Any]:
        """
        处理SPPVT控制 - 主要接口函数
        
        Args:
            control_enabled: 控制是否激活
            control_error: 控制误差
            control_mode_flag: 控制模式标志
            
        Returns:
            Dict: SPPVT处理结果
        """
        self.call_count += 1
        
        try:
            # 1. 准备SPPVT输入
            sppvt_inputs = self._prepare_sppvt_inputs(control_enabled, control_error, control_mode_flag)
            
            # 2. 调用Simulink模型
            if self.matlab_engine is not None:
                # 转换为MATLAB数组格式
                matlab_inputs = [matlab.double([inp]) for inp in sppvt_inputs]
                
                # 调用Simulink仿真
                start_time = 0.0
                stop_time = self.sppvt_params['dt']
                
                # 设置仿真输入
                sim_input = self.matlab_engine.Simulink.SimulationInput(self.model_path)
                
                # 这里需要根据实际的sppvt_control_model.slx的输入端口名称调整
                # 假设有12个输入端口，分别命名为In1, In2, ..., In12
                for i, inp in enumerate(matlab_inputs):
                    port_name = f'In{i+1}'
                    sim_input = sim_input.setExternalInput(f'{port_name}', inp)
                
                # 运行仿真
                sim_output = self.matlab_engine.sim(sim_input)
                
                # 提取输出（假设有5个输出：control, velocity, acceleration, jerk, status）
                sppvt_outputs = [
                    float(sim_output['simout'].signals[0].values[-1]),  # control output
                    float(sim_output['simout'].signals[1].values[-1]),  # velocity output  
                    float(sim_output['simout'].signals[2].values[-1]),  # acceleration output
                    float(sim_output['simout'].signals[3].values[-1]),  # jerk output
                    float(sim_output['simout'].signals[4].values[-1])   # status output
                ]
                
            else:
                # 备用计算（简化版本）
                if self.debug:
                    print("⚠️ MATLAB引擎不可用，使用备用计算")
                
                # 简化的SPPVT计算
                control_output = control_error * self.sppvt_params['kp']
                control_output = np.clip(control_output, self.sppvt_params['max_decel'], self.sppvt_params['max_accel'])
                
                sppvt_outputs = [control_output, 0.0, 0.0, 0.0, 1.0]
            
            # 3. 更新阶段管理器状态
            stage_update = self._update_stage_manager(control_error, sppvt_outputs)
            
            # 4. 构造返回结果
            result = {
                'sppvt_control_output': sppvt_outputs[0],
                'sppvt_velocity_output': sppvt_outputs[1], 
                'sppvt_acceleration_output': sppvt_outputs[2],
                'sppvt_stage_output': self.sppvt_state['stage'],
                'sppvt_status_output': sppvt_outputs[4],
                
                # SPPVT状态输出
                'sppvt_state': stage_update,
                
                # 调试信息
                'debug_info': {
                    'call_count': self.call_count,
                    'inputs_count': len(sppvt_inputs),
                    'outputs_count': len(sppvt_outputs),
                    'sign_changed': stage_update.get('sign_changed', False)
                }
            }
            
            if self.debug and self.call_count % 20 == 0:
                print(f"🎯 SPPVT第{self.call_count}次调用: "
                      f"Control={sppvt_outputs[0]:.3f}, Stage={self.sppvt_state['stage']:.0f}")
                      
            return result
            
        except Exception as e:
            if self.debug:
                print(f"❌ SPPVT处理失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 返回安全的默认值
            return {
                'sppvt_control_output': 0.0,
                'sppvt_velocity_output': 0.0,
                'sppvt_acceleration_output': 0.0,
                'sppvt_stage_output': self.sppvt_state['stage'],
                'sppvt_status_output': 0.0,
                'sppvt_state': {
                    'new_stage_offset': self.sppvt_state['stage_offset'],
                    'new_stage': self.sppvt_state['stage'],
                    'new_error_sign': self.sppvt_state['error_sign'],
                    'new_upgrade_count': self.sppvt_state['upgrade_count'],
                    'new_control_error': 0.0,
                    'new_error_derivative': 0.0,
                    'new_error_second_derivative': 0.0,
                    'sign_changed': False
                },
                'debug_info': {
                    'call_count': self.call_count,
                    'error': str(e)
                }
            }
    
    def get_sppvt_state(self) -> Dict[str, float]:
        """获取当前SPPVT状态"""
        return self.sppvt_state.copy()
    
    def cleanup(self):
        """清理资源"""
        if self.matlab_engine is not None:
            try:
                self.matlab_engine.quit()
                if self.debug:
                    print("🔧 MATLAB引擎已关闭")
            except:
                pass
            self.matlab_engine = None