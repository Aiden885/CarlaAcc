"""
Python版本的SPPVT管理器
高内聚设计，整合SPPVT状态管理、阶段管理和Simulink调用
"""

import numpy as np
import matlab.engine
from typing import Dict, List, Optional, Tuple
import time


class SPPVTManager:
    """
    SPPVT管理器
    整合了原Simulink中的SPPVT_Adapter、Stage_Manager和Simulink调用功能
    """
    
    def __init__(self, matlab_engine=None, model_name='sppvt_control_model'):
        # Simulink接口
        self.matlab_engine = matlab_engine
        self.model_name = model_name
        self.model_loaded = False
        
        # SPPVT状态管理
        self.stage_offset = 0.0  # 级差值
        self.current_stage = 1.0  # 当前阶段
        self.error_sign = 0.0  # 误差符号
        self.upgrade_count = 0.0  # 升级计数
        self.control_error = 0.0  # 控制误差
        self.error_derivative = 0.0  # 控制误差导数  
        self.error_second_derivative = 0.0  # 控制误差二阶导数
        
        # SPPVT参数配置
        self.dt = 0.05  # 时间步长 50ms
        self.kp = 1.0  # 比例系数
        self.max_accel = 2.0  # 最大加速度
        self.max_decel = -3.0  # 最大减速度
        self.delta = 0.05  # SPPVT控制参数
        self.eta = 0.2  # SPPVT控制参数
        self.sppvt_rho = 0.1  # 级差计算参数
        
        # 调试模式
        self.debug = False
        
    def initialize_matlab_engine(self):
        """初始化MATLAB引擎和模型"""
        if self.matlab_engine is None:
            try:
                print("启动MATLAB引擎...")
                self.matlab_engine = matlab.engine.start_matlab()
                print("✅ MATLAB引擎启动成功")
            except Exception as e:
                print(f"❌ MATLAB引擎启动失败: {e}")
                return False
        
        try:
            # 加载Simulink模型
            print(f"加载Simulink模型: {self.model_name}")
            self.matlab_engine.load_system(self.model_name, nargout=0)

            # 配置模型参数
            try:
                # 设置仿真模式为normal（非加速模式）
                self.matlab_engine.set_param(self.model_name, 'SimulationMode', 'normal', nargout=0)
                # 设置停止时间
                self.matlab_engine.set_param(self.model_name, 'StopTime', str(self.dt), nargout=0)
                # 配置输出保存到workspace（使用Structure格式，因为输出端口类型不同）
                self.matlab_engine.set_param(self.model_name, 'SaveOutput', 'on', nargout=0)
                self.matlab_engine.set_param(self.model_name, 'OutputSaveName', 'yout', nargout=0)
                self.matlab_engine.set_param(self.model_name, 'SaveFormat', 'Structure', nargout=0)
                print(f"✅ 模型配置完成: 启用输出保存(Structure格式), 仿真时长={self.dt}s")
                print(f"   注意: 使用SimulationInput对象和setExternalInput方法传递Inport输入")
            except Exception as config_error:
                print(f"⚠️ 配置模型参数时出错: {config_error}")

            self.model_loaded = True
            print(f"✅ 模型 {self.model_name} 加载成功")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def _prepare_simulink_inputs(self, decision_output: Dict, validated_input: Dict) -> List[float]:
        """
        准备Simulink模型的输入数据
        基于sppvt_adapter_code.m的12个输入端口
        
        Args:
            decision_output: ACC决策输出
            validated_input: 验证后的输入数据
            
        Returns:
            List[float]: 12个输入信号的值
        """
        # 确定控制误差
        if decision_output.get('control_enabled', False):
            error_value = validated_input.get('control_error', 0.0)
        else:
            error_value = 0.0
        
        # 构造12个输入信号 (按照sppvt_adapter_code.m的输出顺序)
        simulink_inputs = [
            float(error_value),  # [1] error_value 控制误差
            float(self.dt),  # [2] dt 时间步长
            float(self.stage_offset),  # [3] stage_offset 当前级差
            float(self.kp),  # [4] kp 比例系数
            float(self.max_accel),  # [5] max_accel 最大加速度
            float(self.max_decel),  # [6] max_decel 最大减速度
            float(self.control_error),  # [7] prev_error 上次误差
            float(self.error_derivative),  # [8] prev_velocity 上次速度(误差导数)
            float(self.error_second_derivative),  # [9] prev_accel 上次加速度(误差二阶导数)
            float(self.delta),  # [10] delta SPPVT参数
            float(self.eta),  # [11] eta SPPVT参数
            float(validated_input.get('control_mode_flag', 1))  # [12] mode_flag 控制模式
        ]
        
        return simulink_inputs
    
    def _update_stage_management(self, error_value: float, should_upgrade: bool, 
                               sppvt_rho: float) -> Tuple[float, float, float, float]:
        """
        阶段管理逻辑 (基于stage_manager.m)
        
        Returns:
            Tuple: (new_stage_offset, new_stage, new_error_sign, new_upgrade_count)
        """
        # 计算当前误差符号
        if abs(error_value) < 1e-6:
            current_sign = 0.0  # 接近零
        elif error_value > 0:
            current_sign = 1.0  # 正误差
        else:
            current_sign = -1.0  # 负误差
        
        # 初始化输出
        new_stage = self.current_stage
        new_stage_offset = self.stage_offset
        new_error_sign = self.error_sign
        new_upgrade_count = self.upgrade_count
        sign_changed = False
        
        # 误差符号变化检测
        if (self.error_sign != 0) and (current_sign != 0) and (self.error_sign != current_sign):
            # 符号变化：重置到初始阶段
            sign_changed = True
            new_stage = 1.0
            new_stage_offset = 0.0
            new_upgrade_count = 0.0
            
        elif should_upgrade and not sign_changed:
            # 无符号变化且满足升级条件
            new_stage = new_stage + 1.0
            new_upgrade_count = new_upgrade_count + 1.0
            
            # 基于误差符号计算新的级差
            if error_value > 0:
                # 正误差：增加正级差
                new_stage_offset = self.stage_offset + sppvt_rho * abs(error_value)
            else:
                # 负误差：增加负级差
                new_stage_offset = self.stage_offset - sppvt_rho * abs(error_value)
            
            # 限制级差范围
            new_stage_offset = max(-100.0, min(100.0, new_stage_offset))
        
        # 更新误差符号历史（只对非零误差）
        if current_sign != 0:
            new_error_sign = current_sign
        
        return new_stage_offset, new_stage, new_error_sign, new_upgrade_count
    
    def process_sppvt_control(self, decision_output: Dict, validated_input: Dict) -> Dict:
        """
        处理SPPVT控制的主要接口

        Args:
            decision_output: ACC决策输出
            validated_input: 验证后的输入数据

        Returns:
            Dict: SPPVT控制输出
        """
        if not self.model_loaded:
            if not self.initialize_matlab_engine():
                raise RuntimeError("MATLAB引擎或模型初始化失败")

        # 1. 准备Simulink输入 (12个输入端口)
        simulink_inputs = self._prepare_simulink_inputs(decision_output, validated_input)

        # 2. 调用Simulink模型
        start_time = time.time()

        # 准备外部输入数据（数值数组格式）
        # 对于多个Inport块，使用ExternalInput参数
        # 格式：第一列是时间，后续12列是12个输入值
        # 由于仿真时长是dt，提供两个时间点：[0, dt]

        # 构造输入数组 (2行13列)：时间列 + 12个输入列
        # 注意：两个时间点使用相同的输入值（保持常量）
        ext_input_data = [
            [0.0] + simulink_inputs,      # t=0时刻的输入
            [self.dt] + simulink_inputs   # t=dt时刻的输入
        ]

        # 转换为MATLAB double数组
        ext_input = matlab.double(ext_input_data)

        # 将外部输入写入workspace
        self.matlab_engine.workspace['ext_input'] = ext_input

        # 使用SimulationInput对象配置仿真
        sim_in = self.matlab_engine.eval(f"Simulink.SimulationInput('{self.model_name}')", nargout=1)
        sim_in = self.matlab_engine.setExternalInput(sim_in, 'ext_input', nargout=1)

        # 运行仿真
        sim_out = self.matlab_engine.sim(sim_in, nargout=1)

        simulation_time = (time.time() - start_time) * 1000  # 转换为毫秒

        # 3. 从sim输出对象中提取信号 (8个输出端口)
        # 输出端口: control_output, velocity, acceleration, jerk, should_upgrade, com1, com2, com3

        # 保存sim_out到workspace
        self.matlab_engine.workspace['sim_out'] = sim_out

        # 直接提取输出，不使用try-except，让错误暴露出来
        # Structure格式：sim_out.yout.signals(i).values包含第i个输出的数据
        sppvt_control_output = float(self.matlab_engine.eval('sim_out.yout.signals(1).values(end)', nargout=1))
        sppvt_velocity_output = float(self.matlab_engine.eval('sim_out.yout.signals(2).values(end)', nargout=1))
        sppvt_acceleration_output = float(self.matlab_engine.eval('sim_out.yout.signals(3).values(end)', nargout=1))
        sppvt_jerk_output = float(self.matlab_engine.eval('sim_out.yout.signals(4).values(end)', nargout=1))
        should_upgrade_flag = float(self.matlab_engine.eval('sim_out.yout.signals(5).values(end)', nargout=1))
        upgrade_com1 = float(self.matlab_engine.eval('sim_out.yout.signals(6).values(end)', nargout=1))
        upgrade_com2 = float(self.matlab_engine.eval('sim_out.yout.signals(7).values(end)', nargout=1))
        upgrade_com3 = float(self.matlab_engine.eval('sim_out.yout.signals(8).values(end)', nargout=1))

        if self.debug:
            print(f"✅ 成功提取8个输出信号")
            print(f"   control_output={sppvt_control_output:.6f}")
            print(f"   velocity={sppvt_velocity_output:.6f}")
            print(f"   acceleration={sppvt_acceleration_output:.6f}")
            print(f"   jerk={sppvt_jerk_output:.6f}")
            print(f"   升级条件: should_upgrade={should_upgrade_flag}, com1={upgrade_com1}, com2={upgrade_com2}, com3={upgrade_com3}")

        # 4. 阶段管理更新
        error_value = simulink_inputs[0]  # 第一个输入是误差

        # 使用Simulink返回的升级标志
        should_upgrade = (should_upgrade_flag > 0.5)

        new_stage_offset, new_stage, new_error_sign, new_upgrade_count = \
            self._update_stage_management(error_value, should_upgrade, self.sppvt_rho)

        # 5. 更新内部状态
        self.stage_offset = new_stage_offset
        self.current_stage = new_stage
        self.error_sign = new_error_sign
        self.upgrade_count = new_upgrade_count
        self.control_error = error_value
        self.error_derivative = sppvt_velocity_output  # 更新误差导数
        self.error_second_derivative = sppvt_acceleration_output  # 更新误差二阶导数

        # 6. 构造输出
        output = {
            # SPPVT控制输出
            'sppvt_control_output': sppvt_control_output,
            'sppvt_velocity_output': sppvt_velocity_output,
            'sppvt_acceleration_output': sppvt_acceleration_output,
            'sppvt_jerk_output': sppvt_jerk_output,  # 加加速度输出
            'sppvt_stage_output': new_stage,
            'sppvt_status_output': 1.0 if abs(sppvt_control_output) > 0 else 0.0,  # 状态指示

            # 更新的SPPVT状态
            'new_stage_offset': new_stage_offset,
            'new_stage': new_stage,
            'new_error_sign': new_error_sign,
            'new_upgrade_count': new_upgrade_count,
            'new_control_error': error_value,
            'new_error_derivative': sppvt_velocity_output,
            'new_error_second_derivative': sppvt_acceleration_output,

            # 性能信息
            'simulation_time_ms': simulation_time,

            # 目标加速度 (主要控制输出)
            'target_accel': sppvt_control_output
        }

        # 调试输出
        if self.debug:
            print(f"SPPVT Manager: Error={error_value:.3f}, Stage={new_stage:.0f}, "
                  f"Offset={new_stage_offset:.3f}, Control={sppvt_control_output:.3f}, "
                  f"SimTime={simulation_time:.1f}ms")

        return output
    
    def reset_sppvt_state(self):
        """重置SPPVT状态"""
        self.stage_offset = 0.0
        self.current_stage = 1.0
        self.error_sign = 0.0
        self.upgrade_count = 0.0
        self.control_error = 0.0
        self.error_derivative = 0.0
        self.error_second_derivative = 0.0
    
    def get_sppvt_state(self) -> Dict:
        """获取当前SPPVT状态"""
        return {
            'stage_offset': self.stage_offset,
            'current_stage': self.current_stage,
            'error_sign': self.error_sign,
            'upgrade_count': self.upgrade_count,
            'control_error': self.control_error,
            'error_derivative': self.error_derivative,
            'error_second_derivative': self.error_second_derivative
        }
    
    def cleanup(self):
        """清理资源"""
        if self.matlab_engine and self.model_loaded:
            try:
                # 关闭模型，不保存更改（第二个参数为0）
                self.matlab_engine.close_system(self.model_name, 0, nargout=0)
                print(f"✅ 模型 {self.model_name} 已关闭")
            except:
                pass

        if self.matlab_engine:
            try:
                self.matlab_engine.quit()
                print("✅ MATLAB引擎已关闭")
            except:
                pass