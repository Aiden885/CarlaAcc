#!/usr/bin/env python3
"""
MATLAB决策模块接口 - 与Python版本acc_decision.py具有相同的接口
提供MATLAB Simulink和Python之间的决策逻辑互换能力
"""

try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print("⚠️ MATLAB Engine未安装，将使用Python版本的决策逻辑")

import numpy as np
from enum import Enum
from acc_decision import ACCCommand, ACCState, ACCControlMode


class MATLABDecisionModule:
    """
    MATLAB版本的ACC决策模块
    与Python版本的ACCDecisionModule具有相同的接口
    """
    
    def __init__(self, initial_V3_kmh=50.0, initial_G1_m=15.0, initial_time_gap=2.0):
        """
        初始化MATLAB ACC决策模块
        
        Args:
            initial_V3_kmh: 初始V3速度 (km/h)
            initial_G1_m: 初始G1距离 (m) 
            initial_time_gap: 初始时间间隔 (s)
        """
        self.initial_V3_kmh = initial_V3_kmh
        self.initial_G1_m = initial_G1_m
        self.initial_G2_s = initial_time_gap
        
        # 调试模式
        self.debug = False
        
        # MATLAB引擎
        self.matlab_engine = None
        self.model_name = 'acc_decision_md'
        
        # 当前状态跟踪（与Python版本同步）
        self.current_state = ACCState.ADAPTIVE_NO_HISTORY_STANDBY
        self.current_control_mode = None
        self.previous_state = None
        
        # 参数跟踪
        self.V3_kmh = initial_V3_kmh
        self.G1_m = initial_G1_m
        self.G2_s = initial_time_gap
        self.V1_kmh = 0  # 适速/低速界限
        
        # 历史数据
        self.has_history = False
        self.history_V3_kmh = None
        self.history_G1_m = None
        self.history_G2_s = None
        
        # 初始化MATLAB引擎
        self._init_matlab_engine()
        
        print("MATLAB ACC决策模块初始化完成")
        print(f"初始参数: V3={self.V3_kmh}km/h, G1={self.G1_m}m, G2={self.G2_s}s")
    
    def _init_matlab_engine(self):
        """初始化MATLAB引擎"""
        if not MATLAB_AVAILABLE:
            print("⚠️ MATLAB不可用，决策模块将无法工作")
            return
        
        try:
            print("启动MATLAB引擎...")
            self.matlab_engine = matlab.engine.start_matlab()
            
            # 检查模型是否存在
            if not self._check_model_exists():
                print("🔧 模型不存在，正在创建...")
                self._create_model()
            
            # 加载模型
            self.matlab_engine.load_system(self.model_name, nargout=0)
            print(f"✅ MATLAB模型 {self.model_name} 加载成功")
            
        except Exception as e:
            print(f"❌ MATLAB引擎初始化失败: {e}")
            self.matlab_engine = None
    
    def _check_model_exists(self):
        """检查模型文件是否存在"""
        try:
            model_files = self.matlab_engine.dir(f'{self.model_name}.slx')
            return len(model_files) > 0
        except:
            return False
    
    def _create_model(self):
        """创建MATLAB模型"""
        try:
            # 运行模型创建脚本
            self.matlab_engine.create_decision_md_simulink(nargout=0)
            self.matlab_engine.configure_decision_md_stateflow(nargout=0)
            print("✅ MATLAB模型创建完成")
        except Exception as e:
            print(f"❌ 创建MATLAB模型失败: {e}")
            raise
    
    def process_command(self, command, ego_speed_kmh, has_target=False, current_distance=None):
        """
        处理ACC指令并执行状态转移
        
        Args:
            command: ACC指令
            ego_speed_kmh: 当前车速 (km/h)
            has_target: 是否存在前车目标
            current_distance: 当前与前车距离 (m, 可选)
        
        Returns:
            tuple: (新状态, 控制模式, 执行结果消息)
        """
        if self.matlab_engine is None:
            return self.current_state, None, "MATLAB引擎不可用"
        
        if not isinstance(command, ACCCommand):
            return self.current_state, None, "无效指令"
        
        if self.debug:
            print(f"MATLAB决策: 处理指令 {command.value}, 速度 {ego_speed_kmh:.1f}km/h, 有前车: {has_target}")
        
        try:
            # 将Python枚举转换为MATLAB输入
            command_input = self._command_to_matlab_input(command)
            distance_input = current_distance if (has_target and current_distance is not None) else -1.0
            
            # 调用MATLAB模型进行决策
            result = self.matlab_engine.sim(self.model_name, 
                                          'StartTime', '0',
                                          'StopTime', '0.05',
                                          'SaveOutput', 'on',
                                          'ReturnWorkspaceOutputs', 'on',
                                          'LoadExternalInput', 'on',
                                          'ExternalInput', self._prepare_matlab_inputs(
                                              command_input, ego_speed_kmh, has_target, distance_input))
            
            # 解析MATLAB输出
            new_state, control_mode, message = self._parse_matlab_output(result)
            
            # 更新内部状态
            self.previous_state = self.current_state
            self.current_state = new_state
            self.current_control_mode = control_mode
            
            if self.debug:
                print(f"MATLAB决策结果: {self.current_state.value} -> {control_mode.value if control_mode else None}")
            
            return new_state, control_mode, message
            
        except Exception as e:
            error_msg = f"MATLAB决策执行失败: {e}"
            print(f"❌ {error_msg}")
            return self.current_state, None, error_msg
    
    def _command_to_matlab_input(self, command):
        """将Python ACC指令转换为MATLAB输入编号"""
        command_mapping = {
            ACCCommand.DECREASE_SPEED: 0,    # I0
            ACCCommand.INCREASE_SPEED: 1,    # I1  
            ACCCommand.DECREASE_DISTANCE: 2, # I2
            ACCCommand.INCREASE_DISTANCE: 3, # I3
            ACCCommand.THROTTLE: 4,          # I4
            ACCCommand.BRAKE: 5,             # I5
            ACCCommand.CANCEL: 6,            # I6
        }
        return command_mapping.get(command, -1)
    
    def _prepare_matlab_inputs(self, command_input, ego_speed_kmh, has_target, distance_input):
        """准备MATLAB模型输入"""
        # 创建输入结构体，对应Simulink模型的输入端口
        inputs = {
            'time': [0],
            'signals': {
                'values': [
                    [command_input],        # command_input
                    [ego_speed_kmh],        # ego_speed_kmh
                    [has_target],           # has_target
                    [distance_input],       # current_distance
                    [False]                 # reset_signal
                ],
                'dimensions': [1, 1, 1, 1, 1]
            }
        }
        return matlab.double(inputs)
    
    def _parse_matlab_output(self, sim_result):
        """解析MATLAB仿真输出"""
        try:
            # 从仿真结果中提取输出
            outputs = sim_result['yout']  # 假设输出被保存在yout中
            
            # 解析状态和控制模式
            current_state_num = int(outputs[0][-1])  # 最后一个时间步的状态
            control_mode_num = int(outputs[1][-1])   # 最后一个时间步的控制模式
            message_code = int(outputs[8][-1])       # 消息代码
            
            # 更新参数
            self.V3_kmh = float(outputs[4][-1])
            self.G1_m = float(outputs[5][-1])
            self.G2_s = float(outputs[6][-1])
            self.has_history = bool(outputs[7][-1])
            
            # 转换为Python枚举
            new_state = self._matlab_state_to_python(current_state_num)
            control_mode = self._matlab_control_mode_to_python(control_mode_num)
            
            message = f"MATLAB决策: 状态{current_state_num} 模式{control_mode_num} 消息{message_code}"
            
            return new_state, control_mode, message
            
        except Exception as e:
            print(f"❌ 解析MATLAB输出失败: {e}")
            return self.current_state, None, f"输出解析错误: {e}"
    
    def _matlab_state_to_python(self, state_num):
        """将MATLAB状态编号转换为Python枚举"""
        state_mapping = {
            0: ACCState.IN_CONTROL,
            1: ACCState.ADAPTIVE_HISTORY_STANDBY,
            2: ACCState.ADAPTIVE_NO_HISTORY_STANDBY,
            3: ACCState.LOW_SPEED
        }
        return state_mapping.get(state_num, ACCState.ADAPTIVE_NO_HISTORY_STANDBY)
    
    def _matlab_control_mode_to_python(self, mode_num):
        """将MATLAB控制模式编号转换为Python枚举"""
        mode_mapping = {
            0: None,
            1: ACCControlMode.SPEED_DECREASE,    # R1
            2: ACCControlMode.SPEED_INCREASE,    # R2
            3: ACCControlMode.DISTANCE_DECREASE, # R3
            4: ACCControlMode.DISTANCE_INCREASE, # R4
            5: ACCControlMode.NO_CONTINUE_CONTROL, # R5
            6: ACCControlMode.CONTINUE_CONTROL,  # R6
            7: ACCControlMode.TORQUE_ARBITRATION, # R7
            8: ACCControlMode.SYSTEM_STANDBY,    # R8
        }
        return mode_mapping.get(mode_num, None)
    
    def get_current_parameters(self):
        """获取当前ACC参数（与Python版本接口兼容）"""
        return {
            'V3_kmh': self.V3_kmh,
            'G1_m': self.G1_m,
            'G2_s': self.G2_s,
            'state': self.current_state.value,
            'has_history': self.has_history,
            'is_active': self.current_state in [ACCState.IN_CONTROL],
            'pending_distance_adjustment': 0.0,  # MATLAB版本中简化
            'cruise_mode_active': False,  # decision.md中不包含巡航模式
            'force_cruise_mode': False,
            'current_control_mode': self.current_control_mode.value if self.current_control_mode else None
        }
    
    def get_decision_output(self, ego_speed_kmh, current_distance=None):
        """获取决策输出，供控制模块使用（与Python版本接口兼容）"""
        has_target = current_distance is not None and current_distance < 100.0
        
        return {
            'acc_active': self.current_state in [ACCState.IN_CONTROL],
            'V3_kmh': self.V3_kmh,
            'V3_ms': self.V3_kmh / 3.6,
            'G1_m': self.G1_m,
            'G2_s': self.G2_s,
            'state': self.current_state.value,
            'state_description': self._get_state_description(),
            'control_enabled': self.current_state in [ACCState.IN_CONTROL],
            'has_target': has_target,
            'effective_has_target': has_target,
            'cruise_mode_active': False,
            'force_cruise_mode': False,
            'pending_distance_adjustment': 0.0,
            'current_control_mode': self.current_control_mode.value if self.current_control_mode else None,
            'torque_arbitration_active': self.current_control_mode == ACCControlMode.TORQUE_ARBITRATION
        }
    
    def _get_state_description(self):
        """获取状态描述"""
        descriptions = {
            ACCState.IN_CONTROL: "S0-在控",
            ACCState.ADAPTIVE_HISTORY_STANDBY: "S1-适速有史待命",
            ACCState.ADAPTIVE_NO_HISTORY_STANDBY: "S2-适速无史待命",
            ACCState.LOW_SPEED: "S3-低速状态",
        }
        return descriptions.get(self.current_state, "未知状态")
    
    def set_debug(self, enable):
        """启用/禁用调试模式"""
        self.debug = enable
    
    def reset(self):
        """重置ACC决策模块"""
        if self.matlab_engine and self._check_model_exists():
            try:
                # 重置MATLAB模型状态
                self.matlab_engine.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
                self.matlab_engine.set_param(self.model_name, 'SimulationCommand', 'start', nargout=0)
            except Exception as e:
                print(f"⚠️ MATLAB模型重置失败: {e}")
        
        # 重置Python端状态
        self.current_state = ACCState.ADAPTIVE_NO_HISTORY_STANDBY
        self.current_control_mode = None
        self.previous_state = None
        self.has_history = False
        self.history_V3_kmh = None
        self.history_G1_m = None
        self.history_G2_s = None
        
        print("MATLAB ACC决策模块已重置")
    
    def __del__(self):
        """析构函数，清理MATLAB引擎"""
        if self.matlab_engine:
            try:
                self.matlab_engine.quit()
                print("🔌 MATLAB引擎已关闭")
            except:
                pass


def create_decision_module(use_matlab=True, **kwargs):
    """
    工厂函数：创建决策模块（MATLAB或Python版本）
    
    Args:
        use_matlab: 是否使用MATLAB版本
        **kwargs: 初始化参数
    
    Returns:
        决策模块实例
    """
    if use_matlab and MATLAB_AVAILABLE:
        print("🚀 创建MATLAB版本的ACC决策模块")
        return MATLABDecisionModule(**kwargs)
    else:
        print("🐍 创建Python版本的ACC决策模块")
        from acc_decision import ACCDecisionModule
        return ACCDecisionModule(**kwargs)


if __name__ == "__main__":
    # 测试MATLAB决策模块
    print("=== MATLAB决策模块测试 ===")
    
    if not MATLAB_AVAILABLE:
        print("❌ MATLAB不可用，无法测试")
        exit(1)
    
    # 创建MATLAB决策模块
    matlab_decision = MATLABDecisionModule()
    matlab_decision.set_debug(True)
    
    # 测试基本决策
    test_cases = [
        ("激活系统", ACCCommand.THROTTLE, 35.0, False),
        ("降速指令", ACCCommand.DECREASE_SPEED, 35.0, False),
        ("增速指令", ACCCommand.INCREASE_SPEED, 35.0, False),
        ("刹车指令", ACCCommand.BRAKE, 35.0, True),
    ]
    
    for desc, command, speed, has_target in test_cases:
        print(f"\n测试: {desc}")
        state, mode, msg = matlab_decision.process_command(command, speed, has_target)
        print(f"结果: {state.value} | {mode.value if mode else None} | {msg}")
    
    print("\n=== MATLAB决策模块测试完成 ===")