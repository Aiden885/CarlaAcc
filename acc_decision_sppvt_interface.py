#!/usr/bin/env python3
"""
ACC决策+SPPVT一体化Simulink接口模块
将ACC状态机决策逻辑与SPPVT控制算法集成到统一的Simulink计算核心

"""

import time
import threading
import numpy as np

# 导入MATLAB引擎
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError as e:
    matlab = None
    MATLAB_AVAILABLE = False
    print(f"警告: MATLAB引擎不可用: {e}")

# 导入现有模块作为备用方案
try:
    from acc_decision import ACCDecisionModule, ACCCommand
    DECISION_BACKUP_AVAILABLE = True
except ImportError:
    print("警告: acc_decision模块不可用，无备用方案")
    DECISION_BACKUP_AVAILABLE = False

# 导入新的实时SPPVT状态管理器
try:
    from realtime_sppvt_state_manager import RealtimeSPPVTStateManager
    REALTIME_SPPVT_AVAILABLE = True
except ImportError as e:
    print(f"警告: 实时SPPVT状态管理器不可用: {e}")
    REALTIME_SPPVT_AVAILABLE = False


class ACCDecisionSPPVTInterface:
    """
    ACC决策+SPPVT一体化Simulink接口
    
    功能:
    - 接收Python预处理后的标准化输入
    - 调用Simulink执行决策状态机 + SPPVT控制计算
    - 返回标准化的控制输出和状态信息
    - 提供完整的调试和错误处理机制
    """
    
    def __init__(self, debug=True, use_realtime_sppvt=True):
        self.debug = debug
        self.use_realtime_sppvt = use_realtime_sppvt
        self.matlab_engine = None
        self.model_name = 'ACC_Decision_SPPVT_Integrated'
        self.engine_lock = threading.Lock()

        # 实时SPPVT状态管理器（新方案）
        self.realtime_sppvt_manager = None

        # 备用决策模块（如果Simulink不可用）
        self.backup_decision = None

        # 性能监控
        self.call_count = 0
        self.total_compute_time = 0.0
        self.last_call_time = 0.0

        # 错误计数
        self.error_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3

        # 根据配置选择初始化方案
        if self.use_realtime_sppvt and REALTIME_SPPVT_AVAILABLE:
            self._init_realtime_sppvt()
        else:
            # 旧方案：传统Simulink接口
            self._init_matlab_simulink()

        # 初始化备用决策模块
        self._init_backup_decision()

        if self.debug:
            print("✅ ACC决策+SPPVT一体化接口初始化完成")
            print(f"   实时SPPVT管理器: {'可用' if self.realtime_sppvt_manager else '不可用'}")
            print(f"   传统MATLAB引擎: {'可用' if self.matlab_engine else '不可用'}")
            print(f"   备用决策状态: {'可用' if self.backup_decision else '不可用'}")
    
    def _init_matlab_simulink(self):
        """初始化MATLAB引擎和Simulink模型"""
        if not MATLAB_AVAILABLE:
            if self.debug:
                print("MATLAB引擎不可用，将使用备用方案")
            return
        
        try:
            if self.debug:
                print("正在启动MATLAB引擎...")
            
            # 启动MATLAB引擎
            self.matlab_engine = matlab.engine.start_matlab()
            
            # 添加当前路径
            self.matlab_engine.addpath(self.matlab_engine.pwd(), nargout=0)
            
            # 创建总线结构
            try:
                self.matlab_engine.eval("create_decision_sppvt_bus()", nargout=0)
                if self.debug:
                    print("✅ ACC决策+SPPVT总线结构创建完成")
            except:
                if self.debug:
                    print("⚠️ 总线结构创建跳过（可能已存在）")
            
            # 检查模型文件是否存在
            model_exists = self.matlab_engine.exist(f'{self.model_name}.slx', 'file')
            if model_exists == 0:
                if self.debug:
                    print(f"⚠️ Simulink模型 {self.model_name}.slx 不存在，将创建基础版本")
                # 这里可以创建基础模型或使用备用方案
                raise FileNotFoundError(f"Simulink模型文件不存在: {self.model_name}.slx")
            
            # 加载模型
            self.matlab_engine.load_system(self.model_name, nargout=0)
            
            # 配置模型参数
            self.matlab_engine.set_param(self.model_name, 'StopTime', '0.05', nargout=0)
            self.matlab_engine.set_param(self.model_name, 'FixedStep', '0.001', nargout=0)
            
            if self.debug:
                print(f"✅ Simulink模型 {self.model_name} 加载完成")
        
        except Exception as e:
            print(f"❌ MATLAB引擎或Simulink模型初始化失败: {e}")
            self.matlab_engine = None
    
    def _init_realtime_sppvt(self):
        """初始化实时SPPVT状态管理器"""
        try:
            if self.debug:
                print("正在初始化实时SPPVT状态管理器...")

            self.realtime_sppvt_manager = RealtimeSPPVTStateManager(
                model_name=self.model_name
            )

            if self.debug:
                print("✅ 实时SPPVT状态管理器初始化完成")

        except Exception as e:
            print(f"❌ 实时SPPVT状态管理器初始化失败: {e}")
            self.realtime_sppvt_manager = None
            # 回退到传统方案
            if self.debug:
                print("回退到传统Simulink接口...")
            self._init_matlab_simulink()

    def _init_backup_decision(self):
        """初始化备用决策模块"""
        if DECISION_BACKUP_AVAILABLE:
            try:
                self.backup_decision = ACCDecisionModule(
                    initial_min_speed_kmh=30.0,
                    initial_target_speed_kmh=50.0,
                    initial_time_gap=2.0
                )
                if self.debug:
                    print("✅ 备用决策模块初始化完成")
            except Exception as e:
                print(f"❌ 备用决策模块初始化失败: {e}")
                self.backup_decision = None
    
    def process_decision_and_control(self, input_data):
        """
        一体化处理：决策判断 + SPPVT控制计算

        Args:
            input_data (dict): 标准化输入数据，包含以下字段：
                'ego_speed_kmh': float,      # 自车速度 km/h
                'ego_speed_ms': float,       # 自车速度 m/s
                'command_type': int,         # 0=NONE, 1=I0, 2=I1, ..., 7=I6
                'command_active': bool,      # 指令是否激活
                'manual_throttle_active': bool,  # 手动油门是否激活
                'control_error': float,      # Two Mode计算的控制误差
                'control_mode_flag': int,    # 1=distance, 2=speed
                'V_target_kmh': float,       # 目标速度
                'V_min_kmh': float,          # 最小速度
                'G2_s': float,               # 时距参数
                'timestamp': float           # 时间戳

        Returns:
            dict: 标准化输出数据，包含以下字段：
                'target_accel': float,              # SPPVT目标加速度 m/s²
                'control_enabled': bool,            # 控制使能状态
                'current_state': int,               # 0=S0, 1=S1, 2=S2, 3=S3
                'current_decision': int,            # 0=NONE, 1=R1, ..., 8=R8
                'torque_arbitration_active': bool,  # 扭矩仲裁激活
                'updated_V_target_kmh': float,      # 更新的目标速度
                'updated_G2_s': float,              # 更新的时距参数
                'sppvt_stage': int,                 # SPPVT阶段
                'sppvt_upgrade_count': int,         # SPPVT升级次数
                'debug_message': str                # 调试信息
        """
        start_time = time.time()
        self.call_count += 1

        # 验证输入数据
        if not self._validate_input(input_data):
            return self._get_error_output("输入数据验证失败")

        # 优先使用实时SPPVT状态管理器（新方案）
        if self.realtime_sppvt_manager is not None and self.consecutive_errors < self.max_consecutive_errors:
            result = self._call_realtime_sppvt(input_data)
            if result is not None:
                self.consecutive_errors = 0  # 重置错误计数
                compute_time = time.time() - start_time
                self.total_compute_time += compute_time
                self.last_call_time = compute_time

                if self.debug:
                    self._print_realtime_sppvt_debug(input_data, result, compute_time)

                return result

        # 实时SPPVT失败，尝试传统Simulink方案
        elif self.matlab_engine is not None and self.consecutive_errors < self.max_consecutive_errors:
            result = self._call_simulink(input_data)
            if result is not None:
                self.consecutive_errors = 0  # 重置错误计数
                compute_time = time.time() - start_time
                self.total_compute_time += compute_time
                self.last_call_time = compute_time

                if self.debug:
                    self._print_simulink_debug(input_data, result, compute_time)

                return result

        # 所有Simulink方案失败，使用备用方案
        self.consecutive_errors += 1
        self.error_count += 1

        if self.debug:
            print(f"⚠️ 所有Simulink方案失败，使用备用方案 (连续错误: {self.consecutive_errors})")

        return self._call_backup(input_data)
    
    def _validate_input(self, input_data):
        """验证输入数据格式和范围"""
        required_fields = [
            'ego_speed_kmh', 'ego_speed_ms', 'command_type', 'command_active',
            'manual_throttle_active', 'control_error', 'control_mode_flag',
            'V_target_kmh', 'V_min_kmh', 'G2_s', 'timestamp'
        ]
        
        for field in required_fields:
            if field not in input_data:
                if self.debug:
                    print(f"❌ 缺少必需字段: {field}")
                return False
        
        # 范围检查
        if not (0 <= input_data['command_type'] <= 7):
            if self.debug:
                print(f"❌ command_type超出范围: {input_data['command_type']}")
            return False
        
        if not (1 <= input_data['control_mode_flag'] <= 2):
            if self.debug:
                print(f"❌ control_mode_flag超出范围: {input_data['control_mode_flag']}")
            return False
        
        return True

    def _call_realtime_sppvt(self, input_data):
        """使用实时SPPVT状态管理器进行计算"""
        try:
            # 使用实时SPPVT状态管理器执行单步仿真
            control_enabled = self._determine_control_enabled(input_data)
            control_mode_flag = input_data['control_mode_flag'] == 1  # 1=distance, 2=speed

            # 调用实时SPPVT管理器
            sppvt_output = self.realtime_sppvt_manager.run_single_step_simulation(
                control_error=input_data['control_error'],
                ego_speed_ms=input_data['ego_speed_ms'],
                control_mode_flag=control_mode_flag,
                control_enabled=control_enabled
            )

            if sppvt_output is None:
                return None

            # 执行决策逻辑（简化版本，在实际应用中应该也是从Simulink获取）
            decision_result = self._execute_decision_logic(input_data)

            # 获取SPPVT性能统计
            sppvt_stats = self.realtime_sppvt_manager.get_performance_stats()

            # 组合返回结果
            return {
                'target_accel': sppvt_output,
                'control_enabled': control_enabled,
                'current_state': decision_result.get('state', 2),
                'current_decision': decision_result.get('decision', 0),
                'torque_arbitration_active': decision_result.get('torque_arbitration', False),
                'updated_V_target_kmh': input_data['V_target_kmh'],  # 简化处理
                'updated_G2_s': input_data['G2_s'],  # 简化处理
                'sppvt_stage': 1,  # 从状态管理器获取
                'sppvt_upgrade_count': 0,  # 从状态管理器获取
                'debug_message': f"实时SPPVT计算成功,当前级差:{sppvt_stats.get('current_stage_offset', 0):.3f}"
            }

        except Exception as e:
            if self.debug:
                print(f"❌ 实时SPPVT调用失败: {e}")
                import traceback
                traceback.print_exc()
            return None

    def _determine_control_enabled(self, input_data):
        """确定控制是否使能（简化决策逻辑）"""
        # 这是一个简化的实现，实际中应该从完整的决策状态机获取
        if input_data['manual_throttle_active']:
            return False

        if not input_data['command_active']:
            return False

        # 基于command_type的简单使能逻辑
        enabling_commands = [1, 2, 3, 4]  # I0, I1, I2, I3
        return input_data['command_type'] in enabling_commands

    def _execute_decision_logic(self, input_data):
        """执行决策逻辑（简化版本）"""
        # 这是一个简化的实现，实际中应该调用完整的Simulink决策模块
        if input_data['manual_throttle_active']:
            return {'state': 0, 'decision': 0, 'torque_arbitration': False}

        if not input_data['command_active']:
            return {'state': 2, 'decision': 0, 'torque_arbitration': False}

        # 基于速度的简单状态判断
        speed = input_data['ego_speed_kmh']
        if speed < 20:
            state = 0  # S0
        elif speed < 40:
            state = 1  # S1
        else:
            state = 2  # S2

        # 简单的决策映射
        command_decision_map = {
            1: 1,  # I0 -> R1
            2: 2,  # I1 -> R2
            3: 3,  # I2 -> R3
            4: 4,  # I3 -> R4
            5: 5,  # I4 -> R5
            6: 6,  # I5 -> R6
            7: 7,  # I6 -> R7
        }

        decision = command_decision_map.get(input_data['command_type'], 0)
        torque_arbitration = decision in [1, 2, 5, 6]  # 需要扭矩仲裁的决策

        return {
            'state': state,
            'decision': decision,
            'torque_arbitration': torque_arbitration
        }

    def _call_simulink(self, input_data):
        """调用Simulink进行计算"""
        with self.engine_lock:
            try:
                # 准备Simulink输入数据
                simulink_input = self._prepare_simulink_input(input_data)
                
                # 将输入数据传入MATLAB工作空间
                self.matlab_engine.workspace['decision_sppvt_input'] = simulink_input
                
                # 运行仿真
                self.matlab_engine.eval(f"simOut = sim('{self.model_name}');", nargout=0)
                
                # 获取输出结果
                output_result = self._extract_simulink_output()
                
                return output_result
                
            except Exception as e:
                if self.debug:
                    print(f"❌ Simulink调用失败: {e}")
                    import traceback
                    traceback.print_exc()
                return None
    
    def _prepare_simulink_input(self, input_data):
        """准备Simulink输入数据格式"""
        # 转换为MATLAB兼容的数据格式
        matlab_input = {}
        
        # 数值类型转换
        matlab_input['ego_speed_kmh'] = float(input_data['ego_speed_kmh'])
        matlab_input['ego_speed_ms'] = float(input_data['ego_speed_ms'])
        matlab_input['command_type'] = int(input_data['command_type'])
        matlab_input['command_active'] = bool(input_data['command_active'])
        matlab_input['manual_throttle_active'] = bool(input_data['manual_throttle_active'])
        matlab_input['control_error'] = float(input_data['control_error'])
        matlab_input['control_mode_flag'] = int(input_data['control_mode_flag'])
        matlab_input['V_target_kmh'] = float(input_data['V_target_kmh'])
        matlab_input['V_min_kmh'] = float(input_data['V_min_kmh'])
        matlab_input['G2_s'] = float(input_data['G2_s'])
        matlab_input['timestamp'] = float(input_data['timestamp'])
        
        return matlab_input
    
    def _extract_simulink_output(self):
        """从Simulink输出中提取结果"""
        try:
            # 从工作空间获取输出（假设模型会将结果写入工作空间）
            self.matlab_engine.eval("output_data = simOut.yout;", nargout=0)
            
            # 提取各个输出信号（这里需要根据实际Simulink模型调整）
            target_accel = float(self.matlab_engine.eval("output_data.signals(1).values(end)"))
            control_enabled = bool(self.matlab_engine.eval("output_data.signals(2).values(end)"))
            current_state = int(self.matlab_engine.eval("output_data.signals(3).values(end)"))
            current_decision = int(self.matlab_engine.eval("output_data.signals(4).values(end)"))
            torque_arbitration_active = bool(self.matlab_engine.eval("output_data.signals(5).values(end)"))
            updated_V_target_kmh = float(self.matlab_engine.eval("output_data.signals(6).values(end)"))
            updated_G2_s = float(self.matlab_engine.eval("output_data.signals(7).values(end)"))
            sppvt_stage = int(self.matlab_engine.eval("output_data.signals(8).values(end)"))
            sppvt_upgrade_count = int(self.matlab_engine.eval("output_data.signals(9).values(end)"))
            debug_message = int(self.matlab_engine.eval("output_data.signals(10).values(end)"))
            
            return {
                'target_accel': target_accel,
                'control_enabled': control_enabled,
                'current_state': current_state,
                'current_decision': current_decision,
                'torque_arbitration_active': torque_arbitration_active,
                'updated_V_target_kmh': updated_V_target_kmh,
                'updated_G2_s': updated_G2_s,
                'sppvt_stage': sppvt_stage,
                'sppvt_upgrade_count': sppvt_upgrade_count,
                'debug_message': f"Simulink计算成功,调试码:{debug_message}"
            }
            
        except Exception as e:
            if self.debug:
                print(f"❌ Simulink输出提取失败: {e}")
            return None
    
    def _call_backup(self, input_data):
        """使用备用Python方案"""
        if self.backup_decision is None:
            return self._get_error_output("无可用的计算方案")
        
        try:
            # 使用备用决策模块处理指令
            if input_data['command_active']:
                command_enum = self._convert_command_type(input_data['command_type'])
                state, decision, msg = self.backup_decision.process_command(
                    command_enum, input_data['ego_speed_kmh']
                )
            
            # 获取决策状态
            decision_output = self.backup_decision.get_decision_output(
                input_data['ego_speed_kmh'], 
                None,  # 距离信息在Two Mode中已处理
                input_data['manual_throttle_active']
            )
            
            # 简单的SPPVT计算（备用方案）
            if decision_output['control_enabled']:
                # 简化的比例控制
                target_accel = max(-3.0, min(2.0, 1.0 * input_data['control_error']))
            else:
                target_accel = 0.0
            
            return {
                'target_accel': target_accel,
                'control_enabled': decision_output['control_enabled'],
                'current_state': decision_output['state'],
                'current_decision': decision_output.get('current_decision', 0),
                'torque_arbitration_active': decision_output['torque_arbitration_active'],
                'updated_V_target_kmh': decision_output['V_target_kmh'],
                'updated_G2_s': decision_output['G2_s'],
                'sppvt_stage': 1,  # 备用方案固定值
                'sppvt_upgrade_count': 0,
                'debug_message': "Python备用方案计算"
            }
            
        except Exception as e:
            if self.debug:
                print(f"❌ 备用方案计算失败: {e}")
            return self._get_error_output(f"备用方案失败: {e}")
    
    def _convert_command_type(self, command_type):
        """转换命令类型为枚举"""
        command_map = {
            0: None,  # NONE
            1: ACCCommand.DECREASE_SPEED,  # I0
            2: ACCCommand.INCREASE_SPEED,  # I1
            3: ACCCommand.DECREASE_DISTANCE,  # I2
            4: ACCCommand.INCREASE_DISTANCE,  # I3
            5: ACCCommand.THROTTLE,  # I4
            6: ACCCommand.BRAKE,  # I5
            7: ACCCommand.CANCEL   # I6
        }
        return command_map.get(command_type, None)
    
    def _get_error_output(self, error_msg):
        """获取错误情况下的默认输出"""
        return {
            'target_accel': 0.0,
            'control_enabled': False,
            'current_state': 2,  # S2=适速无史待命
            'current_decision': 0,  # NONE
            'torque_arbitration_active': False,
            'updated_V_target_kmh': 50.0,
            'updated_G2_s': 2.0,
            'sppvt_stage': 1,
            'sppvt_upgrade_count': 0,
            'debug_message': f"错误: {error_msg}"
        }
    
    def _print_realtime_sppvt_debug(self, input_data, result, compute_time):
        """打印实时SPPVT调试信息"""
        print(f"🚀 实时SPPVT决策+控制调试 (耗时: {compute_time*1000:.1f}ms)")
        print(f"   📤 输入: 指令I{input_data['command_type']}, 速度{input_data['ego_speed_kmh']:.1f}km/h, 误差{input_data['control_error']:.3f}")
        print(f"   📥 输出: 状态S{result['current_state']}→R{result['current_decision']}, 加速度{result['target_accel']:.3f}m/s²")
        print(f"   🎯 SPPVT: {result['debug_message']}")
        print(f"   ⚖️ 扭矩仲裁: {'激活' if result['torque_arbitration_active'] else '关闭'}")
        print(f"   📊 性能: 总调用{self.call_count}次, 平均耗时{self.total_compute_time/self.call_count*1000:.1f}ms")

        # 显示SPPVT管理器的详细性能统计
        if self.realtime_sppvt_manager:
            sppvt_stats = self.realtime_sppvt_manager.get_performance_stats()
            if sppvt_stats != {"status": "no_data"}:
                print(f"   🔧 SPPVT详情: 级差={sppvt_stats.get('current_stage_offset', 0):.3f}, "
                      f"SPPVT耗时={sppvt_stats.get('avg_execution_time_ms', 0):.1f}ms")

    def _print_simulink_debug(self, input_data, result, compute_time):
        """打印Simulink调试信息"""
        print(f"🔧 传统Simulink决策+SPPVT调试 (耗时: {compute_time*1000:.1f}ms)")
        print(f"   📤 输入: 指令I{input_data['command_type']}, 速度{input_data['ego_speed_kmh']:.1f}km/h, 误差{input_data['control_error']:.3f}")
        print(f"   📥 输出: 状态S{result['current_state']}→R{result['current_decision']}, 加速度{result['target_accel']:.3f}m/s²")
        print(f"   🎯 SPPVT: 阶段{result['sppvt_stage']}, 升级{result['sppvt_upgrade_count']}次")
        print(f"   ⚖️ 扭矩仲裁: {'激活' if result['torque_arbitration_active'] else '关闭'}")
        print(f"   📊 性能: 总调用{self.call_count}次, 平均耗时{self.total_compute_time/self.call_count*1000:.1f}ms")
    
    def get_statistics(self):
        """获取接口统计信息"""
        return {
            'call_count': self.call_count,
            'total_compute_time': self.total_compute_time,
            'average_compute_time': self.total_compute_time / max(1, self.call_count),
            'error_count': self.error_count,
            'consecutive_errors': self.consecutive_errors,
            'matlab_engine_status': 'active' if self.matlab_engine else 'inactive',
            'backup_decision_status': 'active' if self.backup_decision else 'inactive'
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.call_count = 0
        self.total_compute_time = 0.0
        self.error_count = 0
        self.consecutive_errors = 0
    
    def __del__(self):
        """析构函数，清理资源"""
        if self.matlab_engine is not None:
            try:
                self.matlab_engine.close_system(self.model_name, 0, nargout=0)
                self.matlab_engine.quit()
            except:
                pass


# 工厂函数，便于外部调用
def create_decision_sppvt_interface(debug=True):
    """
    创建ACC决策+SPPVT一体化接口实例
    
    Args:
        debug (bool): 是否启用调试输出
    
    Returns:
        ACCDecisionSPPVTInterface: 接口实例
    """
    return ACCDecisionSPPVTInterface(debug=debug)


# 使用示例
if __name__ == "__main__":
    # 创建接口
    interface = create_decision_sppvt_interface(debug=True)
    
    # 测试数据
    test_input = {
        'ego_speed_kmh': 45.0,
        'ego_speed_ms': 12.5,
        'command_type': 1,  # I0=降速
        'command_active': True,
        'manual_throttle_active': False,
        'control_error': -2.5,  # 距离误差
        'control_mode_flag': 1,  # distance mode
        'V_target_kmh': 50.0,
        'V_min_kmh': 30.0,
        'G2_s': 2.0,
        'timestamp': time.time()
    }
    
    # 测试调用
    print("\n=== ACC决策+SPPVT一体化接口测试 ===")
    result = interface.process_decision_and_control(test_input)
    
    print(f"\n✅ 测试结果:")
    for key, value in result.items():
        print(f"   {key}: {value}")
    
    # 显示统计信息
    stats = interface.get_statistics()
    print(f"\n📊 接口统计:")
    for key, value in stats.items():
        print(f"   {key}: {value}")