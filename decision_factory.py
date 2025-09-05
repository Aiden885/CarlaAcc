#!/usr/bin/env python3
"""
决策模块工厂 - 提供MATLAB和Python决策模块的统一接口
支持在项目中无缝切换决策实现
"""

import os
import sys
from enum import Enum

# 尝试导入MATLAB接口
try:
    from matlab_decision_interface import MATLABDecisionModule, MATLAB_AVAILABLE
except ImportError:
    MATLAB_AVAILABLE = False
    MATLABDecisionModule = None

# 导入Python版本
from acc_decision import ACCDecisionModule as PythonDecisionModule


class DecisionBackend(Enum):
    """决策后端类型"""
    PYTHON = "python"
    MATLAB = "matlab" 
    AUTO = "auto"  # 自动选择


class DecisionModuleFactory:
    """决策模块工厂类"""
    
    @staticmethod
    def create(backend=DecisionBackend.AUTO, **kwargs):
        """
        创建决策模块
        
        Args:
            backend: 决策后端类型
            **kwargs: 决策模块初始化参数
            
        Returns:
            决策模块实例
        """
        if backend == DecisionBackend.AUTO:
            # 自动选择：优先使用MATLAB，如果不可用则使用Python
            backend = DecisionBackend.MATLAB if MATLAB_AVAILABLE else DecisionBackend.PYTHON
        
        if backend == DecisionBackend.MATLAB:
            if not MATLAB_AVAILABLE:
                print("⚠️ MATLAB不可用，切换到Python版本")
                backend = DecisionBackend.PYTHON
            else:
                print("🚀 使用MATLAB决策模块")
                return MATLABDecisionModule(**kwargs)
        
        if backend == DecisionBackend.PYTHON:
            print("🐍 使用Python决策模块")
            return PythonDecisionModule(**kwargs)
        
        raise ValueError(f"不支持的决策后端: {backend}")
    
    @staticmethod
    def get_available_backends():
        """获取可用的决策后端"""
        backends = [DecisionBackend.PYTHON]
        if MATLAB_AVAILABLE:
            backends.append(DecisionBackend.MATLAB)
        return backends
    
    @staticmethod
    def benchmark_backends(test_iterations=100):
        """
        性能基准测试
        
        Args:
            test_iterations: 测试迭代次数
        """
        import time
        from acc_decision import ACCCommand
        
        print("=== 决策模块性能基准测试 ===")
        
        backends_to_test = []
        if MATLAB_AVAILABLE:
            backends_to_test.append((DecisionBackend.MATLAB, "MATLAB"))
        backends_to_test.append((DecisionBackend.PYTHON, "Python"))
        
        results = {}
        
        for backend, name in backends_to_test:
            print(f"\n测试 {name} 版本...")
            
            # 创建决策模块
            decision_module = DecisionModuleFactory.create(backend=backend)
            decision_module.set_debug(False)  # 关闭调试输出以提高测试速度
            
            # 预热
            for _ in range(10):
                decision_module.process_command(ACCCommand.INCREASE_SPEED, 35.0, False)
            
            # 性能测试
            start_time = time.time()
            
            commands = [ACCCommand.INCREASE_SPEED, ACCCommand.DECREASE_SPEED, 
                       ACCCommand.INCREASE_DISTANCE, ACCCommand.DECREASE_DISTANCE,
                       ACCCommand.THROTTLE, ACCCommand.BRAKE, ACCCommand.CANCEL]
            
            for i in range(test_iterations):
                cmd = commands[i % len(commands)]
                speed = 30.0 + (i % 50)  # 变化速度 30-80 km/h
                has_target = (i % 3) == 0  # 随机有无前车
                
                try:
                    decision_module.process_command(cmd, speed, has_target)
                except Exception as e:
                    print(f"    ⚠️ 执行错误: {e}")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_time_per_call = (elapsed_time / test_iterations) * 1000  # ms
            
            results[name] = {
                'total_time': elapsed_time,
                'avg_time_ms': avg_time_per_call,
                'calls_per_sec': test_iterations / elapsed_time
            }
            
            print(f"  总时间: {elapsed_time:.3f}s")
            print(f"  平均每次调用: {avg_time_per_call:.3f}ms")
            print(f"  每秒调用数: {results[name]['calls_per_sec']:.1f}")
            
            # 清理
            if hasattr(decision_module, '__del__'):
                decision_module.__del__()
        
        # 比较结果
        if len(results) > 1:
            print(f"\n=== 性能对比 ===")
            python_time = results.get('Python', {}).get('avg_time_ms', 0)
            matlab_time = results.get('MATLAB', {}).get('avg_time_ms', 0)
            
            if python_time > 0 and matlab_time > 0:
                if python_time < matlab_time:
                    ratio = matlab_time / python_time
                    print(f"Python版本快 {ratio:.1f}x")
                else:
                    ratio = python_time / matlab_time
                    print(f"MATLAB版本快 {ratio:.1f}x")


class UnifiedDecisionModule:
    """
    统一决策模块接口
    提供后端切换能力和增强功能
    """
    
    def __init__(self, preferred_backend=DecisionBackend.AUTO, **kwargs):
        """
        初始化统一决策模块
        
        Args:
            preferred_backend: 首选后端
            **kwargs: 决策模块参数
        """
        self.current_backend = None
        self.decision_module = None
        self.fallback_module = None
        
        # 创建主决策模块
        self._create_decision_module(preferred_backend, **kwargs)
        
        # 创建备用模块（用于故障切换）
        backup_backend = DecisionBackend.PYTHON if preferred_backend != DecisionBackend.PYTHON else DecisionBackend.MATLAB
        try:
            self.fallback_module = DecisionModuleFactory.create(backup_backend, **kwargs)
            self.fallback_module.set_debug(False)  # 备用模块默认不开启调试
        except Exception as e:
            print(f"⚠️ 备用决策模块创建失败: {e}")
            self.fallback_module = None
    
    def _create_decision_module(self, backend, **kwargs):
        """创建决策模块"""
        self.decision_module = DecisionModuleFactory.create(backend, **kwargs)
        self.current_backend = backend
    
    def process_command(self, command, ego_speed_kmh, has_target=False, current_distance=None):
        """
        处理ACC指令（支持故障切换）
        """
        try:
            return self.decision_module.process_command(command, ego_speed_kmh, has_target, current_distance)
        except Exception as e:
            print(f"⚠️ 主决策模块失败: {e}")
            
            if self.fallback_module:
                print("🔄 切换到备用决策模块")
                try:
                    return self.fallback_module.process_command(command, ego_speed_kmh, has_target, current_distance)
                except Exception as e2:
                    print(f"❌ 备用决策模块也失败: {e2}")
                    raise
            else:
                raise
    
    def get_current_parameters(self):
        """获取当前参数"""
        return self.decision_module.get_current_parameters()
    
    def get_decision_output(self, ego_speed_kmh, current_distance=None):
        """获取决策输出"""
        return self.decision_module.get_decision_output(ego_speed_kmh, current_distance)
    
    def set_debug(self, enable):
        """设置调试模式"""
        self.decision_module.set_debug(enable)
    
    def reset(self):
        """重置决策模块"""
        self.decision_module.reset()
        if self.fallback_module:
            self.fallback_module.reset()
    
    def switch_backend(self, new_backend):
        """
        动态切换决策后端
        
        Args:
            new_backend: 新的后端类型
        """
        if new_backend == self.current_backend:
            return
        
        # 保存当前状态
        current_params = self.decision_module.get_current_parameters()
        
        try:
            # 创建新的决策模块
            new_module = DecisionModuleFactory.create(new_backend,
                                                    initial_V3_kmh=current_params['V3_kmh'],
                                                    initial_G1_m=current_params['G1_m'])
            
            # 尝试同步状态（简化实现）
            # 在实际应用中可能需要更复杂的状态同步逻辑
            
            # 清理旧模块
            if hasattr(self.decision_module, '__del__'):
                self.decision_module.__del__()
            
            # 切换到新模块
            self.decision_module = new_module
            self.current_backend = new_backend
            
            print(f"✅ 决策后端已切换到: {new_backend.value}")
            
        except Exception as e:
            print(f"❌ 后端切换失败: {e}")
            raise
    
    def get_backend_info(self):
        """获取后端信息"""
        return {
            'current_backend': self.current_backend.value,
            'has_fallback': self.fallback_module is not None,
            'available_backends': [b.value for b in DecisionModuleFactory.get_available_backends()]
        }


def main():
    """主函数 - 演示决策模块工厂的使用"""
    print("=== ACC决策模块工厂演示 ===")
    
    # 显示可用后端
    available_backends = DecisionModuleFactory.get_available_backends()
    print(f"可用后端: {[b.value for b in available_backends]}")
    
    # 创建统一决策模块
    print("\n创建统一决策模块...")
    unified_decision = UnifiedDecisionModule(preferred_backend=DecisionBackend.AUTO)
    
    # 显示后端信息
    backend_info = unified_decision.get_backend_info()
    print(f"当前后端: {backend_info['current_backend']}")
    print(f"有备用模块: {backend_info['has_fallback']}")
    
    # 测试基本功能
    print("\n测试基本功能...")
    from acc_decision import ACCCommand
    
    test_commands = [
        (ACCCommand.THROTTLE, 35.0, False),
        (ACCCommand.INCREASE_SPEED, 40.0, False),
        (ACCCommand.DECREASE_SPEED, 35.0, True),
        (ACCCommand.BRAKE, 30.0, True)
    ]
    
    for cmd, speed, has_target in test_commands:
        try:
            state, mode, msg = unified_decision.process_command(cmd, speed, has_target)
            print(f"  {cmd.value}: {state.value} -> {mode.value if mode else None}")
        except Exception as e:
            print(f"  {cmd.value}: 失败 - {e}")
    
    # 性能基准测试
    if len(available_backends) > 1:
        print(f"\n运行性能基准测试...")
        DecisionModuleFactory.benchmark_backends(test_iterations=50)


if __name__ == "__main__":
    main()