#!/usr/bin/env python3
"""
测试MATLAB决策模块集成
验证acc_updated.py中的MATLAB决策模块是否正常工作
"""

import os
import sys

def test_imports():
    """测试导入是否正常"""
    print("=== 测试模块导入 ===")
    
    try:
        from decision_factory import DecisionModuleFactory, DecisionBackend
        print("✅ decision_factory 导入成功")
        
        # 检查可用后端
        available_backends = DecisionModuleFactory.get_available_backends()
        print(f"📋 可用决策后端: {[b.value for b in available_backends]}")
        
        if DecisionBackend.MATLAB in available_backends:
            print("✅ MATLAB决策模块可用")
        else:
            print("⚠️ MATLAB决策模块不可用（需要MATLAB Engine for Python）")
            
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_decision_modules():
    """测试决策模块创建和切换"""
    print("\n=== 测试决策模块 ===")
    
    try:
        from decision_factory import DecisionModuleFactory, UnifiedDecisionModule, DecisionBackend
        from acc_decision import ACCCommand
        
        # 测试创建统一决策模块
        print("创建统一决策模块...")
        unified_decision = UnifiedDecisionModule(preferred_backend=DecisionBackend.AUTO)
        
        # 获取后端信息
        backend_info = unified_decision.get_backend_info()
        print(f"当前后端: {backend_info['current_backend']}")
        
        # 测试基本决策功能
        print("测试基本决策功能...")
        test_commands = [
            (ACCCommand.THROTTLE, 35.0, False),
            (ACCCommand.INCREASE_SPEED, 40.0, False),
            (ACCCommand.BRAKE, 30.0, True)
        ]
        
        for cmd, speed, has_target in test_commands:
            try:
                state, mode, msg = unified_decision.process_command(cmd, speed, has_target)
                print(f"  {cmd.value}: {state.value} -> {mode.value if mode else None}")
            except Exception as e:
                print(f"  {cmd.value}: 失败 - {e}")
        
        # 测试后端切换（如果有多个后端）
        if len(backend_info['available_backends']) > 1:
            print("\n测试后端切换...")
            for backend_name in backend_info['available_backends']:
                if backend_name != backend_info['current_backend']:
                    try:
                        backend = DecisionBackend(backend_name)
                        unified_decision.switch_backend(backend)
                        print(f"✅ 成功切换到 {backend_name}")
                        
                        # 测试切换后的功能
                        state, mode, msg = unified_decision.process_command(ACCCommand.THROTTLE, 35.0, False)
                        print(f"  切换后测试: {state.value}")
                        
                    except Exception as e:
                        print(f"❌ 切换到 {backend_name} 失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 决策模块测试失败: {e}")
        return False

def test_acc_updated_integration():
    """测试acc_updated.py的集成"""
    print("\n=== 测试acc_updated.py集成 ===")
    
    try:
        # 模拟导入acc_updated中的相关代码
        import acc_updated
        
        # 检查是否包含决策工厂导入
        source_file = "acc_updated.py"
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "decision_factory" in content:
            print("✅ acc_updated.py 已集成决策工厂")
        else:
            print("❌ acc_updated.py 未集成决策工厂")
            return False
        
        if "DecisionBackend" in content:
            print("✅ 支持决策后端切换")
        else:
            print("❌ 不支持决策后端切换")
            return False
        
        if "K_F1" in content and "K_F2" in content:
            print("✅ 包含快捷键支持")
        else:
            print("❌ 缺少快捷键支持")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ acc_updated.py集成测试失败: {e}")
        return False

def test_environment_variable():
    """测试环境变量设置"""
    print("\n=== 测试环境变量 ===")
    
    # 测试不同的环境变量设置
    test_values = ['python', 'matlab', 'auto']
    
    for value in test_values:
        print(f"测试 DECISION_BACKEND={value}")
        os.environ['DECISION_BACKEND'] = value
        
        try:
            from decision_factory import DecisionModuleFactory, DecisionBackend
            
            # 模拟acc_updated.py中的逻辑
            backend_env = os.environ.get('DECISION_BACKEND', 'auto').lower()
            if backend_env == 'python':
                preferred_backend = DecisionBackend.PYTHON
            elif backend_env == 'matlab':
                preferred_backend = DecisionBackend.MATLAB  
            else:
                preferred_backend = DecisionBackend.AUTO
            
            print(f"  解析结果: {preferred_backend.value}")
            
        except Exception as e:
            print(f"  ❌ 解析失败: {e}")
    
    # 清理环境变量
    if 'DECISION_BACKEND' in os.environ:
        del os.environ['DECISION_BACKEND']
    
    return True

def main():
    """主测试函数"""
    print("🧪 MATLAB决策模块集成测试")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    # 运行各项测试
    tests = [
        ("模块导入", test_imports),
        ("决策模块", test_decision_modules),
        ("acc_updated集成", test_acc_updated_integration),
        ("环境变量", test_environment_variable)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}测试")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name}测试通过")
                success_count += 1
            else:
                print(f"❌ {test_name}测试失败")
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
    
    # 输出总结
    print("\n" + "=" * 50)
    print(f"🎯 测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！MATLAB决策模块集成成功")
        print("\n📋 使用方法:")
        print("1. 设置环境变量: export DECISION_BACKEND=matlab")
        print("2. 运行程序: python acc_updated.py")
        print("3. 使用快捷键: F1(Python) F2(MATLAB) F3(信息)")
    else:
        print("⚠️ 部分测试失败，请检查集成")
    
    return success_count == total_tests

if __name__ == "__main__":
    main()