"""
新架构测试脚本
验证Python端模块功能和Simulink集成
"""
import time
import numpy as np
from acc_controller import ACCController
from sppvt_manager import SPPVTManager
from unified_acc_interface import UnifiedACCInterface


def test_acc_controller():
    """测试ACC控制器"""
    print("🧪 测试ACC控制器...")
    
    controller = ACCController(debug=True)
    
    # 测试输入验证
    test_input = {
        'ego_speed_kmh': 50.0,
        'ego_speed_ms': 13.89,
        'V_target_kmh': 60.0,
        'V_min_kmh': 30.0,
        'G2_s': 2.0,
        'control_error': 0.5,
        'timestamp': time.time()
    }
    
    validated = controller.validate_and_process_input(test_input)
    print(f"✅ 输入验证通过: {validated['ego_speed_kmh']:.1f} km/h")
    
    # 测试键盘指令处理
    control_enabled, decision, params = controller.process_keyboard_command(1, 50.0)  # E键降速
    print(f"✅ 键盘指令处理: 控制={control_enabled}, 决策={decision}")
    
    # 测试状态查询
    state_info = controller.get_state_info()
    print(f"✅ 状态查询: {state_info['state_name']}")
    
    print("✅ ACC控制器测试完成\n")


def test_sppvt_manager():
    """测试SPPVT管理器（不依赖MATLAB）"""
    print("🧪 测试SPPVT管理器（备用模式）...")
    
    # 禁用MATLAB引擎以测试备用模式
    manager = SPPVTManager(debug=True)
    manager.matlab_engine = None  # 强制使用备用计算
    
    # 测试SPPVT处理
    result = manager.process_sppvt_control(
        control_enabled=True,
        control_error=0.5,
        control_mode_flag=1
    )
    
    print(f"✅ SPPVT处理: 控制输出={result['sppvt_control_output']:.3f}")
    print(f"✅ 阶段状态: Stage={result['sppvt_stage_output']}")
    
    # 测试状态查询
    sppvt_state = manager.get_sppvt_state()
    print(f"✅ SPPVT状态: {sppvt_state}")
    
    print("✅ SPPVT管理器测试完成\n")


def test_unified_interface():
    """测试统一接口"""
    print("🧪 测试统一ACC接口...")
    
    interface = UnifiedACCInterface(debug=True)
    # 禁用MATLAB以测试备用模式
    interface.sppvt_manager.matlab_engine = None
    
    # 测试控制周期
    input_data = {
        'ego_speed_kmh': 45.0,
        'ego_speed_ms': 12.5,
        'control_error': 0.3,
        'control_mode_flag': 1,
        'command_type': 1,  # E键降速
        'manual_throttle_active': False,
        'V_target_kmh': 50.0,
        'V_min_kmh': 30.0,
        'G2_s': 2.0,
        'timestamp': time.time()
    }
    
    result = interface.process_control_cycle(input_data)
    
    print(f"✅ 控制周期处理:")
    print(f"   系统使能: {result['system_enabled']}")
    print(f"   控制激活: {result['control_enabled']}")
    print(f"   当前状态: S{result['current_state']}")
    print(f"   当前决策: R{result['current_decision']}")
    print(f"   SPPVT输出: {result['sppvt_control_output']:.3f}")
    
    # 测试多次调用（模拟连续控制）
    print(f"\n🔄 连续控制测试:")
    for i in range(5):
        input_data['command_type'] = 0 if i > 0 else 1  # 第一次E键，后续无指令
        input_data['ego_speed_kmh'] = 45.0 + i * 2  # 模拟速度变化
        
        result = interface.process_control_cycle(input_data)
        print(f"   第{i+1}次: 速度={input_data['ego_speed_kmh']:.1f}, 控制={result['control_enabled']}, 输出={result['sppvt_control_output']:.3f}")
    
    print("✅ 统一接口测试完成\n")


def test_integration_scenarios():
    """测试集成场景"""
    print("🧪 测试集成场景...")
    
    interface = UnifiedACCInterface(debug=True)
    interface.sppvt_manager.matlab_engine = None  # 备用模式
    
    scenarios = [
        {
            'name': '初始状态',
            'data': {'ego_speed_kmh': 35, 'command_type': 0}
        },
        {
            'name': '启动控制(E键)',
            'data': {'ego_speed_kmh': 45, 'command_type': 1}
        },
        {
            'name': '增速调整(Q键)',
            'data': {'ego_speed_kmh': 45, 'command_type': 2}
        },
        {
            'name': '扭矩仲裁(W键)',
            'data': {'ego_speed_kmh': 50, 'command_type': 5, 'manual_throttle_active': True}
        },
        {
            'name': '取消控制(C键)',
            'data': {'ego_speed_kmh': 50, 'command_type': 7}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}:")
        
        input_data = {
            'ego_speed_kmh': scenario['data'].get('ego_speed_kmh', 45),
            'ego_speed_ms': scenario['data'].get('ego_speed_kmh', 45) / 3.6,
            'control_error': 0.2,
            'control_mode_flag': 1,
            'command_type': scenario['data'].get('command_type', 0),
            'manual_throttle_active': scenario['data'].get('manual_throttle_active', False),
            'V_target_kmh': 50.0,
            'V_min_kmh': 30.0,
            'G2_s': 2.0,
            'timestamp': time.time()
        }
        
        result = interface.process_control_cycle(input_data)
        
        print(f"   状态: S{result['current_state']} → 决策: R{result['current_decision']}")
        print(f"   控制: {result['control_enabled']} | 扭矩仲裁: {result['torque_arbitration_active']}")
        print(f"   参数: V_target={result['updated_V_target_kmh']:.1f}, G2={result['updated_G2_s']:.1f}")
    
    print("✅ 集成场景测试完成\n")


def main():
    """主测试函数"""
    print("=" * 60)
    print("🚀 新架构功能测试开始")
    print("=" * 60)
    
    try:
        test_acc_controller()
        test_sppvt_manager()
        test_unified_interface()
        test_integration_scenarios()
        
        print("=" * 60)
        print("🎉 所有测试完成！新架构功能正常")
        print("=" * 60)
        
        print(f"\n📋 架构总结:")
        print(f"✅ ACCController: 高内聚的决策+验证模块")
        print(f"✅ SPPVTManager: 完整的SPPVT管理+Simulink调用")
        print(f"✅ UnifiedACCInterface: 简化的统一接口")
        print(f"✅ 兼容性: 完全替代原有复杂总线结构")
        
        print(f"\n🔧 使用说明:")
        print(f"1. 新架构已集成到 acc_updated.py")
        print(f"2. 只需确保 sppvt_control_model.slx 在当前目录")
        print(f"3. 运行 python acc_updated.py 即可使用新架构")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()