#!/usr/bin/env python3
"""
测试Two Mode控制器与Simulink SPPVT的集成
验证修改后的two_mode_controller.py是否正常工作
"""

import sys
import traceback

def test_two_mode_simulink_integration():
    """测试Two Mode控制器的Simulink集成"""
    print("=== Two Mode + Simulink SPPVT 集成测试 ===\n")

    try:
        # 1. 测试导入
        print("1. 测试导入...")
        from two_mode_controller import two_mode_control, TwoModeController
        print("   OK 导入成功")

        # 2. 测试控制器初始化
        print("\n2. 测试控制器初始化...")
        controller = TwoModeController()
        print("   OK 控制器初始化成功")
        print(f"   - V_threshold: {controller.V_threshold * 3.6:.1f}km/h")
        print(f"   - G2: {controller.G2:.1f}s")
        print(f"   - SPPVT管理器: {type(controller.sppvt_manager).__name__}")
        print(f"   - 默认control_mode_flag: {controller.control_mode_flag}")

        # 3. 测试TIME模式 (低速，有前车)
        print("\n3. 测试TIME模式 (30km/h, 距离20m)...")
        ego_speed = 30 / 3.6  # 30km/h 转 m/s
        control_output, info = two_mode_control(
            ego_speed_ms=ego_speed,
            current_distance=20.0,
            target_speed_ms=50/3.6
        )
        print(f"   OK TIME模式测试成功")
        print(f"   - 控制输出: {control_output:.3f} m/s^2")
        print(f"   - 模式: {info['mode']}")
        print(f"   - 误差: {info['error']:.3f}")
        print(f"   - 控制器模式标志: {controller.control_mode_flag}")

        # 4. 测试SPEED模式 (高速)
        print("\n4. 测试SPEED模式 (70km/h)...")
        ego_speed = 70 / 3.6  # 70km/h 转 m/s
        control_output, info = two_mode_control(
            ego_speed_ms=ego_speed,
            current_distance=None,  # 无前车
            target_speed_ms=60/3.6
        )
        print(f"   OK SPEED模式测试成功")
        print(f"   - 控制输出: {control_output:.3f} m/s^2")
        print(f"   - 模式: {info['mode']}")
        print(f"   - 误差: {info['error']:.3f}")
        print(f"   - 控制器模式标志: {controller.control_mode_flag}")

        # 5. 测试模式切换
        print("\n5. 测试模式切换 (45km/h → 55km/h)...")

        # 低速TIME模式
        ego_speed1 = 45 / 3.6
        control_output1, info1 = two_mode_control(
            ego_speed_ms=ego_speed1,
            current_distance=25.0,
            target_speed_ms=50/3.6
        )
        mode1 = info1['mode']

        # 高速SPEED模式
        ego_speed2 = 55 / 3.6
        control_output2, info2 = two_mode_control(
            ego_speed_ms=ego_speed2,
            current_distance=None,
            target_speed_ms=50/3.6
        )
        mode2 = info2['mode']

        print(f"   OK 模式切换测试成功")
        print(f"   - 45km/h: {mode1}")
        print(f"   - 55km/h: {mode2}")

        print("\n=== 集成测试完成 ===")
        print("OK Two Mode控制器成功集成Simulink SPPVT")
        return True

    except Exception as e:
        print(f"\nERROR 测试失败: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return False

def test_acc_planning_import():
    """测试ACC规划控制导入"""
    print("\n=== ACC规划控制导入测试 ===")

    try:
        from acc_planning_control import ACCPlanningControl
        print("OK ACC规划控制导入成功")
        return True
    except Exception as e:
        print(f"ERROR ACC规划控制导入失败: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始Two Mode + Simulink SPPVT集成验证...\n")

    # 测试Two Mode集成
    test1_result = test_two_mode_simulink_integration()

    # 测试ACC规划控制导入
    test2_result = test_acc_planning_import()

    print(f"\n{'='*50}")
    print("最终测试结果:")
    print(f"- Two Mode + Simulink集成: {'OK 成功' if test1_result else 'ERROR 失败'}")
    print(f"- ACC规划控制导入: {'OK 成功' if test2_result else 'ERROR 失败'}")

    if test1_result and test2_result:
        print("\n SUCCESS: 所有集成测试通过！可以进行完整CARLA测试。")
        sys.exit(0)
    else:
        print("\n WARNING: 存在集成问题，需要修复后再进行CARLA测试。")
        sys.exit(1)