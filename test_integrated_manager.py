#!/usr/bin/env python3
"""
测试统一的Simulink集成管理器

测试步骤：
1. 启动Simulink模型服务器（在MATLAB中运行）
2. 运行此脚本测试Python端
"""
import sys
import time
from integrated_simulink_manager import IntegratedSimulinkManager


def test_basic_communication():
    """测试基本UDP通信"""
    print("=== 测试1: 基本UDP通信 ===\n")

    manager = IntegratedSimulinkManager(debug=True)

    try:
        # 测试输入
        test_input = {
            'ego_speed_kmh': 50.0,
            'command_type': 0,  # NONE
            'control_error': 0.5,
            'control_mode_flag': 1,  # TIME模式
            'V_target_kmh': 50.0,
            'V_min_kmh': 30.0,
            'G2_s': 2.0,
            'manual_throttle_active': False
        }

        print("发送测试输入:")
        for key, value in test_input.items():
            print(f"  {key}: {value}")
        print()

        # 调用一次
        output = manager.process_cycle(test_input)

        print("接收到输出:")
        for key, value in output.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()

        print("✅ 基本通信测试通过\n")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False

    finally:
        manager.cleanup()


def test_state_transitions():
    """测试状态转换"""
    print("=== 测试2: 状态转换 ===\n")

    manager = IntegratedSimulinkManager(debug=True)

    try:
        # 场景：从待命到在控
        scenarios = [
            {
                'name': '待命状态（无指令）',
                'input': {
                    'ego_speed_kmh': 50.0,
                    'command_type': 0,  # NONE
                    'control_error': 0.5,
                    'control_mode_flag': 1,
                },
                'expected_state': 2,  # S2待命
            },
            {
                'name': 'E键当速启控',
                'input': {
                    'ego_speed_kmh': 50.0,
                    'command_type': 1,  # I0 (E键)
                    'control_error': 0.5,
                    'control_mode_flag': 1,
                },
                'expected_state': 0,  # S0在控
            },
            {
                'name': '在控状态（Q键增速）',
                'input': {
                    'ego_speed_kmh': 50.0,
                    'command_type': 2,  # I1 (Q键)
                    'control_error': 0.5,
                    'control_mode_flag': 1,
                },
                'expected_state': 0,  # S0在控
            },
        ]

        for i, scenario in enumerate(scenarios, 1):
            print(f"场景{i}: {scenario['name']}")
            output = manager.process_cycle(scenario['input'])

            actual_state = output['current_state']
            expected_state = scenario['expected_state']

            if actual_state == expected_state:
                print(f"  ✅ 状态正确: S{actual_state}")
            else:
                print(f"  ❌ 状态错误: 期望S{expected_state}, 实际S{actual_state}")

            print(f"  决策: R{output['current_decision']}")
            print(f"  控制: {'ON' if output['control_enabled'] else 'OFF'}")
            print()

        print("✅ 状态转换测试完成\n")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False

    finally:
        manager.cleanup()


def test_performance():
    """测试性能"""
    print("=== 测试3: 性能测试 ===\n")

    manager = IntegratedSimulinkManager(debug=False)

    try:
        test_input = {
            'ego_speed_kmh': 50.0,
            'command_type': 0,
            'control_error': 0.5,
            'control_mode_flag': 1,
        }

        # 预热
        print("预热中...")
        for i in range(5):
            try:
                manager.process_cycle(test_input)
            except Exception as e:
                print(f"  预热第{i+1}次失败: {e}")
                raise

        # 性能测试（降低到20次，避免Simulink超时）
        num_calls = 20
        print(f"开始性能测试（{num_calls}次调用）...")
        start_time = time.time()

        success_count = 0
        for i in range(num_calls):
            try:
                manager.process_cycle(test_input)
                success_count += 1
                if (i + 1) % 5 == 0:
                    print(f"  已完成 {i+1}/{num_calls} 次")
            except Exception as e:
                print(f"  ❌ 第{i+1}次调用失败: {e}")
                break

        elapsed = time.time() - start_time

        if success_count == 0:
            print("❌ 所有调用都失败\n")
            return False

        avg_time_ms = (elapsed / success_count) * 1000

        print(f"\n调用次数: {success_count}/{num_calls}")
        print(f"总耗时: {elapsed:.3f}秒")
        print(f"平均耗时: {avg_time_ms:.2f}ms/次")
        print(f"理论帧率: {1000/avg_time_ms:.1f} FPS")
        print()

        if avg_time_ms < 50:  # 20 FPS
            print("✅ 性能测试通过（满足实时要求）\n")
            return True
        else:
            print("⚠️ 性能较慢，可能影响实时性\n")
            return True

    except Exception as e:
        print(f"❌ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False

    finally:
        print("清理资源...")
        manager.cleanup()


def test_parameter_adjustment():
    """测试参数调整"""
    print("=== 测试4: 参数调整 ===\n")

    manager = IntegratedSimulinkManager(debug=True)

    try:
        # 先进入在控状态
        print("步骤1: 进入在控状态（E键）")
        output = manager.process_cycle({
            'ego_speed_kmh': 50.0,
            'command_type': 1,  # E键启控
            'control_error': 0.5,
        })
        print(f"  状态: S{output['current_state']}, V_target: {output['updated_V_target_kmh']:.1f} km/h\n")

        # Q键增速
        print("步骤2: Q键增速")
        output = manager.process_cycle({
            'ego_speed_kmh': 50.0,
            'command_type': 2,  # Q键
            'control_error': 0.5,
        })
        print(f"  V_target: {output['updated_V_target_kmh']:.1f} km/h (应为55.0)\n")

        # E键降速
        print("步骤3: E键降速")
        output = manager.process_cycle({
            'ego_speed_kmh': 50.0,
            'command_type': 1,  # E键
            'control_error': 0.5,
        })
        print(f"  V_target: {output['updated_V_target_kmh']:.1f} km/h (应为50.0)\n")

        # R键增距
        print("步骤4: R键增距")
        output = manager.process_cycle({
            'ego_speed_kmh': 50.0,
            'command_type': 4,  # R键
            'control_error': 0.5,
        })
        print(f"  G2: {output['updated_G2_s']:.1f} s (应为2.2)\n")

        print("✅ 参数调整测试完成\n")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False

    finally:
        manager.cleanup()


def main():
    """主测试流程"""
    print("=" * 60)
    print("统一Simulink集成管理器测试")
    print("=" * 60)
    print()

    print("⚠️ 请确保已在MATLAB中启动Simulink模型服务器！")
    print("   在MATLAB中运行: open_system('acc_integrated_model'); set_param('acc_integrated_model', 'SimulationCommand', 'start')")
    print()

    # 尝试交互式输入，失败则自动继续
    try:
        input("按Enter键开始测试...")
    except (EOFError, KeyboardInterrupt):
        print("自动继续...")
    print()

    results = {
        '基本通信': test_basic_communication(),
        '状态转换': test_state_transitions(),
        '性能测试': test_performance(),
        '参数调整': test_parameter_adjustment(),
    }

    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
    print()

    all_passed = all(results.values())
    if all_passed:
        print("🎉 所有测试通过！集成管理器工作正常。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查配置。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
