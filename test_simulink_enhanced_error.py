#!/usr/bin/env python3
"""
测试Simulink的enhanced_error输出
验证Port 4是否正确输出 enhanced_error = error_value + stage_offset
"""

import sys
import time
sys.path.insert(0, '/home/aiden/PycharmProjects/CarlaAcc')

from integrated_simulink_manager import IntegratedSimulinkManager
from acc_config import ACCConfig

def test_simulink_enhanced_error():
    print("="*80)
    print("测试Simulink Enhanced Error输出")
    print("="*80)

    # 初始化
    config = ACCConfig()
    manager = IntegratedSimulinkManager(config)

    print("\n[初始化完成]\n")

    # 测试场景1：有误差，control_enabled=True
    print("【测试1：有误差输入，模拟在控状态】")
    print("-"*60)

    # 先发送一次让系统进入在控状态
    test_input_1 = {
        'ego_speed_kmh': 80.0,
        'ego_speed_ms': 80.0 / 3.6,
        'control_error': 2.5,  # 时距误差2.5秒
        'control_mode_flag': 1,  # TIME模式
        'command_type': 0,
        'command_active': False,
        'manual_throttle_active': False,
        'V_target_kmh': 120.0,
        'V_min_kmh': 30.0,
        'G2_s': 4.0,
        'timestamp': time.time()
    }

    print(f"输入: control_error={test_input_1['control_error']:.3f}s (TIME模式)")

    # 先直接测试UDP获取原始数据
    print("\n  === 准备发送到Simulink的输入 (9个double) ===")
    inputs = manager._prepare_simulink_inputs(test_input_1)
    print(f"  inputs[4] (error_value): {inputs[4]:.6f}")
    print(f"  inputs[5] (stage_offset): {inputs[5]:.6f}")

    # 直接调用Simulink获取原始输出
    print("\n  === 从Simulink接收的原始输出 (10个double) ===")
    try:
        raw_outputs = manager._call_simulink(inputs)
        for i, val in enumerate(raw_outputs):
            if i == 8:
                print(f"  outputs[{i}] (enhanced_error/Port 4): {val:.6f}  ← 检查这个值")
            else:
                print(f"  outputs[{i}]: {val:.6f}")
    except Exception as e:
        print(f"  调用Simulink失败: {e}")

    output_1 = manager.process_cycle(test_input_1)

    print(f"输出:")
    print(f"\n  === Decision输出 (1-5) ===")
    print(f"  current_state: {output_1.get('current_state')} (0=在控, 2=无史待命)")
    print(f"  current_decision: {output_1.get('current_decision')}")
    print(f"  control_enabled: {output_1.get('control_enabled')}")
    print(f"  next_has_history: {output_1.get('next_has_history', 'N/A')}")
    print(f"  next_last_active_decision: {output_1.get('next_last_active_decision', 'N/A')}")

    print(f"\n  === SPPVT输出 (6-10) ===")
    print(f"  sppvt_control_output: {output_1.get('sppvt_control_output'):.6f}")
    print(f"  sppvt_velocity_output: {output_1.get('sppvt_velocity_output'):.6f}")
    print(f"  sppvt_acceleration_output: {output_1.get('sppvt_acceleration_output'):.6f}")
    print(f"  sppvt_enhanced_error: {output_1.get('sppvt_enhanced_error'):.6f}  ← 这是Port 4")
    print(f"  sppvt_status_output: {output_1.get('sppvt_status_output'):.6f}")

    print(f"\n  === Python状态 ===")
    print(f"  stage_offset: {manager.sppvt_state['stage_offset']:.6f}")

    expected_enhanced_error = test_input_1['control_error'] + manager.sppvt_state['stage_offset']
    print(f"\n预期 enhanced_error = error_value + stage_offset")
    print(f"                    = {test_input_1['control_error']:.3f} + {manager.sppvt_state['stage_offset']:.3f}")
    print(f"                    = {expected_enhanced_error:.3f}")
    print(f"实际 enhanced_error = {output_1.get('sppvt_enhanced_error'):.6f}")

    if abs(output_1.get('sppvt_enhanced_error', 0) - expected_enhanced_error) < 0.01:
        print("✓ 测试通过")
    else:
        print("✗ 测试失败：enhanced_error不匹配！")

    # 测试场景2：连续运行几步，观察enhanced_error变化
    print("\n\n【测试2：连续运行10步，观察所有SPPVT输出】")
    print("-"*120)
    print(f"{'Step':<6} {'error':<8} {'ctrl_out':<10} {'velocity':<10} {'accel':<10} {'enhanced':<12} {'upgrade':<8} {'offset':<8}")
    print("-"*120)

    for step in range(10):
        test_input = {
            'ego_speed_kmh': 80.0,
            'ego_speed_ms': 80.0 / 3.6,
            'control_error': 2.5 - step * 0.1,  # 误差逐渐减小
            'control_mode_flag': 1,
            'command_type': 0,
            'command_active': False,
            'manual_throttle_active': False,
            'V_target_kmh': 120.0,
            'V_min_kmh': 30.0,
            'G2_s': 4.0,
            'timestamp': time.time()
        }

        output = manager.process_cycle(test_input)

        print(f"{step:<6} "
              f"{test_input['control_error']:<8.3f} "
              f"{output.get('sppvt_control_output', 0):<10.6f} "
              f"{output.get('sppvt_velocity_output', 0):<10.6f} "
              f"{output.get('sppvt_acceleration_output', 0):<10.6f} "
              f"{output.get('sppvt_enhanced_error', 0):<12.6f} "
              f"{output.get('sppvt_status_output', 0):<8.1f} "
              f"{manager.sppvt_state['stage_offset']:<8.3f}")

    # 测试场景3：无前车场景（200m外）
    print("\n\n【测试3：无前车场景（距离>200m）】")
    print("-"*60)

    test_input_3 = {
        'ego_speed_kmh': 80.0,
        'ego_speed_ms': 80.0 / 3.6,
        'control_error': 0.0,  # 无前车，control_error=0
        'control_mode_flag': 2,  # SPEED模式
        'command_type': 0,
        'command_active': False,
        'manual_throttle_active': False,
        'V_target_kmh': 120.0,
        'V_min_kmh': 30.0,
        'G2_s': 4.0,
        'timestamp': time.time()
    }

    print(f"输入: control_error={test_input_3['control_error']:.3f} (SPEED模式，无前车)")

    output_3 = manager.process_cycle(test_input_3)

    print(f"输出:")
    print(f"  control_enabled: {output_3.get('control_enabled')}")
    print(f"\n  === SPPVT所有输出 ===")
    print(f"  sppvt_control_output: {output_3.get('sppvt_control_output'):.6f}")
    print(f"  sppvt_velocity_output: {output_3.get('sppvt_velocity_output'):.6f}")
    print(f"  sppvt_acceleration_output: {output_3.get('sppvt_acceleration_output'):.6f}")
    print(f"  sppvt_enhanced_error: {output_3.get('sppvt_enhanced_error'):.6f}")
    print(f"  sppvt_status_output: {output_3.get('sppvt_status_output'):.6f}")
    print(f"  stage_offset: {manager.sppvt_state['stage_offset']:.6f}")

    print("\n" + "="*80)
    print("测试完成")
    print("="*80)

    # 清理
    manager.reset()

if __name__ == "__main__":
    try:
        test_simulink_enhanced_error()
    except KeyboardInterrupt:
        print("\n\n测试被中断")
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()
