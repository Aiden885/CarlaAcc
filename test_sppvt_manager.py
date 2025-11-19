"""
测试SPPVT Manager的Simulink调用
验证是否能够正确获取SPPVT计算输出
"""

import sys
import time

try:
    from sppvt_manager_python import SPPVTManager
    print("✅ 成功导入 SPPVTManager")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def test_sppvt_manager():
    """测试SPPVT Manager"""
    print("\n" + "="*80)
    print("SPPVT Manager 测试")
    print("="*80)

    # 1. 创建SPPVTManager实例
    print("\n[步骤1] 创建SPPVTManager实例...")
    try:
        sppvt_mgr = SPPVTManager()
        sppvt_mgr.debug = True  # 启用调试输出
        print("✅ SPPVTManager实例创建成功")
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        return False

    # 2. 初始化MATLAB引擎
    print("\n[步骤2] 初始化MATLAB引擎和Simulink模型...")
    try:
        if not sppvt_mgr.initialize_matlab_engine():
            print("❌ MATLAB引擎初始化失败")
            return False
        print("✅ MATLAB引擎和模型初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 准备测试数据
    print("\n[步骤3] 准备测试数据...")

    # 模拟ACC决策输出
    decision_output = {
        'control_enabled': True,  # 控制使能
        'current_state': 0,       # S0 - IN_CONTROL
        'current_decision': 1,    # R1决策
    }

    # 模拟验证后的输入数据
    validated_input = {
        'control_error': -0.5,    # 控制误差: -0.5s (负值表示跟车距离过近)
        'control_mode_flag': 1,   # TIME模式
        'ego_speed_kmh': 50.0,    # 自车速度 50km/h
    }

    print(f"  decision_output: {decision_output}")
    print(f"  validated_input: control_error={validated_input['control_error']}, "
          f"mode_flag={validated_input['control_mode_flag']}")

    # 4. 调用SPPVT控制
    print("\n[步骤4] 调用process_sppvt_control...")
    try:
        start_time = time.time()
        output = sppvt_mgr.process_sppvt_control(decision_output, validated_input)
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"✅ SPPVT控制调用成功 (耗时: {elapsed_ms:.1f}ms)")

    except Exception as e:
        print(f"❌ SPPVT控制调用失败: {e}")
        import traceback
        traceback.print_exc()
        sppvt_mgr.cleanup()
        return False

    # 5. 验证输出结果
    print("\n[步骤5] 验证输出结果...")
    print("-"*80)

    # 检查关键输出字段
    required_fields = [
        'sppvt_control_output',
        'sppvt_velocity_output',
        'sppvt_acceleration_output',
        'sppvt_jerk_output',
        'sppvt_stage_output',
        'new_stage_offset',
        'target_accel',
    ]

    all_present = True
    for field in required_fields:
        if field in output:
            value = output[field]
            print(f"  ✓ {field:30s} = {value:.6f}")
        else:
            print(f"  ✗ {field:30s} = <缺失>")
            all_present = False

    print("-"*80)

    # 检查数值合理性
    print("\n[步骤6] 检查输出数值合理性...")
    issues = []

    control_output = output.get('sppvt_control_output', 0.0)
    velocity_output = output.get('sppvt_velocity_output', 0.0)
    acceleration_output = output.get('sppvt_acceleration_output', 0.0)
    jerk_output = output.get('sppvt_jerk_output', 0.0)

    # 检查是否为NaN
    import math
    if math.isnan(control_output):
        issues.append("control_output 是 NaN")
    if math.isnan(velocity_output):
        issues.append("velocity_output 是 NaN")
    if math.isnan(acceleration_output):
        issues.append("acceleration_output 是 NaN")
    if math.isnan(jerk_output):
        issues.append("jerk_output 是 NaN")

    # 检查control_output是否为0（对于非零误差输入）
    if abs(control_output) < 1e-9 and abs(validated_input['control_error']) > 0.1:
        issues.append(f"control_output为0，但控制误差为{validated_input['control_error']}")

    if issues:
        print("  ❌ 发现问题:")
        for issue in issues:
            print(f"     - {issue}")
        success = False
    else:
        print("  ✅ 所有输出数值正常")
        success = all_present

    # 7. 清理资源
    print("\n[步骤7] 清理资源...")
    try:
        sppvt_mgr.cleanup()
        print("✅ 资源清理完成")
    except Exception as e:
        print(f"⚠️ 清理时出现警告: {e}")

    # 总结
    print("\n" + "="*80)
    if success:
        print("✅ 测试通过！SPPVT Manager能够正确调用并获取Simulink输出")
    else:
        print("❌ 测试失败！存在问题需要修复")
    print("="*80 + "\n")

    return success


if __name__ == "__main__":
    try:
        success = test_sppvt_manager()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)