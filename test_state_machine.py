"""
测试状态机转移表实现
验证新的表驱动实现与原始硬编码实现行为完全一致
"""
import sys
from acc_controller import ACCController


def test_state_transitions():
    """测试所有状态转移"""
    print("\n" + "="*80)
    print("状态机转移表测试")
    print("="*80)

    controller = ACCController(debug=False)

    # 测试用例: (初始状态, has_history, 指令, 期望状态, 期望决策, 期望enabled, 期望has_history)
    test_cases = [
        # S2 (初始状态) + E → S0, R5, enabled, has_history=True
        (2, False, 1, 0, 5, True, True, "S2+E: 当速启控"),

        # S0 + E → S0, R1, enabled
        (0, True, 1, 0, 1, True, True, "S0+E: 降速"),

        # S0 + Q → S0, R2, enabled
        (0, True, 2, 0, 2, True, True, "S0+Q: 增速"),

        # S0 + T → S0, R3, enabled
        (0, True, 3, 0, 3, True, True, "S0+T: 降距"),

        # S0 + R → S0, R4, enabled
        (0, True, 4, 0, 4, True, True, "S0+R: 增距"),

        # S0 + W → S0, R7, enabled
        (0, True, 5, 0, 7, True, True, "S0+W: 扭矩仲裁"),

        # S0 + S → S1, R8, disabled
        (0, True, 6, 1, 8, False, True, "S0+S: 退出控制"),

        # S0 + C → S1, R8, disabled
        (0, True, 7, 1, 8, False, True, "S0+C: 取消"),

        # S1 + E → S0, R5, enabled
        (1, True, 1, 0, 5, True, True, "S1+E: 当速启控"),

        # S1 + Q → S0, R6, enabled
        (1, True, 2, 0, 6, True, True, "S1+Q: 继承启控"),

        # S1 + T → S1, R8, disabled (待命状态不响应)
        (1, True, 3, 1, 8, False, True, "S1+T: 待命不响应"),

        # S2 + Q → S2, R8, disabled (只有E能启控)
        (2, False, 2, 2, 8, False, False, "S2+Q: 待命不响应"),

        # S0 + cmd0 → S0, last_active_decision, enabled
        (0, True, 0, 0, 5, True, True, "S0+cmd0: 保持控制"),

        # S1 + cmd0 → S1, R8, disabled
        (1, True, 0, 1, 8, False, True, "S1+cmd0: 待命状态"),

        # S2 + cmd0 → S2, R8, disabled
        (2, False, 0, 2, 8, False, False, "S2+cmd0: 待命状态"),
    ]

    passed = 0
    failed = 0

    for initial_state, initial_history, command, expected_state, expected_decision, \
        expected_enabled, expected_history, description in test_cases:

        # 设置初始状态
        controller.current_state = initial_state
        controller.has_history = initial_history
        controller.last_active_decision = 5  # 假设上次是R5

        # 执行状态机
        enabled, decision, _ = controller.process_keyboard_command(command, 50.0)

        # 验证结果
        state_match = (controller.current_state == expected_state)
        decision_match = (decision == expected_decision)
        enabled_match = (enabled == expected_enabled)
        history_match = (controller.has_history == expected_history)

        all_match = state_match and decision_match and enabled_match and history_match

        if all_match:
            print(f"✅ {description}")
            passed += 1
        else:
            print(f"❌ {description}")
            if not state_match:
                print(f"   状态不匹配: 期望S{expected_state}, 实际S{controller.current_state}")
            if not decision_match:
                print(f"   决策不匹配: 期望R{expected_decision}, 实际R{decision}")
            if not enabled_match:
                print(f"   使能不匹配: 期望{expected_enabled}, 实际{enabled}")
            if not history_match:
                print(f"   历史不匹配: 期望{expected_history}, 实际{controller.has_history}")
            failed += 1

    print("\n" + "="*80)
    print(f"测试结果: {passed}通过, {failed}失败")
    print("="*80 + "\n")

    return failed == 0


def test_speed_based_transitions():
    """测试基于车速的自动状态转移"""
    print("\n" + "="*80)
    print("车速自动转移测试")
    print("="*80)

    controller = ACCController(debug=False)
    controller.params['V_min_kmh'] = 30.0

    # 测试低速转移
    controller.current_state = 0  # S0
    controller.has_history = True
    controller.process_keyboard_command(0, 20.0)  # 低于V_min

    if controller.current_state == 3:
        print("✅ 低速自动转入S3")
    else:
        print(f"❌ 低速转移失败: 期望S3, 实际S{controller.current_state}")
        return False

    # 测试恢复到S1 (有历史)
    controller.process_keyboard_command(0, 50.0)  # 高于V_min

    if controller.current_state == 1:
        print("✅ 车速恢复转入S1 (有历史)")
    else:
        print(f"❌ 恢复转移失败: 期望S1, 实际S{controller.current_state}")
        return False

    # 测试恢复到S2 (无历史)
    controller.current_state = 3
    controller.has_history = False
    controller.process_keyboard_command(0, 50.0)

    if controller.current_state == 2:
        print("✅ 车速恢复转入S2 (无历史)")
    else:
        print(f"❌ 恢复转移失败: 期望S2, 实际S{controller.current_state}")
        return False

    print("="*80 + "\n")
    return True


def test_parameter_adjustments():
    """测试参数调整逻辑"""
    print("\n" + "="*80)
    print("参数调整测试")
    print("="*80)

    controller = ACCController(debug=False)
    controller.current_state = 0  # S0 在控状态
    controller.params['V_target_kmh'] = 50.0
    controller.params['G2_s'] = 2.0

    # 测试E键降速
    _, _, params = controller.process_keyboard_command(1, 50.0)
    if 'V_target_kmh' in params and params['V_target_kmh'] == 45.0:
        print("✅ E键降速: 50 → 45 km/h")
    else:
        print(f"❌ E键降速失败: {params}")
        return False

    # 测试Q键增速
    _, _, params = controller.process_keyboard_command(2, 50.0)
    if 'V_target_kmh' in params and params['V_target_kmh'] == 50.0:
        print("✅ Q键增速: 45 → 50 km/h")
    else:
        print(f"❌ Q键增速失败: {params}")
        return False

    # 测试T键降距
    _, _, params = controller.process_keyboard_command(3, 50.0)
    if 'G2_s' in params and abs(params['G2_s'] - 1.8) < 0.01:
        print("✅ T键降距: 2.0 → 1.8 s")
    else:
        print(f"❌ T键降距失败: {params}")
        return False

    # 测试R键增距
    _, _, params = controller.process_keyboard_command(4, 50.0)
    if 'G2_s' in params and abs(params['G2_s'] - 2.0) < 0.01:
        print("✅ R键增距: 1.8 → 2.0 s")
    else:
        print(f"❌ R键增距失败: {params}")
        return False

    print("="*80 + "\n")
    return True


if __name__ == "__main__":
    try:
        success = True
        success &= test_state_transitions()
        success &= test_speed_based_transitions()
        success &= test_parameter_adjustments()

        if success:
            print("\n" + "="*80)
            print("🎉 所有测试通过！状态机转移表实现正确")
            print("="*80 + "\n")
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("❌ 部分测试失败")
            print("="*80 + "\n")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
