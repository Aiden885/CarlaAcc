"""
自动测试E键功能（无需手动输入）
"""
import sys
import time
from simulink_udp_interface import SimulinkUDPClient

def test_decision_states():
    """测试决策模型的状态转换"""
    print("="*70)
    print("ACC决策状态调试（自动测试）")
    print("="*70)

    try:
        client = SimulinkUDPClient(
            send_port=25000,
            recv_port=25001,
            num_inputs=4,
            num_outputs=5,
            timeout=1.0,
            debug=True,
            local_send_port=9090  # 绑定固定源端口
        )

        # 初始状态（应该是S2: 无史待命）
        current_state = 2.0  # S2
        has_history = 0.0
        last_decision = 8.0  # R8: 系统待命

        print("\n测试场景1: 初始状态（无指令）")
        print("-"*70)
        command_type = 0.0  # NONE
        inputs = [current_state, command_type, has_history, last_decision]
        client.call(inputs)  # 预热调用
        outputs = client.call(inputs)

        print(f"输入: state={int(current_state)}, cmd={int(command_type)}, history={int(has_history)}, last_dec={int(last_decision)}")
        print(f"输出: next_state={int(outputs[0])}, decision={int(outputs[1])}, enabled={int(outputs[2])}")

        if int(outputs[0]) == 2:
            print("✅ 正确：状态保持S2")
        else:
            print(f"⚠️  注意：状态变成了S{int(outputs[0])}")

        if int(outputs[2]) == 0:
            print("✅ 正确：control_enabled=False（待命状态）")
        else:
            print("❌ 错误！没有按键但control_enabled=True")

        # 更新状态
        current_state = outputs[0]
        has_history = outputs[3]
        last_decision = outputs[4]

        print("\n测试场景2: 按E键（当速启控，I0）")
        print("-"*70)
        command_type = 1.0  # I0: E键降速/当速启控
        inputs = [current_state, command_type, has_history, last_decision]
        client.call(inputs)  # 预热调用
        outputs = client.call(inputs)

        print(f"输入: state={int(current_state)}, cmd={int(command_type)}, history={int(has_history)}, last_dec={int(last_decision)}")
        print(f"输出: next_state={int(outputs[0])}, decision={int(outputs[1])}, enabled={int(outputs[2])}")

        success = True

        if int(outputs[2]) == 1:
            print("✅ 正确：control_enabled=True（应该进入控制状态）")
        else:
            print("❌ 错误！按了E键但control_enabled=False")
            success = False

        # 检查是否进入S0（在控状态）
        if int(outputs[0]) == 0:
            print("✅ 正确：进入S0（在控状态）")
        else:
            print(f"❌ 错误：状态是S{int(outputs[0])}，不是S0")
            success = False

        # 检查decision
        if int(outputs[1]) == 5:
            print("✅ 正确：decision=R5（当速启控）")
        else:
            print(f"❌ 错误：decision=R{int(outputs[1])}，不是R5")
            success = False

        # 更新状态
        current_state = outputs[0]
        has_history = outputs[3]
        last_decision = outputs[4]

        print("\n测试场景3: 在控状态下无指令")
        print("-"*70)
        command_type = 0.0  # NONE
        inputs = [current_state, command_type, has_history, last_decision]
        client.call(inputs)  # 预热调用
        outputs = client.call(inputs)

        print(f"输入: state={int(current_state)}, cmd={int(command_type)}, history={int(has_history)}, last_dec={int(last_decision)}")
        print(f"输出: next_state={int(outputs[0])}, decision={int(outputs[1])}, enabled={int(outputs[2])}")

        if int(outputs[2]) == 1:
            print("✅ 正确：在控状态保持control_enabled=True")
        else:
            print("❌ 错误！在控状态但control_enabled=False")
            success = False

        print("\n测试场景4: 按C键（取消，I6）")
        print("-"*70)
        command_type = 7.0  # I6: C键取消
        inputs = [current_state, command_type, has_history, last_decision]
        client.call(inputs)  # 预热调用
        outputs = client.call(inputs)

        print(f"输入: state={int(current_state)}, cmd={int(command_type)}, history={int(has_history)}, last_dec={int(last_decision)}")
        print(f"输出: next_state={int(outputs[0])}, decision={int(outputs[1])}, enabled={int(outputs[2])}")

        if int(outputs[2]) == 0:
            print("✅ 正确：control_enabled=False（退出控制）")
        else:
            print("❌ 错误！按了C键但还在控制")

        # 检查是否回到待命状态
        if int(outputs[0]) in [1, 2]:
            print(f"✅ 正确：回到S{int(outputs[0])}（待命状态）")
        else:
            print(f"⚠️  注意：状态是S{int(outputs[0])}")

        client.cleanup()

        print("\n" + "="*70)
        if success:
            print("✅✅✅ 测试通过！E键功能正常！")
        else:
            print("❌❌❌ 测试失败！E键功能仍有问题")
        print("="*70)
        return success

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == '__main__':
    success = test_decision_states()
    sys.exit(0 if success else 1)
