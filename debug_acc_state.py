"""
ACC状态调试工具
帮助追踪为什么ACC控制异常
"""
import sys
import time
from simulink_udp_interface import SimulinkUDPClient

def test_decision_states():
    """测试决策模型的状态转换"""
    print("="*70)
    print("ACC决策状态调试")
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

        if int(outputs[2]) == 1:
            print("❌ 错误！没有按键但control_enabled=True")
            print("   这意味着系统自动开始控制了")
        else:
            print("✅ 正确：control_enabled=False（待命状态）")

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

        if int(outputs[2]) == 1:
            print("✅ 正确：control_enabled=True（应该进入控制状态）")
        else:
            print("❌ 错误！按了E键但control_enabled=False")
            print("   这意味着按键没有生效")

        # 检查是否进入S0（在控状态）
        if int(outputs[0]) == 0:
            print("✅ 正确：进入S0（在控状态）")
        else:
            print(f"⚠️  注意：状态是S{int(outputs[0])}，不是S0")

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
        print("测试完成")
        print("="*70)
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("\n⚠️  请确保:")
    print("  1. Simulink决策模型已启动")
    print("  2. UDP端口配置正确（25000/25001）")

    input("\n按回车键开始测试...")

    if test_decision_states():
        print("\n✅ 如果所有测试都通过，说明Simulink决策模型工作正常")
        print("   问题可能在Python端的按键处理或状态同步")
        print("\n检查项目:")
        print("  1. 确认acc_system_enabled状态（按空格键切换）")
        print("  2. 检查pending_keyboard_command是否正确传递")
        print("  3. 检查初始化时决策模块的状态")
    else:
        print("\n❌ Simulink模型通信有问题")
        print("   请确认模型已启动并配置正确")

if __name__ == '__main__':
    main()
