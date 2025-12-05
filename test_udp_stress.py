"""
UDP压力测试 - 模拟acc_updated.py的实际使用场景
测试Simulink模型在连续通信下的稳定性
"""
import time
import sys
from simulink_udp_interface import SimulinkUDPClient

def test_decision_model():
    """测试决策模型的状态转换逻辑"""
    print("\n" + "="*70)
    print("测试决策模型 (acc_decision_core)")
    print("="*70)

    try:
        client = SimulinkUDPClient(
            send_port=25000,
            recv_port=25001,
            num_inputs=4,
            num_outputs=5,
            timeout=1.0,  # 1秒超时
            debug=True,
            local_send_port=9090  # 绑定固定源端口
        )

        print("\n测试完整的ACC决策状态转换流程...")
        success_count = 0
        fail_count = 0

        # 定义测试场景
        test_scenarios = [
            # (描述, 重复次数, command_type, 期望state, 期望decision, 期望enabled)
            ("初始待命状态", 5, 0, 2, 8, 0),  # S2无史待命
            ("按E键启动控制", 1, 1, 0, 5, 1),  # S2+E → S0, R5(当速启控), enabled
            ("保持在控状态", 10, 0, 0, -1, 1),  # S0保持（decision=-1会使用last_active）
            ("在控时按E键降速", 1, 1, 0, 1, 1),  # S0+E → S0, R1(降速)
            ("在控时按Q键增速", 1, 2, 0, 2, 1),  # S0+Q → S0, R2(增速)
            ("在控时按T键降距", 1, 3, 0, 3, 1),  # S0+T → S0, R3(降距)
            ("在控时按R键增距", 1, 4, 0, 4, 1),  # S0+R → S0, R4(增距)
            ("继续在控状态", 5, 0, 0, -1, 1),
            ("按C键退出控制", 1, 7, 1, 8, 0),  # S0+C → S1(有史待命), R8
            ("有史待命状态", 5, 0, 1, 8, 0),  # S1保持待命
            ("再次按E键启控", 1, 1, 0, 5, 1),  # S1+E → S0, R5(当速启控)
            ("再次在控状态", 5, 0, 0, -1, 1),
            ("最终退出控制", 1, 7, 1, 8, 0),  # S0+C → S1
        ]

        # 初始状态
        current_state = 2.0  # S2: 无史待命
        has_history = 0.0
        last_decision = 8.0  # R8: 系统待命

        frame = 0
        for scenario_desc, repeat, cmd, exp_state, exp_decision, exp_enabled in test_scenarios:
            print(f"\n【{scenario_desc}】 (重复{repeat}次)")

            for r in range(repeat):
                command_type = float(cmd)
                inputs = [current_state, command_type, has_history, last_decision]

                try:
                    # ⚠️ 解决Simulink UDP延迟问题：
                    # 第一次调用的返回是上一帧的结果，丢弃
                    # 第二次调用才是当前输入的结果
                    client.call(inputs)  # 预热调用，丢弃结果
                    outputs = client.call(inputs)  # 实际调用，使用结果

                    success_count += 1
                    frame += 1

                    # 解析输出
                    out_state = int(round(outputs[0]))
                    out_decision = int(round(outputs[1]))
                    out_enabled = int(round(outputs[2]))
                    out_history = int(round(outputs[3]))
                    out_last_dec = int(round(outputs[4]))

                    # 验证输出
                    state_ok = (out_state == exp_state)
                    # decision=-1表示使用last_active_decision，不验证
                    decision_ok = (exp_decision == -1 or out_decision == exp_decision)
                    enabled_ok = (out_enabled == exp_enabled)

                    all_ok = state_ok and decision_ok and enabled_ok

                    if not all_ok or r == 0:  # 显示第一帧或错误帧
                        status = "✅" if all_ok else "❌"
                        print(f"  帧#{frame}: {status} cmd={cmd} → state=S{out_state}, dec=R{out_decision}, enabled={out_enabled}")

                        if not all_ok:
                            print(f"    期望: state=S{exp_state}, dec=R{exp_decision}, enabled={exp_enabled}")
                            fail_count += 1

                    # 更新状态
                    current_state = outputs[0]
                    has_history = outputs[3]
                    last_decision = outputs[4]

                except Exception as e:
                    fail_count += 1
                    print(f"  帧#{frame}: ❌ 失败 - {e}")
                    client.cleanup()
                    return False

                time.sleep(0.05)  # 20 FPS

        client.cleanup()

        print(f"\n" + "="*70)
        print(f"决策模型测试结果:")
        print(f"  成功: {success_count - fail_count}/{success_count}")
        print(f"  失败: {fail_count}/{success_count}")

        if fail_count == 0:
            print("✅ 所有状态转换都正确！")
        else:
            print("❌ 部分状态转换有误，请检查查找表配置")

        return fail_count == 0

    except Exception as e:
        print(f"❌ 决策模型初始化失败: {e}")
        return False


def test_sppvt_model():
    """测试SPPVT模型的连续通信"""
    print("\n" + "="*70)
    print("测试SPPVT模型 (sppvt_control_model)")
    print("="*70)

    try:
        client = SimulinkUDPClient(
            send_port=26000,
            recv_port=26001,
            num_inputs=5,
            num_outputs=5,
            timeout=1.0,  # 1秒超时
            debug=True,
            local_send_port=9091  # 绑定固定源端口
        )

        print("\n开始连续测试100帧...")
        success_count = 0
        fail_count = 0

        # 初始化SPPVT状态
        stage_offset = 0.0
        prev_error = 0.0
        prev_velocity = 0.0
        prev_accel = 0.0

        # 模拟从低速到高速的场景
        for i in range(100):
            # 模拟speed从0.1增加到20 m/s
            speed = 0.1 + i * 0.2
            distance_error = 50.0 - i * 0.3  # 距离误差逐渐减小

            # 计算control_error（时距误差）
            # 这是导致问题的关键！速度低时时距会很大
            control_error = distance_error / speed if speed > 0.5 else 0.0

            # 限制control_error范围（防止极端值）
            control_error = max(-100.0, min(100.0, control_error))

            inputs = [control_error, stage_offset, prev_error, prev_velocity, prev_accel]

            try:
                # ⚠️ 双调用解决Simulink UDP延迟问题
                client.call(inputs)  # 预热调用，丢弃结果
                outputs = client.call(inputs)
                success_count += 1

                # 检查输出是否合理
                control_out = outputs[0]
                velocity_out = outputs[1]
                accel_out = outputs[2]

                # 更新状态
                prev_error = control_error
                prev_velocity = velocity_out
                prev_accel = accel_out

                if i % 20 == 0:
                    print(f"  帧#{i}: 成功")
                    print(f"    输入: error={control_error:.2f}s, speed={speed:.2f}m/s")
                    print(f"    输出: control={control_out:.3f}, vel={velocity_out:.3f}, accel={accel_out:.3f}")

                # 检查输出是否异常
                if abs(velocity_out) > 100 or abs(accel_out) > 100:
                    print(f"  ⚠️  警告: 输出值过大! vel={velocity_out:.1f}, accel={accel_out:.1f}")

            except Exception as e:
                fail_count += 1
                print(f"  帧#{i}: ❌ 失败 - {e}")
                print(f"    当时输入: error={control_error:.2f}s, speed={speed:.2f}m/s")
                break  # 一次失败就停止

            time.sleep(0.05)  # 20 FPS

        client.cleanup()

        print(f"\nSPPVT模型测试结果:")
        print(f"  成功: {success_count}/100")
        print(f"  失败: {fail_count}/100")

        return fail_count == 0

    except Exception as e:
        print(f"❌ SPPVT模型初始化失败: {e}")
        return False


def main():
    print("="*70)
    print("Simulink UDP 压力测试")
    print("="*70)
    print("\n⚠️  请确保以下操作已完成:")
    print("  1. 在MATLAB中运行: test_simulink_config.m")
    print("  2. 确认两个模型配置正确")
    print("  3. 启动两个Simulink模型")
    print("     - set_param('acc_decision_core', 'SimulationCommand', 'start');")
    print("     - set_param('sppvt_control_model', 'SimulationCommand', 'start');")

    input("\n按回车键开始测试...")

    # 测试决策模型
    decision_ok = test_decision_model()

    # 等待一下
    time.sleep(1)

    # 测试SPPVT模型
    sppvt_ok = test_sppvt_model()

    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)

    if decision_ok and sppvt_ok:
        print("✅ 所有测试通过！UDP通信稳定")
        print("\n可以运行主程序:")
        print("  python acc_updated.py")
        return 0
    else:
        print("❌ 测试失败")
        if not decision_ok:
            print("  - 决策模型通信有问题")
        if not sppvt_ok:
            print("  - SPPVT模型通信有问题")

        print("\n建议检查:")
        print("  1. Simulink模型是否正在运行")
        print("  2. 端口配置是否正确（25000/25001, 26000/26001）")
        print("  3. 模型配置是否正确（运行test_simulink_config.m检查）")
        print("  4. SPPVT模型是否添加了Saturation限制")
        return 1


if __name__ == '__main__':
    sys.exit(main())
