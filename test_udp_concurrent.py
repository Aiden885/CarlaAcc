"""
并发UDP测试 - 同时测试两个Simulink模型
解决阻塞模式下顺序测试的冲突问题
"""
import time
import sys
import threading
from simulink_udp_interface import SimulinkUDPClient

def test_decision_concurrent():
    """并发测试决策模型"""
    print("\n[决策模型] 开始测试...")

    try:
        client = SimulinkUDPClient(
            send_port=25000,
            recv_port=25001,
            num_inputs=4,
            num_outputs=5,
            timeout=10.0,
            debug=False,
            local_send_port=9090
        )

        success = 0
        fail = 0

        # 简化测试：只测试关键转换
        current_state = 2.0
        has_history = 0.0
        last_decision = 8.0

        for i in range(20):
            # 每5帧改变一次命令
            if i == 0:
                cmd = 0  # 待命
            elif i == 5:
                cmd = 1  # E键启动
            elif i == 15:
                cmd = 7  # C键退出
            else:
                cmd = 0  # 保持

            inputs = [current_state, float(cmd), has_history, last_decision]

            try:
                client.call(inputs)  # 预热
                outputs = client.call(inputs)
                success += 1

                current_state = outputs[0]
                has_history = outputs[3]
                last_decision = outputs[4]

                if i % 5 == 0:
                    print(f"[决策] 帧#{i}: state=S{int(outputs[0])}, decision=R{int(outputs[1])}")

            except Exception as e:
                fail += 1
                print(f"[决策] 帧#{i}: 失败 - {e}")
                break

            time.sleep(0.05)

        client.cleanup()
        print(f"[决策模型] 完成: {success}/20 成功")
        return fail == 0

    except Exception as e:
        print(f"[决策模型] 初始化失败: {e}")
        return False

def test_sppvt_concurrent():
    """并发测试SPPVT模型"""
    print("\n[SPPVT模型] 开始测试...")

    try:
        client = SimulinkUDPClient(
            send_port=26000,
            recv_port=26001,
            num_inputs=5,
            num_outputs=5,
            timeout=10.0,
            debug=False,
            local_send_port=9091
        )

        success = 0
        fail = 0

        # SPPVT状态
        stage_offset = 0.0
        prev_error = 0.0
        prev_vel = 0.0
        prev_accel = 0.0

        for i in range(20):
            # 有意义的测试数据：模拟不同的误差
            error = 5.0 - i * 0.2  # 误差从5.0逐渐减小到1.0
            inputs = [error, stage_offset, prev_error, prev_vel, prev_accel]

            try:
                # ⚠️ 双调用解决延迟问题
                client.call(inputs)  # 预热
                outputs = client.call(inputs)  # 实际结果
                success += 1

                # 更新状态
                prev_error = error
                prev_vel = outputs[1]
                prev_accel = outputs[2]

                if i % 1 == 0:
                    print(f"[SPPVT] 帧#{i}: error={error:.1f}s → control={outputs[0]:.3f}, vel={outputs[1]:.3f}, accel={outputs[2]:.3f}")

            except Exception as e:
                fail += 1
                print(f"[SPPVT] 帧#{i}: 失败 - {e}")
                break

            time.sleep(0.05)

        client.cleanup()
        print(f"[SPPVT模型] 完成: {success}/20 成功")
        return fail == 0

    except Exception as e:
        print(f"[SPPVT模型] 初始化失败: {e}")
        return False

def main():
    print("="*70)
    print("并发UDP测试 - 同时测试两个Simulink模型")
    print("="*70)
    print("\n⚠️  请确保:")
    print("  1. 两个Simulink模型都在运行（阻塞模式）")
    print("  2. UDP端口配置正确")

    input("\n按回车开始并发测试...")

    # 创建两个线程
    decision_result = [False]
    sppvt_result = [False]

    def run_decision():
        decision_result[0] = test_decision_concurrent()

    def run_sppvt():
        sppvt_result[0] = test_sppvt_concurrent()

    t1 = threading.Thread(target=run_decision)
    t2 = threading.Thread(target=run_sppvt)

    # 同时启动两个线程
    print("\n🚀 启动并发测试...")
    t1.start()
    t2.start()

    # 等待两个线程完成
    t1.join()
    t2.join()

    print("\n" + "="*70)
    print("测试结果")
    print("="*70)

    if decision_result[0] and sppvt_result[0]:
        print("✅ 两个模型都测试通过！")
        print("\n可以运行主程序:")
        print("  python acc_updated.py")
        return 0
    else:
        print("❌ 测试失败")
        if not decision_result[0]:
            print("  - 决策模型失败")
        if not sppvt_result[0]:
            print("  - SPPVT模型失败")
        return 1

if __name__ == '__main__':
    sys.exit(main())
