#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的Python接口
"""

from acc_decision_sppvt_interface import ACCDecisionSPPVTInterface
import time

def test_python_interface():
    print("测试Python接口...")

    try:
        # 初始化接口
        interface = ACCDecisionSPPVTInterface()

        # 测试数据
        test_input = {
            'ego_speed_kmh': 50.0,
            'ego_speed_ms': 13.89,
            'command_type': 1,  # I0 当速启控
            'command_active': True,
            'manual_throttle_active': False,
            'control_error': 1.5,
            'control_mode_flag': 1,
            'V_target_kmh': 50.0,
            'V_min_kmh': 30.0,
            'G2_s': 2.0,
            'timestamp': time.time()
        }

        print("执行单步决策+SPPVT处理...")
        result = interface.process_decision_and_control(test_input)

        if result:
            # 严格验证测试结果
            validation_errors = []

            # 1. 检查SPPVT输出合理性
            sppvt_output = result['sppvt_control_output']
            if abs(sppvt_output) < 0.001:
                validation_errors.append(f"SPPVT输出异常小: {sppvt_output:.6f} (可能算法未工作)")

            # 2. 检查性能要求 (假设测试过程记录了执行时间)
            if hasattr(interface, 'last_call_time') and interface.last_call_time:
                exec_time_ms = interface.last_call_time * 1000
                if exec_time_ms > 500:  # 500ms性能警告线
                    validation_errors.append(f"性能不达标: {exec_time_ms:.1f}ms > 500ms")
                if exec_time_ms > 2000:  # 2秒严重性能问题
                    validation_errors.append(f"严重性能问题: {exec_time_ms:.1f}ms (实时控制不可接受)")

            # 3. 检查决策逻辑合理性
            command_type = test_input['command_type']  # I0指令
            decision = result['current_decision']
            state = result['current_state']

            if command_type == 1 and decision == 8:  # I0指令但决策是R8(系统待命)
                validation_errors.append(f"决策逻辑异常: I0指令(当速启控)应该激活控制，但决策是R{decision}(系统待命)")

            # 4. 检查状态转移合理性
            if test_input['ego_speed_kmh'] >= test_input['V_min_kmh'] and state == 3:
                validation_errors.append(f"状态转移异常: 车速{test_input['ego_speed_kmh']:.1f}km/h >= 最低速度{test_input['V_min_kmh']:.1f}km/h，不应该在S3(低速状态)")

            # 5. 检查控制使能逻辑
            if command_type == 1 and not result['control_enabled']:  # I0指令应该启用控制
                validation_errors.append(f"控制使能异常: I0指令应该启用控制，但control_enabled=False")

            # 输出测试结果
            if validation_errors:
                print("❌ FAILED: Python接口测试失败！")
                print("   接口通信: ✅ 成功")
                print("   功能验证: ❌ 失败")
                print("\n具体问题:")
                for i, error in enumerate(validation_errors, 1):
                    print(f"   {i}. {error}")

                print("\n详细输出:")
                print(f"   状态: S{result['current_state']} -> S{result['next_state']}")
                print(f"   决策: R{result['current_decision']}")
                print(f"   控制: {result['control_enabled']}")
                print(f"   SPPVT: {result['sppvt_control_output']:.6f}")
                print(f"   目标速度: {result['updated_V_target_kmh']} km/h")
                if hasattr(interface, 'last_call_time') and interface.last_call_time:
                    print(f"   执行耗时: {interface.last_call_time * 1000:.1f}ms")
                return False
            else:
                print("✅ SUCCESS: Python接口测试成功！")
                print(f"   状态: S{result['current_state']} -> S{result['next_state']}")
                print(f"   决策: R{result['current_decision']}")
                print(f"   控制: {result['control_enabled']}")
                print(f"   SPPVT: {result['sppvt_control_output']:.3f}")
                print(f"   目标速度: {result['updated_V_target_kmh']} km/h")
                if hasattr(interface, 'last_call_time') and interface.last_call_time:
                    print(f"   执行耗时: {interface.last_call_time * 1000:.1f}ms")

            # 第二次调用测试状态持续
            print("\n执行第二次调用测试状态持续...")
            test_input['command_active'] = False  # 不激活新命令
            test_input['command_type'] = 0        # 无命令

            result2 = interface.process_decision_and_control(test_input)
            if result2:
                # 验证第二次调用的状态持续性
                continuity_errors = []

                # 检查性能是否有改善
                if hasattr(interface, 'last_call_time') and interface.last_call_time:
                    exec_time_ms = interface.last_call_time * 1000
                    if exec_time_ms > 200:  # 第二次调用应该更快
                        continuity_errors.append(f"第二次调用性能仍然差: {exec_time_ms:.1f}ms > 200ms")

                # 检查状态是否合理持续
                if result2['current_state'] != result['next_state']:
                    continuity_errors.append(f"状态持续异常: 第一次next_state={result['next_state']}, 第二次current_state={result2['current_state']}")

                if continuity_errors:
                    print("\n❌ 第二次调用验证失败:")
                    for error in continuity_errors:
                        print(f"   - {error}")
                    return False
                else:
                    print(f"\n✅ 第二次调用验证通过:")
                    print(f"   状态: S{result2['current_state']} -> S{result2['next_state']}")
                    print(f"   决策: R{result2['current_decision']}")
                    print(f"   控制: {result2['control_enabled']}")
                    print(f"   SPPVT: {result2['sppvt_control_output']:.3f}")
                    if hasattr(interface, 'last_call_time') and interface.last_call_time:
                        print(f"   执行耗时: {interface.last_call_time * 1000:.1f}ms")

            return True
        else:
            print("❌ CRITICAL FAILED: Python接口完全无响应")
            return False

    except Exception as e:
        print(f"ERROR: 测试异常: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_python_interface()
    exit(0 if success else 1)