#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版两模式控制测试脚本
避免Windows控制台编码问题
"""

import time
import sys
import os

# 导入项目模块
try:
    from two_mode_controller import TwoModeController, calculate_two_mode_desired_distance, two_mode_control
    from acc_decision import ACCDecisionModule, ACCCommand, ACCState
    print("[OK] 成功导入两模式控制模块")
except ImportError as e:
    print(f"[ERROR] 导入模块失败: {e}")
    exit(1)


def test_mode_switching():
    """测试模式切换逻辑"""
    print("\n" + "="*50)
    print("测试1: 两模式切换逻辑验证")
    print("="*50)
    
    controller = TwoModeController(
        V_threshold_kmh=50.0,
        G2_s=2.0,
        target_speed_kmh=60.0
    )
    
    test_speeds = [20, 35, 45, 49, 50, 51, 55, 70, 80]  # km/h
    correct_count = 0
    
    for speed_kmh in test_speeds:
        speed_ms = speed_kmh / 3.6
        
        # 测试模式判断
        mode = controller.determine_control_mode(speed_ms)
        
        # 测试期望距离计算
        desired_distance, control_mode = controller.calculate_desired_distance(speed_ms)
        
        # 验证逻辑正确性
        expected_mode = 'TIME' if speed_kmh <= 50.0 else 'SPEED'
        is_correct = (mode == expected_mode) and (mode == control_mode)
        
        if is_correct:
            correct_count += 1
        
        status = "[OK]" if is_correct else "[FAIL]"
        print(f"  {status} 速度: {speed_kmh:2.0f}km/h -> 模式: {mode:5s} -> 期望距离: {desired_distance:5.1f}m")
    
    success_rate = correct_count / len(test_speeds) * 100
    print(f"\n模式切换测试结果: {correct_count}/{len(test_speeds)} 正确, 成功率: {success_rate:.1f}%")
    
    return success_rate > 95


def test_acc_decision():
    """测试ACC决策模块"""
    print("\n" + "="*50)
    print("测试2: ACC决策模块测试")
    print("="*50)
    
    acc_decision = ACCDecisionModule(
        initial_target_speed_kmh=50.0,
        initial_time_gap=2.0
    )
    acc_decision.set_debug(False)  # 关闭调试输出以简化显示
    
    # 测试序列
    test_sequence = [
        (ACCCommand.ENGAGE, 40.0, False, "ACC开启"),
        (ACCCommand.INCREASE_SPEED, 45.0, False, "增速操作"),
        (ACCCommand.CRUISE_MODE, 55.0, True, "切换定速巡航"),
        (ACCCommand.DECREASE_SPEED, 60.0, True, "定速模式减速"),
    ]
    
    for i, (command, speed, has_target, description) in enumerate(test_sequence):
        distance = 25.0 if has_target else None
        
        state, mode, msg = acc_decision.process_command(
            command, speed, has_target, distance
        )
        
        print(f"  步骤{i+1}: {description}")
        print(f"    指令: {command.value} -> 状态: {state.value}")
        print(f"    结果: {msg}")
        print()
    
    # 检查最终参数
    final_params = acc_decision.get_current_parameters()
    is_reasonable = (
        20 <= final_params['V_target_kmh'] <= 120 and
        1.0 <= final_params['G2_s'] <= 5.0
    )
    
    print(f"ACC决策测试: {'通过' if is_reasonable else '失败'}")
    print(f"最终参数: V_target={final_params['V_target_kmh']:.1f}km/h, G2={final_params['G2_s']:.1f}s")
    
    return is_reasonable


def test_performance_simulation():
    """简化的性能仿真测试"""
    print("\n" + "="*50)
    print("测试3: 性能仿真测试")
    print("="*50)
    
    simulation_time = 30.0  # 30秒仿真
    dt = 0.1  # 100ms时间步长
    steps = int(simulation_time / dt)
    
    # 初始状态
    ego_speed = 30.0 / 3.6  # 30 km/h
    target_speed = 50.0 / 3.6  # 50 km/h
    distance = 30.0  # 30m
    
    speed_errors = []
    distance_errors = []
    control_modes = []
    
    print(f"仿真参数: 时长={simulation_time}s, 步长={dt}s, 总步数={steps}")
    print("正在运行仿真...")
    
    start_time = time.time()
    
    for step in range(steps):
        t = step * dt
        
        # 模拟前车速度变化
        import math
        target_speed = (50 + 10 * math.sin(0.1 * t)) / 3.6
        
        # 使用两模式控制计算期望距离
        desired_distance, mode = calculate_two_mode_desired_distance(ego_speed)
        
        # 简化的车辆动力学仿真
        distance_error = distance - desired_distance
        speed_error = target_speed - ego_speed
        
        # 简化的控制逻辑
        if mode == 'TIME':
            accel = -0.5 * distance_error + 0.3 * speed_error
        else:
            accel = 0.8 * speed_error - 0.2 * distance_error
        
        # 限制加速度
        accel = max(-3.0, min(2.0, accel))
        
        # 更新状态
        ego_speed += accel * dt
        ego_speed = max(0, ego_speed)
        
        # 更新距离
        relative_speed = target_speed - ego_speed
        distance += relative_speed * dt
        distance = max(5.0, distance)
        
        # 记录数据
        if step % 50 == 0:  # 每5秒记录一次
            speed_errors.append(abs(target_speed - ego_speed) * 3.6)
            distance_errors.append(abs(distance - desired_distance))
            control_modes.append(mode)
    
    simulation_time_elapsed = time.time() - start_time
    
    # 分析结果
    avg_speed_error = sum(speed_errors) / len(speed_errors) if speed_errors else 0
    avg_distance_error = sum(distance_errors) / len(distance_errors) if distance_errors else 0
    
    # 统计模式
    time_mode_count = control_modes.count('TIME')
    speed_mode_count = control_modes.count('SPEED')
    
    print(f"仿真耗时: {simulation_time_elapsed:.3f}秒")
    print(f"速度跟踪: 平均误差 {avg_speed_error:.1f}km/h")
    print(f"距离控制: 平均误差 {avg_distance_error:.1f}m")
    print(f"模式分布: 时距控制 {time_mode_count}次, 定速控制 {speed_mode_count}次")
    
    # 判断性能
    performance_good = (
        avg_speed_error < 5.0 and
        avg_distance_error < 3.0 and
        simulation_time_elapsed < 1.0
    )
    
    print(f"性能测试: {'通过' if performance_good else '失败'}")
    return performance_good


def main():
    """主测试函数"""
    print("两模式控制系统测试工具")
    print("测试时间: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    
    tests = [
        ("模式切换逻辑", test_mode_switching),
        ("ACC决策模块", test_acc_decision),
        ("性能仿真", test_performance_simulation)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n正在执行: {test_name}")
            result = test_func()
            if result:
                passed_tests += 1
                print(f"[OK] {test_name} 测试通过")
            else:
                print(f"[FAIL] {test_name} 测试失败")
        except Exception as e:
            print(f"[ERROR] {test_name} 测试异常: {e}")
    
    # 总结
    print("\n" + "="*50)
    print("测试总结")
    print("="*50)
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n[SUCCESS] 所有测试通过！系统准备就绪")
        print("建议:")
        print("1. 运行 CARLA 仿真测试: python acc_updated.py")
        print("2. 分析实际数据: python analyze_test_data.py")
        return True
    else:
        print(f"\n[WARNING] {total_tests - passed_tests} 个测试失败")
        print("建议检查相关模块后重新测试")
        return False


if __name__ == "__main__":
    main()