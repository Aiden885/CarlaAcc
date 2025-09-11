#!/usr/bin/env python3
"""
扭矩仲裁功能测试脚本
测试ACC决策模块的扭矩仲裁逻辑
"""

from acc_decision import ACCDecisionModule, ACCCommand, ACCState
import time

def test_torque_arbitration():
    """测试扭矩仲裁功能"""
    print("=== 扭矩仲裁功能测试 ===")
    
    # 创建决策模块
    acc_decision = ACCDecisionModule(initial_target_speed_kmh=50.0, initial_time_gap=2.0)
    acc_decision.set_debug(True)
    
    ego_speed = 45.0  # km/h
    
    print(f"\n初始状态: {acc_decision.current_state.value}")
    print(f"扭矩仲裁状态: {acc_decision.torque_arbitration_active}")
    
    # 步骤1: 先激活ACC控制（从适速无史待命 -> 在控）
    print("\n=== 步骤1: 激活ACC控制 ===")
    state, decision, msg = acc_decision.process_command(ACCCommand.DECREASE_SPEED, ego_speed)
    print(f"激活ACC结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")
    print(f"扭矩仲裁状态: {acc_decision.torque_arbitration_active}")
    
    # 步骤2: 在控制状态下按油门键触发扭矩仲裁
    print("\n=== 步骤2: 在控制状态下按油门键 ===")
    state, decision, msg = acc_decision.process_command(ACCCommand.THROTTLE, ego_speed)
    print(f"油门指令结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")  # 应该仍为S0在控
    print(f"扭矩仲裁状态: {acc_decision.torque_arbitration_active}")  # 应该为True
    
    # 步骤3: 测试决策输出
    print("\n=== 步骤3: 测试决策输出 ===")
    # 模拟持续按油门
    decision_output = acc_decision.get_decision_output(ego_speed, None, manual_throttle_active=True)
    print(f"持续按油门时的输出:")
    print(f"  control_enabled: {decision_output['control_enabled']}")
    print(f"  torque_arbitration_active: {decision_output['torque_arbitration_active']}")
    print(f"  current_state: {decision_output['state']}")
    
    # 模拟松开油门
    decision_output = acc_decision.get_decision_output(ego_speed, None, manual_throttle_active=False)
    print(f"\n松开油门后的输出:")
    print(f"  control_enabled: {decision_output['control_enabled']}")
    print(f"  torque_arbitration_active: {decision_output['torque_arbitration_active']}")
    print(f"  current_state: {decision_output['state']}")
    
    # 步骤4: 测试其他状态下的油门指令
    print("\n=== 步骤4: 测试其他状态下的油门指令 ===")
    # 先取消ACC进入待命
    acc_decision.process_command(ACCCommand.CANCEL, ego_speed)
    print(f"取消ACC后状态: {acc_decision.current_state.value}")
    
    # 在待命状态下按油门
    state, decision, msg = acc_decision.process_command(ACCCommand.THROTTLE, ego_speed)
    print(f"待命状态下油门指令结果: {msg}")
    print(f"状态: {acc_decision.current_state.value}")
    print(f"扭矩仲裁状态: {acc_decision.torque_arbitration_active}")  # 应该为False
    
    print("\n=== 扭矩仲裁测试完成 ===")

if __name__ == "__main__":
    test_torque_arbitration()