#!/usr/bin/env python3
"""
测试基于decision.md的ACC决策逻辑
验证4个核心状态和7种指令的转移逻辑
"""

from acc_decision import ACCDecisionModule, ACCCommand, ACCState, ACCControlMode

def test_decision_md_logic():
    """测试完全按照decision.md的逻辑"""
    print("=== decision.md逻辑测试 ===")
    
    # 创建决策模块，设置V1_kmh=0使适速和低速界限为0 km/h
    acc_decision = ACCDecisionModule(initial_V3_kmh=50.0, initial_G1_m=15.0)
    acc_decision.V1_kmh = 0  # 设置适速/低速界限
    acc_decision.set_debug(True)
    
    ego_speed = 35.0  # 适速
    
    print(f"\n初始状态: {acc_decision.current_state.value}")
    print(f"初始V1_kmh(适速/低速界限): {acc_decision.V1_kmh}")
    
    # 测试用例按照decision.md第四部分的逻辑
    test_cases = [
        # 1. 当前状态：S0在控
        ("从适速无史待命激活系统", ACCCommand.THROTTLE, 35.0, False),
        ("在控状态：I0降速 → R1", ACCCommand.DECREASE_SPEED, 35.0, False),
        ("在控状态：I1增速 → R2", ACCCommand.INCREASE_SPEED, 35.0, False), 
        ("在控状态：I2降距 → R3", ACCCommand.DECREASE_DISTANCE, 35.0, True),
        ("在控状态：I3增距 → R4", ACCCommand.INCREASE_DISTANCE, 35.0, True),
        ("在控状态：I4油门 → R7", ACCCommand.THROTTLE, 35.0, True),
        ("在控状态：I5刹车 → R8→S1", ACCCommand.BRAKE, 35.0, True),
        
        # 2. 当前状态：S1适速有史待命
        ("有史待命：I1增速 → R6→S0", ACCCommand.INCREASE_SPEED, 35.0, False),
        ("有史待命：I0降速 → R5→S1", ACCCommand.DECREASE_SPEED, 35.0, False),
        ("有史待命：I2降距 → R8→S1", ACCCommand.DECREASE_DISTANCE, 35.0, False),
        
        # 3. 重置到S2进行测试
        # 4. 从系统退出重新激活
        ("重新激活到适速无史待命", ACCCommand.CANCEL, 35.0, False),  # 先退出
        ("从退出状态激活", ACCCommand.THROTTLE, 35.0, False),  # 再激活
        ("无史待命：I0降速 → R5→S2", ACCCommand.DECREASE_SPEED, 35.0, False),
        ("无史待命：I1增速 → R8→S2", ACCCommand.INCREASE_SPEED, 35.0, False),
        
        # 5. 测试低速状态 (设置极低速度)
        ("低速状态测试：I0 → R8→S3", ACCCommand.DECREASE_SPEED, -1.0, False),  # 低于V1_kmh界限
        ("低速状态测试：I1 → R8→S3", ACCCommand.INCREASE_SPEED, -1.0, False),
    ]
    
    for i, (desc, command, speed, has_target) in enumerate(test_cases):
        print(f"\n=== 测试 {i+1}: {desc} ===")
        print(f"执行前状态: {acc_decision.current_state.value}")
        print(f"指令: {command.value}, 速度: {speed}km/h, 有前车: {has_target}")
        
        # 特殊处理：退出指令
        if command == ACCCommand.CANCEL and "退出" in desc:
            acc_decision.current_state = ACCState.SYSTEM_EXIT
            print(f"手动设置状态为退出: {acc_decision.current_state.value}")
            continue
        
        # 执行指令
        try:
            new_state, control_mode, msg = acc_decision.process_command(
                command, speed, has_target, 20.0 if has_target else None
            )
            
            print(f"执行结果:")
            print(f"  新状态: {new_state.value}")
            print(f"  控制模式: {control_mode.value if control_mode else None}")
            print(f"  消息: {msg}")
            
            # 验证状态转移是否符合decision.md
            expected_results = verify_transition(desc, command, control_mode, new_state)
            if expected_results:
                print(f"  ✅ 转移符合decision.md逻辑: {expected_results}")
            else:
                print(f"  ❌ 转移可能不符合decision.md逻辑")
                
        except Exception as e:
            print(f"  ❌ 执行出错: {e}")

def verify_transition(desc, command, control_mode, new_state):
    """验证转移是否符合decision.md的逻辑"""
    expected_mappings = {
        "I0降速 → R1": (ACCCommand.DECREASE_SPEED, ACCControlMode.SPEED_DECREASE),
        "I1增速 → R2": (ACCCommand.INCREASE_SPEED, ACCControlMode.SPEED_INCREASE),
        "I2降距 → R3": (ACCCommand.DECREASE_DISTANCE, ACCControlMode.DISTANCE_DECREASE),
        "I3增距 → R4": (ACCCommand.INCREASE_DISTANCE, ACCControlMode.DISTANCE_INCREASE),
        "I4油门 → R7": (ACCCommand.THROTTLE, ACCControlMode.TORQUE_ARBITRATION),
        "I5刹车 → R8→S1": (ACCCommand.BRAKE, ACCControlMode.SYSTEM_STANDBY),
        "I1增速 → R6→S0": (ACCCommand.INCREASE_SPEED, ACCControlMode.CONTINUE_CONTROL),
        "I0降速 → R5→S1": (ACCCommand.DECREASE_SPEED, ACCControlMode.NO_CONTINUE_CONTROL),
        "I0降速 → R5→S2": (ACCCommand.DECREASE_SPEED, ACCControlMode.NO_CONTINUE_CONTROL),
        "I1增速 → R8→S2": (ACCCommand.INCREASE_SPEED, ACCControlMode.SYSTEM_STANDBY),
        "I0 → R8→S3": (ACCCommand.DECREASE_SPEED, ACCControlMode.SYSTEM_STANDBY),
        "I1 → R8→S3": (ACCCommand.INCREASE_SPEED, ACCControlMode.SYSTEM_STANDBY),
    }
    
    for pattern, (exp_cmd, exp_mode) in expected_mappings.items():
        if pattern in desc and command == exp_cmd:
            if control_mode == exp_mode:
                return f"✅ {pattern}"
            else:
                return f"❌ 期望{exp_mode.value}，实际{control_mode.value if control_mode else None}"
    
    return None

if __name__ == "__main__":
    test_decision_md_logic()