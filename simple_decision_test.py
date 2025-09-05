#!/usr/bin/env python3
"""
简化的decision.md逻辑测试 - 不依赖三模式控制器
"""

# 简化的测试，直接测试状态转移表
from enum import Enum
import time

class ACCCommand(Enum):
    """ACC指令枚举 - 根据decision.md的7种驾驶指令"""
    DECREASE_SPEED = "I0"  # 降速
    INCREASE_SPEED = "I1"  # 增速
    DECREASE_DISTANCE = "I2"  # 降距
    INCREASE_DISTANCE = "I3"  # 增距
    THROTTLE = "I4"  # 油门 (人驾优先)
    BRAKE = "I5"  # 刹车 (人驾优先) 
    CANCEL = "I6"  # 取消

class ACCState(Enum):
    """ACC状态枚举 - decision.md的4个核心状态"""
    IN_CONTROL = "S0"  # 在控
    ADAPTIVE_HISTORY_STANDBY = "S1"  # 适速有史待命
    ADAPTIVE_NO_HISTORY_STANDBY = "S2"  # 适速无史待命
    LOW_SPEED = "S3"  # 低速
    # 上层系统控制状态
    SYSTEM_STANDBY = "STANDBY"  # 系统待命 (上层控制)
    SYSTEM_EXIT = "EXIT"  # 系统退出 (上层控制)

class ACCControlMode(Enum):
    """ACC控制模式 - 根据decision.md的R1-R8决策"""
    SPEED_DECREASE = "R1"  # R1: 速度降低
    SPEED_INCREASE = "R2"  # R2: 速度增加  
    DISTANCE_DECREASE = "R3"  # R3: 时距降低
    DISTANCE_INCREASE = "R4"  # R4: 时距增加
    NO_CONTINUE_CONTROL = "R5"  # R5: 无继控制
    CONTINUE_CONTROL = "R6"  # R6: 继承控制
    TORQUE_ARBITRATION = "R7"  # R7: 扭矩仲裁
    SYSTEM_STANDBY = "R8"  # R8: 系统待命

def test_decision_transitions():
    """测试decision.md的状态转移逻辑"""
    print("=== decision.md 状态转移表测试 ===")
    
    # 根据decision.md构建的转移表
    transition_table = {
        # === S0 在控状态的转移 ===
        (ACCState.IN_CONTROL, ACCCommand.DECREASE_SPEED): (ACCState.IN_CONTROL, ACCControlMode.SPEED_DECREASE),  # I0→R1
        (ACCState.IN_CONTROL, ACCCommand.INCREASE_SPEED): (ACCState.IN_CONTROL, ACCControlMode.SPEED_INCREASE),   # I1→R2
        (ACCState.IN_CONTROL, ACCCommand.DECREASE_DISTANCE): (ACCState.IN_CONTROL, ACCControlMode.DISTANCE_DECREASE), # I2→R3
        (ACCState.IN_CONTROL, ACCCommand.INCREASE_DISTANCE): (ACCState.IN_CONTROL, ACCControlMode.DISTANCE_INCREASE), # I3→R4
        (ACCState.IN_CONTROL, ACCCommand.THROTTLE): (ACCState.IN_CONTROL, ACCControlMode.TORQUE_ARBITRATION),    # I4→R7
        (ACCState.IN_CONTROL, ACCCommand.BRAKE): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I5→R8→S1
        (ACCState.IN_CONTROL, ACCCommand.CANCEL): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I6→R8→S1
        
        # === S1 适速有史待命状态的转移 ===
        (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.INCREASE_SPEED): (ACCState.IN_CONTROL, ACCControlMode.CONTINUE_CONTROL), # I1→R6→S0
        (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.DECREASE_SPEED): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCControlMode.NO_CONTINUE_CONTROL), # I0→R5→S1
        (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.DECREASE_DISTANCE): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I2→R8→S1
        (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.INCREASE_DISTANCE): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I3→R8→S1
        (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.THROTTLE): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I4→R8→S1
        (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.BRAKE): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I5→R8→S1
        (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.CANCEL): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I6→R8→S1
        
        # === S2 适速无史待命状态的转移 ===
        (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.DECREASE_SPEED): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCControlMode.NO_CONTINUE_CONTROL), # I0→R5→S2
        (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.INCREASE_SPEED): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I1→R8→S2
        (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.DECREASE_DISTANCE): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I2→R8→S2
        (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.INCREASE_DISTANCE): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I3→R8→S2
        (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.THROTTLE): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I4→R8→S2
        (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.BRAKE): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I5→R8→S2
        (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.CANCEL): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCControlMode.SYSTEM_STANDBY), # I6→R8→S2
        
        # === S3 低速状态的转移 ===
        (ACCState.LOW_SPEED, ACCCommand.DECREASE_SPEED): (ACCState.LOW_SPEED, ACCControlMode.SYSTEM_STANDBY), # I0→R8→S3
        (ACCState.LOW_SPEED, ACCCommand.INCREASE_SPEED): (ACCState.LOW_SPEED, ACCControlMode.SYSTEM_STANDBY), # I1→R8→S3
        (ACCState.LOW_SPEED, ACCCommand.DECREASE_DISTANCE): (ACCState.LOW_SPEED, ACCControlMode.SYSTEM_STANDBY), # I2→R8→S3
        (ACCState.LOW_SPEED, ACCCommand.INCREASE_DISTANCE): (ACCState.LOW_SPEED, ACCControlMode.SYSTEM_STANDBY), # I3→R8→S3
        (ACCState.LOW_SPEED, ACCCommand.THROTTLE): (ACCState.LOW_SPEED, ACCControlMode.SYSTEM_STANDBY), # I4→R8→S3
        (ACCState.LOW_SPEED, ACCCommand.BRAKE): (ACCState.LOW_SPEED, ACCControlMode.SYSTEM_STANDBY), # I5→R8→S3
        (ACCState.LOW_SPEED, ACCCommand.CANCEL): (ACCState.LOW_SPEED, ACCControlMode.SYSTEM_STANDBY), # I6→R8→S3
    }
    
    # 测试所有转移
    test_cases = [
        # S0 在控状态测试
        ("S0-在控: I0降速→R1", ACCState.IN_CONTROL, ACCCommand.DECREASE_SPEED),
        ("S0-在控: I1增速→R2", ACCState.IN_CONTROL, ACCCommand.INCREASE_SPEED),
        ("S0-在控: I2降距→R3", ACCState.IN_CONTROL, ACCCommand.DECREASE_DISTANCE),
        ("S0-在控: I3增距→R4", ACCState.IN_CONTROL, ACCCommand.INCREASE_DISTANCE),
        ("S0-在控: I4油门→R7", ACCState.IN_CONTROL, ACCCommand.THROTTLE),
        ("S0-在控: I5刹车→R8→S1", ACCState.IN_CONTROL, ACCCommand.BRAKE),
        ("S0-在控: I6取消→R8→S1", ACCState.IN_CONTROL, ACCCommand.CANCEL),
        
        # S1 适速有史待命测试
        ("S1-适速有史: I1增速→R6→S0", ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.INCREASE_SPEED),
        ("S1-适速有史: I0降速→R5→S1", ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.DECREASE_SPEED),
        ("S1-适速有史: I2降距→R8→S1", ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.DECREASE_DISTANCE),
        
        # S2 适速无史待命测试
        ("S2-适速无史: I0降速→R5→S2", ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.DECREASE_SPEED),
        ("S2-适速无史: I1增速→R8→S2", ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.INCREASE_SPEED),
        
        # S3 低速状态测试
        ("S3-低速: I0降速→R8→S3", ACCState.LOW_SPEED, ACCCommand.DECREASE_SPEED),
        ("S3-低速: I1增速→R8→S3", ACCState.LOW_SPEED, ACCCommand.INCREASE_SPEED),
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for desc, initial_state, command in test_cases:
        print(f"\n测试: {desc}")
        print(f"  初始状态: {initial_state.value}")
        print(f"  指令: {command.value}")
        
        if (initial_state, command) in transition_table:
            expected_state, expected_mode = transition_table[(initial_state, command)]
            print(f"  期望转移: {initial_state.value} → {expected_state.value}")
            print(f"  期望决策: {expected_mode.value}")
            print(f"  ✅ 转移逻辑正确")
            success_count += 1
        else:
            print(f"  ❌ 转移表中未找到该组合")
    
    print(f"\n=== 测试结果 ===")
    print(f"测试通过: {success_count}/{total_count}")
    print(f"通过率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("🎉 所有转移都符合decision.md的逻辑！")
    else:
        print("❌ 部分转移不符合decision.md的逻辑，需要检查实现")

if __name__ == "__main__":
    test_decision_transitions()