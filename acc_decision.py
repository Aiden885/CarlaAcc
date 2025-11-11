#!/usr/bin/env python3
"""
ACC自适应巡航控制决策模块 - 两模式版本
实现ACC系统的指令处理、状态管理和状态转移逻辑
支持基于两模式控制（时距控制+定速控制）的增速/减速和增距/减距逻辑
支持一键定速巡航功能
添加了当前控制模式追踪
"""

from enum import Enum
import time
from two_mode_controller import set_two_mode_parameters, get_two_mode_status


class ACCCommand(Enum):
    """ACC指令枚举 - 基于新决策文档"""
    DECREASE_SPEED = "I0"  # 降速/开启ACC(当速启控)
    INCREASE_SPEED = "I1"  # 增速/开启ACC(继承启控)
    DECREASE_DISTANCE = "I2"  # 降距
    INCREASE_DISTANCE = "I3"  # 增距
    THROTTLE = "I4"  # 油门(人驾优先)
    BRAKE = "I5"  # 刹车(人驾优先)
    CANCEL = "I6"  # 取消ACC


class ACCState(Enum):
    """ACC状态枚举 - 基于新决策文档"""
    IN_CONTROL = "S0"  # 在控：车速为适速，ACC系统正在主动控制车辆
    ADAPTIVE_HISTORY_STANDBY = "S1"  # 适速有史待命：车速为适速，ACC处于待命状态，且存有上次设定的历史速度数据
    ADAPTIVE_NO_HISTORY_STANDBY = "S2"  # 适速无史待命：车速为适速，ACC处于待命状态，但没有历史速度数据
    LOW_SPEED = "S3"  # 低速：车辆处于低速，ACC处于待命状态


class ACCDecision(Enum):
    """ACC决策枚举 - 基于新决策文档"""
    SPEED_DECREASE = "R1"  # 速度降低：速度减，控制继续，反馈目标车速
    SPEED_INCREASE = "R2"  # 速度增加：速度增，控制继续，反馈目标车速
    TIME_GAP_DECREASE = "R3"  # 时距降低：目标距离减，控制继续，反馈目标距离
    TIME_GAP_INCREASE = "R4"  # 时距增加：目标距离增，控制继续，反馈目标距离
    NO_HISTORY_CONTROL = "R5"  # 无继控制：当前车速，进入控制，反馈目标车速/距离
    CONTINUE_CONTROL = "R6"  # 继承控制：上次目标，进入控制，反馈目标车速/距离
    TORQUE_ARBITRATION = "R7"  # 扭矩仲裁：无关，控制继续，反馈目标车速/距离
    SYSTEM_STANDBY = "R8"  # 系统待命：无关，进入待命，反馈ACC待命


class ACCDecisionModule:
    """
    ACC决策模块 - 基于新决策文档
    负责处理ACC指令、管理状态转移和输出标准化决策
    实现4状态(S0-S3) × 7指令(I0-I6) → 8决策(R1-R8)的核心逻辑
    """

    def __init__(self, initial_min_speed_kmh=30.0, initial_target_speed_kmh=50.0, initial_time_gap=2.0):
        """
        初始化ACC决策模块

        Args:
            initial_min_speed_kmh: 最低速度要求 (km/h) - 低速状态判断阈值
            initial_target_speed_kmh: 目标速度 (km/h) - 两模式控制的切换速度和目标速度
            initial_time_gap: 初始时间间隔 (s) - 时距参数G2
        """
        # 当前系统状态（初始状态为适速无史待命）
        self.current_state = ACCState.ADAPTIVE_NO_HISTORY_STANDBY
        self.previous_state = None

        # 当前决策
        self.current_decision = None
        self.previous_decision = None

        # === 两模式控制参数 ===
        self.V_min_kmh = initial_min_speed_kmh  # 最低速度要求（低速判断阈值）
        self.V_target_kmh = initial_target_speed_kmh  # 目标速度
        self.G2_s = initial_time_gap  # 时距参数

        # 调整步长
        self.speed_step = 1.0  # 速度调整步长：1 km/h per command
        self.distance_step = 2.0  # 距离调整步长：2 m per command

        # 历史状态管理
        self.has_history = False
        self.history_V_target_kmh = None
        self.history_G2_s = None
        self.last_control_time = None

        # 扭矩仲裁标志
        self.torque_arbitration_active = False

        # 状态转移表
        self._initialize_transition_table()

        # 调试模式
        self.debug = False

        # 初始化两模式参数
        self._update_two_mode_parameters()

        print("ACC决策模块初始化完成（基于新决策文档）")
        print(f"初始参数: V_min={self.V_min_kmh}km/h, V_target={self.V_target_kmh}km/h, G2={self.G2_s}s")

    def _initialize_transition_table(self):
        """初始化状态转移表 - 基于新决策文档"""
        self.transition_table = {
            # === S0 在控状态的转移 ===
            (ACCState.IN_CONTROL, ACCCommand.DECREASE_SPEED): (ACCState.IN_CONTROL, ACCDecision.SPEED_DECREASE),
            (ACCState.IN_CONTROL, ACCCommand.INCREASE_SPEED): (ACCState.IN_CONTROL, ACCDecision.SPEED_INCREASE),
            (ACCState.IN_CONTROL, ACCCommand.DECREASE_DISTANCE): (ACCState.IN_CONTROL, ACCDecision.TIME_GAP_DECREASE),
            (ACCState.IN_CONTROL, ACCCommand.INCREASE_DISTANCE): (ACCState.IN_CONTROL, ACCDecision.TIME_GAP_INCREASE),
            (ACCState.IN_CONTROL, ACCCommand.THROTTLE): (ACCState.IN_CONTROL, ACCDecision.TORQUE_ARBITRATION),
            (ACCState.IN_CONTROL, ACCCommand.BRAKE): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),
            (ACCState.IN_CONTROL, ACCCommand.CANCEL): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),

            # === S1 适速有史待命状态的转移 ===
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.INCREASE_SPEED): (ACCState.IN_CONTROL, ACCDecision.CONTINUE_CONTROL),
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.DECREASE_SPEED): (ACCState.IN_CONTROL, ACCDecision.NO_HISTORY_CONTROL),
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.DECREASE_DISTANCE): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.INCREASE_DISTANCE): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.THROTTLE): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.BRAKE): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.CANCEL): (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),

            # === S2 适速无史待命状态的转移 ===
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.DECREASE_SPEED): (ACCState.IN_CONTROL, ACCDecision.NO_HISTORY_CONTROL),
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.INCREASE_SPEED): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.DECREASE_DISTANCE): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.INCREASE_DISTANCE): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.THROTTLE): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.BRAKE): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.CANCEL): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCDecision.SYSTEM_STANDBY),

            # === S3 低速状态的转移 ===
            (ACCState.LOW_SPEED, ACCCommand.DECREASE_SPEED): (ACCState.LOW_SPEED, ACCDecision.SYSTEM_STANDBY),
            (ACCState.LOW_SPEED, ACCCommand.INCREASE_SPEED): (ACCState.LOW_SPEED, ACCDecision.SYSTEM_STANDBY),
            (ACCState.LOW_SPEED, ACCCommand.DECREASE_DISTANCE): (ACCState.LOW_SPEED, ACCDecision.SYSTEM_STANDBY),
            (ACCState.LOW_SPEED, ACCCommand.INCREASE_DISTANCE): (ACCState.LOW_SPEED, ACCDecision.SYSTEM_STANDBY),
            (ACCState.LOW_SPEED, ACCCommand.THROTTLE): (ACCState.LOW_SPEED, ACCDecision.SYSTEM_STANDBY),
            (ACCState.LOW_SPEED, ACCCommand.BRAKE): (ACCState.LOW_SPEED, ACCDecision.SYSTEM_STANDBY),
            (ACCState.LOW_SPEED, ACCCommand.CANCEL): (ACCState.LOW_SPEED, ACCDecision.SYSTEM_STANDBY),
        }

    def _update_two_mode_parameters(self):
        """更新两模式控制器的参数"""
        set_two_mode_parameters(
            V_threshold_kmh=self.V_target_kmh,  # 使用目标速度作为模式切换阈值
            G2_s=self.G2_s,
            target_speed_kmh=self.V_target_kmh  # 目标速度
        )

        if self.debug:
            print(f"两模式参数更新: V_target={self.V_target_kmh}km/h, G2={self.G2_s}s")

    def determine_state_from_speed(self, ego_speed_kmh):
        """
        根据车速确定应该处于的状态类型

        Args:
            ego_speed_kmh: 当前车速 (km/h)

        Returns:
            适合的状态类型
        """
        if ego_speed_kmh < self.V_min_kmh:
            return ACCState.LOW_SPEED
        elif self.has_history:
            return ACCState.ADAPTIVE_HISTORY_STANDBY
        else:
            return ACCState.ADAPTIVE_NO_HISTORY_STANDBY

    def process_command(self, command, ego_speed_kmh, has_target=False, current_distance=None):
        """
        处理ACC指令并执行状态转移 - 基于新决策文档

        Args:
            command: ACC指令
            ego_speed_kmh: 当前车速 (km/h)
            has_target: 是否存在前车目标
            current_distance: 当前与前车距离 (m, 可选)

        Returns:
            tuple: (新状态, 决策, 执行结果消息)
        """
        if not isinstance(command, ACCCommand):
            return self.current_state, None, "无效指令"

        if self.debug:
            print(f"处理指令: {command.value}, 当前状态: {self.current_state.value}")

        # 保存当前状态和决策作为历史
        self.previous_state = self.current_state
        self.previous_decision = self.current_decision

        # 查找状态转移
        transition_key = (self.current_state, command)

        if transition_key in self.transition_table:
            new_state, decision = self.transition_table[transition_key]

            # 保存原状态用于调试输出
            old_state_for_debug = self.current_state

            # 更新当前决策
            self.current_decision = decision

            # 执行状态转移和决策
            success_msg = self._execute_state_transition_and_decision(new_state, decision, command, ego_speed_kmh, has_target, current_distance)

            if self.debug:
                print(f"状态转移: {old_state_for_debug.value} + {command.value} -> {new_state.value}")
                print(f"执行决策: {decision.value}")

            return new_state, decision, success_msg
        else:
            # 无效的状态转移
            return self.current_state, None, f"当前状态 {self.current_state.value} 下不支持指令 {command.value}"

    def _execute_state_transition_and_decision(self, new_state, decision, command, ego_speed_kmh, has_target=False,
                                               current_distance=None):
        """执行具体的状态转移和决策逻辑 - 基于新决策文档"""
        old_state = self.current_state
        self.current_state = new_state

        if self.debug:
            print(f"执行状态转移: {old_state.value} -> {new_state.value}")
            print(f"执行决策: {decision.value}")

        # 根据决策类型执行相应的动作
        if decision == ACCDecision.SPEED_DECREASE:
            # R1: 速度降低 - 速度减，控制继续
            self.V_target_kmh = max(self.V_min_kmh + 1, self.V_target_kmh - self.speed_step)
            self._update_two_mode_parameters()
            return f"R1-速度降低: 目标速度调至{self.V_target_kmh:.1f}km/h"
            
        elif decision == ACCDecision.SPEED_INCREASE:
            # R2: 速度增加 - 速度增，控制继续
            self.V_target_kmh = min(150.0, self.V_target_kmh + self.speed_step)  # 修改最大速度限制为150 km/h
            self._update_two_mode_parameters()
            return f"R2-速度增加: 目标速度调至{self.V_target_kmh:.1f}km/h"
            
        elif decision == ACCDecision.TIME_GAP_DECREASE:
            # R3: 时距降低 - 目标距离减，控制继续
            base_speed_ms = 50.0 / 3.6
            time_adjustment = -self.distance_step / base_speed_ms
            self.G2_s = max(1.0, self.G2_s + time_adjustment)
            self._update_two_mode_parameters()
            return f"R3-时距降低: G2调整{time_adjustment:.2f}s，新G2={self.G2_s:.2f}s"
            
        elif decision == ACCDecision.TIME_GAP_INCREASE:
            # R4: 时距增加 - 目标距离增，控制继续
            base_speed_ms = 50.0 / 3.6
            time_adjustment = self.distance_step / base_speed_ms
            self.G2_s = max(1.0, self.G2_s + time_adjustment)
            self._update_two_mode_parameters()
            return f"R4-时距增加: G2调整{time_adjustment:.2f}s，新G2={self.G2_s:.2f}s"
            
        elif decision == ACCDecision.NO_HISTORY_CONTROL:
            # R5: 无继控制 - 当前车速，进入控制
            self.V_target_kmh = ego_speed_kmh  # 使用当前车速作为目标速度
            self._update_two_mode_parameters()
            self.last_control_time = time.time()
            return f"R5-无继控制: 以当前车速{self.V_target_kmh:.1f}km/h开始控制"
            
        elif decision == ACCDecision.CONTINUE_CONTROL:
            # R6: 继承控制 - 上次目标，进入控制
            if self.history_V_target_kmh:
                self.V_target_kmh = self.history_V_target_kmh
            if self.history_G2_s:
                self.G2_s = self.history_G2_s
            self._update_two_mode_parameters()
            self.last_control_time = time.time()
            return f"R6-继承控制: 恢复目标速度={self.V_target_kmh:.1f}km/h, G2={self.G2_s:.1f}s"
            
        elif decision == ACCDecision.TORQUE_ARBITRATION:
            # R7: 扭矩仲裁 - 驾驶员与ACC取最大油门开度
            self.torque_arbitration_active = True
            return f"R7-扭矩仲裁: 驾驶员与ACC协调控制（取最大油门开度）"
            
        elif decision == ACCDecision.SYSTEM_STANDBY:
            # R8: 系统待命 - 进入待命
            if old_state == ACCState.IN_CONTROL:
                # 从在控状态退出，保存历史
                self._save_history()
            self.torque_arbitration_active = False
            return f"R8-系统待命: ACC进入待命状态，当前状态={new_state.value}"
            
        return f"状态转移: {old_state.value} -> {new_state.value}"

    def _save_history(self):
        """保存当前设定为历史"""
        self.has_history = True
        self.history_V_target_kmh = self.V_target_kmh
        self.history_G2_s = self.G2_s

        if self.debug:
            print(
                f"保存历史: V_target={self.history_V_target_kmh:.1f}km/h, G2={self.history_G2_s:.1f}s")

    def _clear_history(self):
        """清除历史设定"""
        self.has_history = False
        self.history_V_target_kmh = None
        self.history_G2_s = None
        self.last_control_time = None
        self.torque_arbitration_active = False
        self.current_decision = None

        if self.debug:
            print("清除历史设定和决策状态")

    def update_state_by_speed(self, ego_speed_kmh):
        """
        根据车速自动更新状态（仅在非控制状态下） - 基于新决策文档

        Args:
            ego_speed_kmh: 当前车速 (km/h)
        """
        # 只有在非控制状态下才自动切换
        if self.current_state != ACCState.IN_CONTROL:
            ideal_state = self.determine_state_from_speed(ego_speed_kmh)

            # 只有当状态确实需要改变时才更新
            if self.current_state != ideal_state:
                self.previous_state = self.current_state
                self.current_state = ideal_state

                if self.debug:
                    print(f"根据车速自动更新状态: {ego_speed_kmh:.1f}km/h -> {ideal_state.value}")

    def get_current_parameters(self):
        """
        获取当前ACC参数 - 基于新决策文档

        Returns:
            dict: 当前参数字典
        """
        return {
            'V_target_kmh': self.V_target_kmh,
            'V_min_kmh': self.V_min_kmh,
            'G2_s': self.G2_s,
            'state': self.current_state.value,
            'has_history': self.has_history,
            'is_active': self.current_state == ACCState.IN_CONTROL,
            'torque_arbitration_active': self.torque_arbitration_active,
            'current_decision': self.current_decision.value if self.current_decision else None
        }

    def get_decision_output(self, ego_speed_kmh, current_distance=None, manual_throttle_active=False):
        """
        获取决策输出，供控制模块使用 - 基于新决策文档

        Args:
            ego_speed_kmh: 当前车速 (km/h)
            current_distance: 当前目标距离 (m, 可选)
            manual_throttle_active: 是否正在手动按油门 (bool, 用于扭矩仲裁状态管理)

        Returns:
            dict: 决策输出
        """
        # 根据速度自动更新状态
        self.update_state_by_speed(ego_speed_kmh)

        # === 扭矩仲裁状态管理 ===
        # 如果当前在扭矩仲裁状态，但没有持续按油门，则重置仲裁状态
        if self.torque_arbitration_active and not manual_throttle_active:
            self.torque_arbitration_active = False
            if self.debug:
                print("Torque arbitration auto reset: no continuous throttle input")

        # 检查是否有前车
        has_target = current_distance is not None and current_distance < 100.0

        decision_output = {
            'acc_active': self.current_state == ACCState.IN_CONTROL,
            'V_target_kmh': self.V_target_kmh,
            'V_target_ms': self.V_target_kmh / 3.6,
            'V_min_kmh': self.V_min_kmh,
            'G2_s': self.G2_s,
            'state': self.current_state.value,
            'state_description': self._get_state_description(),
            'control_enabled': self.current_state == ACCState.IN_CONTROL,
            'has_target': has_target,
            'torque_arbitration_active': self.torque_arbitration_active,
            'current_decision': self.current_decision.value if self.current_decision else None
        }

        return decision_output

    def _get_state_description(self):
        """获取状态描述 - 基于新决策文档"""
        descriptions = {
            ACCState.IN_CONTROL: "在控：车速为适速，ACC系统正在主动控制车辆",
            ACCState.ADAPTIVE_HISTORY_STANDBY: "适速有史待命：车速为适速，ACC处于待命状态，且存有上次设定的历史速度数据",
            ACCState.ADAPTIVE_NO_HISTORY_STANDBY: "适速无史待命：车速为适速，ACC处于待命状态，但没有历史速度数据",
            ACCState.LOW_SPEED: "低速：车辆处于低速，ACC处于待命状态"
        }
        return descriptions.get(self.current_state, "未知状态")

    def is_in_active_control_mode(self):
        """
        判断是否处于主动控制模式 - 基于新决策文档

        Returns:
            bool: 是否处于需要执行控制的模式
        """
        return self.current_state == ACCState.IN_CONTROL

    def set_debug(self, enable):
        """启用/禁用调试模式"""
        self.debug = enable

    def reset(self):
        """重置ACC决策模块 - 基于新决策文档"""
        self.current_state = ACCState.ADAPTIVE_NO_HISTORY_STANDBY
        self.previous_state = None
        self.current_decision = None
        self.previous_decision = None
        self._clear_history()
        print("ACC决策模块已重置")

    def get_status_info(self):
        """获取详细状态信息 - 基于新决策文档"""
        return {
            'current_state': self.current_state.value,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'current_decision': self.current_decision.value if self.current_decision else None,
            'previous_decision': self.previous_decision.value if self.previous_decision else None,
            'V_target_kmh': self.V_target_kmh,
            'V_min_kmh': self.V_min_kmh,
            'G2_s': self.G2_s,
            'has_history': self.has_history,
            'history_V_target_kmh': self.history_V_target_kmh,
            'history_G2_s': self.history_G2_s,
            'last_control_time': self.last_control_time,
            'state_description': self._get_state_description(),
            'torque_arbitration_active': self.torque_arbitration_active,
            'is_in_active_control_mode': self.is_in_active_control_mode()
        }


def test_acc_decision_logic():
    """测试ACC决策模块的逻辑 - 基于新决策文档"""
    print("=== ACC决策模块逻辑测试（基于新决策文档） ===")

    # 创建决策模块
    acc_decision = ACCDecisionModule(initial_min_speed_kmh=30.0, initial_target_speed_kmh=50.0, initial_time_gap=2.0)
    acc_decision.set_debug(True)

    ego_speed = 35.0  # km/h

    print(f"\n初始状态: {acc_decision.current_state.value}")
    print(f"初始决策: {acc_decision.current_decision}")
    print(f"初始参数: V_target={acc_decision.V_target_kmh}km/h, G2={acc_decision.G2_s}s")

    # 测试1: 从适速无史待命状态开启ACC（降速启控）
    print("\n=== 测试1: 使用I_0（降速/当速启控）开启ACC ===")
    state, decision, msg = acc_decision.process_command(ACCCommand.DECREASE_SPEED, ego_speed, has_target=False)
    print(f"结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")
    print(f"执行决策: {decision.value if decision else None}")

    # 测试2: 在控制状态下增速
    print("\n=== 测试2: 在控制状态下增速 ===")
    state, decision, msg = acc_decision.process_command(ACCCommand.INCREASE_SPEED, ego_speed, has_target=False)
    print(f"结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")
    print(f"执行决策: {decision.value if decision else None}")

    # 测试3: 在控制状态下刹车进入待命
    print("\n=== 测试3: 在控制状态下刹车进入待命 ===")
    state, decision, msg = acc_decision.process_command(ACCCommand.BRAKE, ego_speed, has_target=False)
    print(f"结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")
    print(f"执行决策: {decision.value if decision else None}")

    # 测试4: 从有史待命状态使用I_1继承控制
    print("\n=== 测试4: 从有史待命状态使用I_1继承控制 ===")
    state, decision, msg = acc_decision.process_command(ACCCommand.INCREASE_SPEED, ego_speed, has_target=False)
    print(f"结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")
    print(f"执行决策: {decision.value if decision else None}")

    # 测试5: 在控制状态下油门触发扭矩仲裁
    print("\n=== 测试5: 在控制状态下油门触发扭矩仲裁 ===")
    state, decision, msg = acc_decision.process_command(ACCCommand.THROTTLE, ego_speed, has_target=False)
    print(f"结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")
    print(f"执行决策: {decision.value if decision else None}")
    print(f"扭矩仲裁激活: {acc_decision.torque_arbitration_active}")

    # 测试6: 测试低速状态
    print("\n=== 测试6: 测试低速状态自动转换 ===")
    low_speed = 25.0  # 低于V_min_kmh
    acc_decision.update_state_by_speed(low_speed)
    print(f"低速({low_speed}km/h)后状态: {acc_decision.current_state.value}")
    
    # 在低速状态下尝试指令
    state, decision, msg = acc_decision.process_command(ACCCommand.INCREASE_SPEED, low_speed, has_target=False)
    print(f"低速状态下增速指令结果: {msg}")

    # 测试7: 检查决策输出
    print("\n=== 测试7: 检查决策输出 ===")
    decision_output = acc_decision.get_decision_output(ego_speed, 20.0)
    for key, value in decision_output.items():
        print(f"{key}: {value}")

    print("\n=== 测试完成 ===")
    return acc_decision



def test_decision_md_compliance():
    """全面测试决策系统与decision.md文档的一致性"""
    print("\n" + "="*80)
    print("=== 全面测试：决策系统与decision.md文档一致性验证 ===")
    print("="*80)

    acc_decision = ACCDecisionModule(initial_min_speed_kmh=30.0, initial_target_speed_kmh=50.0, initial_time_gap=2.0)
    acc_decision.set_debug(False)  # 关闭调试以便清晰显示测试结果
    
    def test_state_transition(test_name, current_state, command, expected_state, expected_decision, ego_speed=35.0, has_target=False):
        """测试单个状态转移"""
        print(f"\n[INFO] {test_name}")
        print(f"   当前状态: {current_state.value} -> 指令: {command.value} -> 期望: {expected_state.value}({expected_decision})")
        
        # 设置当前状态
        acc_decision.current_state = current_state
        
        # 执行指令
        result_state, result_decision, msg = acc_decision.process_command(command, ego_speed, has_target)
        
        # 验证结果
        state_correct = result_state == expected_state
        decision_correct = result_decision.value == expected_decision if result_decision else expected_decision is None
        
        status = "[PASS]" if (state_correct and decision_correct) else "[FAIL]"
        print(f"   {status} 实际结果: {result_state.value} -> {result_decision.value if result_decision else 'None'}")
        if not (state_correct and decision_correct):
            print(f"   [FAIL] 期望: {expected_state.value} -> {expected_decision}")
        print(f"   消息: {msg}")
        
        return state_correct and decision_correct

    print("\n" + "="*60)
    print("第一部分：验证状态转移表的完整性")
    print("="*60)
    
    # 根据decision.md文档，逐一测试每个状态转移
    test_results = []
    
    print("\n[S0] 在控状态的转移")
    test_results.append(test_state_transition("S0+I0 -> S0(R1)", ACCState.IN_CONTROL, ACCCommand.DECREASE_SPEED, ACCState.IN_CONTROL, "R1"))
    test_results.append(test_state_transition("S0+I1 -> S0(R2)", ACCState.IN_CONTROL, ACCCommand.INCREASE_SPEED, ACCState.IN_CONTROL, "R2"))
    test_results.append(test_state_transition("S0+I2 -> S0(R3)", ACCState.IN_CONTROL, ACCCommand.DECREASE_DISTANCE, ACCState.IN_CONTROL, "R3"))
    test_results.append(test_state_transition("S0+I3 -> S0(R4)", ACCState.IN_CONTROL, ACCCommand.INCREASE_DISTANCE, ACCState.IN_CONTROL, "R4"))
    test_results.append(test_state_transition("S0+I4 -> S0(R7)", ACCState.IN_CONTROL, ACCCommand.THROTTLE, ACCState.IN_CONTROL, "R7"))
    test_results.append(test_state_transition("S0+I5 -> S1(R8)", ACCState.IN_CONTROL, ACCCommand.BRAKE, ACCState.ADAPTIVE_HISTORY_STANDBY, "R8"))
    test_results.append(test_state_transition("S0+I6 -> S1(R8)", ACCState.IN_CONTROL, ACCCommand.CANCEL, ACCState.ADAPTIVE_HISTORY_STANDBY, "R8"))

    print("\n[TEST] S1 适速有史待命状态的转移")
    test_results.append(test_state_transition("S1+I1 -> S0(R6)", ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.INCREASE_SPEED, ACCState.IN_CONTROL, "R6"))
    test_results.append(test_state_transition("S1+I0 -> S0(R5)", ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.DECREASE_SPEED, ACCState.IN_CONTROL, "R5"))
    test_results.append(test_state_transition("S1+I2 -> S1(R8)", ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.DECREASE_DISTANCE, ACCState.ADAPTIVE_HISTORY_STANDBY, "R8"))
    test_results.append(test_state_transition("S1+I3 -> S1(R8)", ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.INCREASE_DISTANCE, ACCState.ADAPTIVE_HISTORY_STANDBY, "R8"))
    test_results.append(test_state_transition("S1+I4 -> S1(R8)", ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.THROTTLE, ACCState.ADAPTIVE_HISTORY_STANDBY, "R8"))
    test_results.append(test_state_transition("S1+I5 -> S1(R8)", ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.BRAKE, ACCState.ADAPTIVE_HISTORY_STANDBY, "R8"))
    test_results.append(test_state_transition("S1+I6 -> S1(R8)", ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.CANCEL, ACCState.ADAPTIVE_HISTORY_STANDBY, "R8"))

    print("\n[TEST] S2 适速无史待命状态的转移")
    test_results.append(test_state_transition("S2+I0 -> S0(R5)", ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.DECREASE_SPEED, ACCState.IN_CONTROL, "R5"))
    test_results.append(test_state_transition("S2+I1 -> S2(R8)", ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.INCREASE_SPEED, ACCState.ADAPTIVE_NO_HISTORY_STANDBY, "R8"))
    test_results.append(test_state_transition("S2+I2 -> S2(R8)", ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.DECREASE_DISTANCE, ACCState.ADAPTIVE_NO_HISTORY_STANDBY, "R8"))
    test_results.append(test_state_transition("S2+I3 -> S2(R8)", ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.INCREASE_DISTANCE, ACCState.ADAPTIVE_NO_HISTORY_STANDBY, "R8"))
    test_results.append(test_state_transition("S2+I4 -> S2(R8)", ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.THROTTLE, ACCState.ADAPTIVE_NO_HISTORY_STANDBY, "R8"))
    test_results.append(test_state_transition("S2+I5 -> S2(R8)", ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.BRAKE, ACCState.ADAPTIVE_NO_HISTORY_STANDBY, "R8"))
    test_results.append(test_state_transition("S2+I6 -> S2(R8)", ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.CANCEL, ACCState.ADAPTIVE_NO_HISTORY_STANDBY, "R8"))

    print("\n[TEST] S3 低速状态的转移")
    test_results.append(test_state_transition("S3+I0 -> S3(R8)", ACCState.LOW_SPEED, ACCCommand.DECREASE_SPEED, ACCState.LOW_SPEED, "R8"))
    test_results.append(test_state_transition("S3+I1 -> S3(R8)", ACCState.LOW_SPEED, ACCCommand.INCREASE_SPEED, ACCState.LOW_SPEED, "R8"))
    test_results.append(test_state_transition("S3+I2 -> S3(R8)", ACCState.LOW_SPEED, ACCCommand.DECREASE_DISTANCE, ACCState.LOW_SPEED, "R8"))
    test_results.append(test_state_transition("S3+I3 -> S3(R8)", ACCState.LOW_SPEED, ACCCommand.INCREASE_DISTANCE, ACCState.LOW_SPEED, "R8"))
    test_results.append(test_state_transition("S3+I4 -> S3(R8)", ACCState.LOW_SPEED, ACCCommand.THROTTLE, ACCState.LOW_SPEED, "R8"))
    test_results.append(test_state_transition("S3+I5 -> S3(R8)", ACCState.LOW_SPEED, ACCCommand.BRAKE, ACCState.LOW_SPEED, "R8"))
    test_results.append(test_state_transition("S3+I6 -> S3(R8)", ACCState.LOW_SPEED, ACCCommand.CANCEL, ACCState.LOW_SPEED, "R8"))

    print("\n" + "="*60)
    print("第二部分：验证决策执行的正确性")
    print("="*60)
    
    # 重置到初始状态
    acc_decision = ACCDecisionModule(initial_min_speed_kmh=30.0, initial_target_speed_kmh=50.0, initial_time_gap=2.0)
    
    def test_decision_execution(test_name, setup_commands, final_command, expected_effects):
        """测试决策执行的效果"""
        print(f"\n[INFO] {test_name}")
        
        # 执行设置命令
        for cmd in setup_commands:
            acc_decision.process_command(cmd, 35.0, False)
        
        # 记录执行前的状态
        before_params = acc_decision.get_current_parameters()
        
        # 执行最终命令
        state, decision, msg = acc_decision.process_command(final_command, 35.0, False)
        
        # 记录执行后的状态  
        after_params = acc_decision.get_current_parameters()
        
        print(f"   执行: {final_command.value} -> {decision.value if decision else 'None'}")
        print(f"   消息: {msg}")
        
        # 验证预期效果
        success = True
        for key, expected_value in expected_effects.items():
            if key in after_params:
                actual_value = after_params[key]
                if actual_value != expected_value:
                    print(f"   [FAIL] {key}: 期望={expected_value}, 实际={actual_value}")
                    success = False
                else:
                    print(f"   [PASS] {key}: {actual_value}")
        
        return success

    # 测试各种决策的执行效果
    decision_tests = []
    
    print("\n[TEST] R1 速度降低决策测试")
    acc_decision.reset()
    decision_tests.append(test_decision_execution(
        "R1测试：速度降低应该减少V_target", 
        [ACCCommand.DECREASE_SPEED],  # 先进入控制状态
        ACCCommand.DECREASE_SPEED,     # 再降速
        {'V_target_kmh': 34.0}  # 35.0 - 1.0 = 34.0
    ))
    
    print("\n[TEST] R2 速度增加决策测试")
    acc_decision.reset()
    decision_tests.append(test_decision_execution(
        "R2测试：速度增加应该增加V_target",
        [ACCCommand.DECREASE_SPEED],  # 先进入控制状态
        ACCCommand.INCREASE_SPEED,    # 再增速
        {'V_target_kmh': 36.0}  # 35.0 + 1.0 = 36.0
    ))

    print("\n[TEST] R5 无继控制决策测试")
    acc_decision.reset()
    decision_tests.append(test_decision_execution(
        "R5测试：无继控制应该使用当前车速作为目标",
        [],  # 从初始状态开始
        ACCCommand.DECREASE_SPEED,  # I0触发R5
        {'V_target_kmh': 35.0, 'state': 'S0'}  # 使用当前车速35.0
    ))

    print("\n[TEST] R7 扭矩仲裁决策测试")
    acc_decision.reset()
    decision_tests.append(test_decision_execution(
        "R7测试：扭矩仲裁应该激活仲裁标志",
        [ACCCommand.DECREASE_SPEED],  # 先进入控制状态
        ACCCommand.THROTTLE,          # 触发扭矩仲裁
        {'torque_arbitration_active': True, 'state': 'S0'}
    ))

    print("\n" + "="*60)
    print("第三部分：验证自动状态切换逻辑")
    print("="*60)
    
    def test_auto_state_switching():
        """测试自动状态切换"""
        print("\n[INFO] 自动状态切换测试")
        
        acc_decision.reset()
        print(f"   初始状态: {acc_decision.current_state.value}")
        
        # 测试1: 速度低于V_min时自动切换到低速状态
        acc_decision.current_state = ACCState.ADAPTIVE_NO_HISTORY_STANDBY
        acc_decision.update_state_by_speed(25.0)  # 低于30.0的阈值
        result1 = acc_decision.current_state == ACCState.LOW_SPEED
        print(f"   [PASS] 低速切换(25km/h -> S3): {result1}")
        
        # 测试2: 速度回升但无历史时切换到无史待命
        acc_decision.has_history = False
        acc_decision.update_state_by_speed(35.0)  # 高于30.0的阈值
        result2 = acc_decision.current_state == ACCState.ADAPTIVE_NO_HISTORY_STANDBY
        print(f"   [PASS] 无史待命切换(35km/h -> S2): {result2}")
        
        # 测试3: 速度回升且有历史时切换到有史待命
        acc_decision.has_history = True
        acc_decision.current_state = ACCState.LOW_SPEED
        acc_decision.update_state_by_speed(35.0)
        result3 = acc_decision.current_state == ACCState.ADAPTIVE_HISTORY_STANDBY
        print(f"   [PASS] 有史待命切换(35km/h -> S1): {result3}")
        
        # 测试4: 在控状态下不自动切换
        acc_decision.current_state = ACCState.IN_CONTROL
        original_state = acc_decision.current_state
        acc_decision.update_state_by_speed(25.0)  # 尝试触发低速切换
        result4 = acc_decision.current_state == original_state
        print(f"   [PASS] 在控状态不自动切换: {result4}")
        
        return all([result1, result2, result3, result4])

    auto_switch_result = test_auto_state_switching()

    print("\n" + "="*60)
    print("第四部分：验证历史数据管理")
    print("="*60)
    
    def test_history_management():
        """测试历史数据管理"""
        print("\n[INFO] 历史数据管理测试")
        
        acc_decision.reset()
        
        # 设置初始参数并进入控制状态
        acc_decision.V_target_kmh = 60.0
        acc_decision.G2_s = 2.5
        acc_decision.process_command(ACCCommand.DECREASE_SPEED, 35.0, False)
        
        # 通过刹车退出控制（应该保存历史）
        acc_decision.process_command(ACCCommand.BRAKE, 35.0, False)
        
        # 验证历史数据被保存
        history_saved = (acc_decision.has_history and 
                        acc_decision.history_V_target_kmh == 35.0 and  # R5决策设置的当前车速
                        acc_decision.history_G2_s == 2.5)
        
        print(f"   [PASS] 历史数据保存: {history_saved}")
        print(f"      has_history: {acc_decision.has_history}")
        print(f"      history_V_target: {acc_decision.history_V_target_kmh}")
        print(f"      history_G2: {acc_decision.history_G2_s}")
        
        # 使用I1继承控制
        acc_decision.process_command(ACCCommand.INCREASE_SPEED, 35.0, False)
        
        # 验证参数被恢复（R6决策应该恢复历史值）
        history_restored = acc_decision.V_target_kmh == acc_decision.history_V_target_kmh
        print(f"   [PASS] 历史数据恢复: {history_restored}")
        print(f"      当前V_target: {acc_decision.V_target_kmh}")
        
        return history_saved and history_restored

    history_result = test_history_management()

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    # 计算测试结果
    total_transitions = len(test_results)
    passed_transitions = sum(test_results)
    
    total_decisions = len(decision_tests)
    passed_decisions = sum(decision_tests)
    
    print(f"\n[STAT] 状态转移测试: {passed_transitions}/{total_transitions} 通过 ({passed_transitions/total_transitions*100:.1f}%)")
    print(f"[STAT] 决策执行测试: {passed_decisions}/{total_decisions} 通过 ({passed_decisions/total_decisions*100:.1f}%)")
    print(f"[STAT] 自动状态切换: {'通过' if auto_switch_result else '失败'}")
    print(f"[STAT] 历史数据管理: {'通过' if history_result else '失败'}")
    
    # 总体评估
    overall_success = (passed_transitions == total_transitions and 
                      passed_decisions == total_decisions and 
                      auto_switch_result and 
                      history_result)
    
    print(f"\n[RESULT] 总体评估: {'[PASS] 决策系统完全符合decision.md要求' if overall_success else '[FAIL] 存在不一致的地方'}")
    
    return overall_success


if __name__ == "__main__":
    # 运行基础测试
    basic_test = test_acc_decision_logic()
    
    # 运行全面一致性测试
    compliance_result = test_decision_md_compliance()