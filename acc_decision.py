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
import math
from three_mode_controller import set_two_mode_parameters, get_two_mode_status


class ACCCommand(Enum):
    """ACC指令枚举"""
    DECREASE_SPEED = "I0"  # 降速
    INCREASE_SPEED = "I1"  # 增速
    DECREASE_DISTANCE = "I2"  # 降距
    INCREASE_DISTANCE = "I3"  # 增距
    ENGAGE = "I4"  # 开启
    THROTTLE = "I5"  # 油门
    BRAKE = "I6"  # 刹车
    EXIT = "I7"  # 退出
    CRUISE_MODE = "I8"  # 一键定速巡航


class ACCState(Enum):
    """ACC状态枚举"""
    IN_CONTROL = "S0"  # 在控
    ADAPTIVE_HISTORY_STANDBY = "S1"  # 适速有史待命
    ADAPTIVE_NO_HISTORY_STANDBY = "S2"  # 适速无史待命
    LOW_SPEED = "S3"  # 低速
    SYSTEM_STANDBY = "STANDBY"  # 系统待命
    SYSTEM_EXIT = "EXIT"  # 系统退出
    CRUISE_ONLY = "CRUISE"  # 纯定速巡航


class ACCControlMode(Enum):
    """ACC控制模式"""
    CONTINUE_CONTROL = "CONTINUE"  # 继续控制
    NO_CONTINUE_CONTROL = "NO_CONTINUE"  # 无继控制
    TARGET_DECREASE = "TARGET_DEC"  # 目标减量
    TARGET_INCREASE = "TARGET_INC"  # 目标增量
    DISTANCE_DECREASE = "DIST_DEC"  # 距离降低
    DISTANCE_INCREASE = "DIST_INC"  # 距离增加
    CRUISE_MODE_ENGAGE = "CRUISE_ENGAGE"  # 进入定速巡航模式


class ACCDecisionModule:
    """
    ACC决策模块 - 两模式版本
    负责处理ACC指令、管理状态转移和输出控制决策
    支持基于两模式控制（时距控制+定速控制）的速度和距离调整逻辑
    支持一键定速巡航功能
    添加了当前控制模式追踪
    """

    def __init__(self, initial_min_speed_kmh=30.0, initial_target_speed_kmh=50.0, initial_time_gap=2.0):
        """
        初始化ACC决策模块

        Args:
            initial_min_speed_kmh: 最低速度要求 (km/h) - 低速模式判断阈值
            initial_target_speed_kmh: 目标速度 (km/h) - 两模式控制的切换速度和目标速度
            initial_time_gap: 初始时间间隔 (s) - 对应两模式的G2
        """
        # 当前系统状态
        self.current_state = ACCState.SYSTEM_EXIT
        self.previous_state = None

        # === 新增：当前控制模式追踪 ===
        self.current_control_mode = None
        self.previous_control_mode = None

        # === 两模式控制参数 ===
        self.V_min_kmh = initial_min_speed_kmh  # 最低速度要求（低速判断阈值）
        self.V_target_kmh = initial_target_speed_kmh  # 目标速度（模式切换阈值和目标速度）
        self.G2_s = initial_time_gap  # 时距参数

        # 调整步长
        self.speed_step = 1.0  # 速度调整步长：1 km/h per command
        self.distance_step = 2.0  # 距离调整步长：2 m per command

        # === 定速巡航模式控制 ===
        self.cruise_mode_active = False  # 是否处于纯定速巡航模式
        self.force_cruise_mode = False  # 是否强制忽略前车（一键定速）

        # 历史状态管理
        self.has_history = False
        self.history_V_target_kmh = None
        self.history_G2_s = None
        self.last_control_time = None

        # 待存储的距离设定（无前车时的增距/减距指令）
        self.pending_distance_adjustment = 0.0  # 累积的距离调整

        # 状态转移表
        self._initialize_transition_table()

        # 调试模式
        self.debug = False

        # 初始化两模式参数
        self._update_two_mode_parameters()

        print("ACC决策模块初始化完成")
        print(f"初始参数: V_min={self.V_min_kmh}km/h, V_target={self.V_target_kmh}km/h, G2={self.G2_s}s")

    def _initialize_transition_table(self):
        """初始化状态转移表"""
        self.transition_table = {
            # 在控状态的转移
            (ACCState.IN_CONTROL, ACCCommand.DECREASE_SPEED): (ACCState.IN_CONTROL, ACCControlMode.TARGET_DECREASE),
            (ACCState.IN_CONTROL, ACCCommand.INCREASE_SPEED): (ACCState.IN_CONTROL, ACCControlMode.TARGET_INCREASE),
            (ACCState.IN_CONTROL, ACCCommand.DECREASE_DISTANCE): (ACCState.IN_CONTROL,
                                                                  ACCControlMode.DISTANCE_DECREASE),
            (ACCState.IN_CONTROL, ACCCommand.INCREASE_DISTANCE): (ACCState.IN_CONTROL,
                                                                  ACCControlMode.DISTANCE_INCREASE),
            (ACCState.IN_CONTROL, ACCCommand.CRUISE_MODE): (ACCState.CRUISE_ONLY, ACCControlMode.CRUISE_MODE_ENGAGE),
            (ACCState.IN_CONTROL, ACCCommand.THROTTLE): (ACCState.ADAPTIVE_HISTORY_STANDBY, None),
            (ACCState.IN_CONTROL, ACCCommand.BRAKE): (ACCState.ADAPTIVE_HISTORY_STANDBY, None),
            (ACCState.IN_CONTROL, ACCCommand.EXIT): (ACCState.SYSTEM_EXIT, None),

            # 适速有史待命状态的转移
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.ENGAGE): (ACCState.IN_CONTROL,
                                                                     ACCControlMode.CONTINUE_CONTROL),
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.CRUISE_MODE): (ACCState.CRUISE_ONLY,
                                                                          ACCControlMode.CRUISE_MODE_ENGAGE),
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.THROTTLE): (ACCState.SYSTEM_STANDBY, None),
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.BRAKE): (ACCState.SYSTEM_STANDBY, None),
            (ACCState.ADAPTIVE_HISTORY_STANDBY, ACCCommand.EXIT): (ACCState.SYSTEM_EXIT, None),

            # 适速无史待命状态的转移
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.ENGAGE): (ACCState.IN_CONTROL,
                                                                        ACCControlMode.NO_CONTINUE_CONTROL),
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.CRUISE_MODE): (ACCState.CRUISE_ONLY,
                                                                             ACCControlMode.CRUISE_MODE_ENGAGE),
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.THROTTLE): (ACCState.SYSTEM_STANDBY, None),
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.BRAKE): (ACCState.SYSTEM_STANDBY, None),
            (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, ACCCommand.EXIT): (ACCState.SYSTEM_EXIT, None),

            # 纯定速巡航状态的转移
            (ACCState.CRUISE_ONLY, ACCCommand.DECREASE_SPEED): (ACCState.CRUISE_ONLY, ACCControlMode.TARGET_DECREASE),
            (ACCState.CRUISE_ONLY, ACCCommand.INCREASE_SPEED): (ACCState.CRUISE_ONLY, ACCControlMode.TARGET_INCREASE),
            (ACCState.CRUISE_ONLY, ACCCommand.ENGAGE): (ACCState.IN_CONTROL, ACCControlMode.CONTINUE_CONTROL),
            # 切换回自适应模式
            (ACCState.CRUISE_ONLY, ACCCommand.THROTTLE): (ACCState.ADAPTIVE_HISTORY_STANDBY, None),
            (ACCState.CRUISE_ONLY, ACCCommand.BRAKE): (ACCState.ADAPTIVE_HISTORY_STANDBY, None),
            (ACCState.CRUISE_ONLY, ACCCommand.EXIT): (ACCState.SYSTEM_EXIT, None),

            # 低速状态的转移
            (ACCState.LOW_SPEED, ACCCommand.ENGAGE): (ACCState.IN_CONTROL, ACCControlMode.NO_CONTINUE_CONTROL),
            (ACCState.LOW_SPEED, ACCCommand.CRUISE_MODE): (ACCState.CRUISE_ONLY, ACCControlMode.CRUISE_MODE_ENGAGE),
            (ACCState.LOW_SPEED, ACCCommand.EXIT): (ACCState.SYSTEM_EXIT, None),

            # 系统待命状态的转移（可以重新激活）
            (ACCState.SYSTEM_STANDBY, ACCCommand.ENGAGE): (ACCState.ADAPTIVE_HISTORY_STANDBY, None),
            (ACCState.SYSTEM_STANDBY, ACCCommand.CRUISE_MODE): (ACCState.CRUISE_ONLY,
                                                                ACCControlMode.CRUISE_MODE_ENGAGE),
            (ACCState.SYSTEM_STANDBY, ACCCommand.EXIT): (ACCState.SYSTEM_EXIT, None),

            # === 系统退出状态的转移 ===
            (ACCState.SYSTEM_EXIT, ACCCommand.ENGAGE): (ACCState.ADAPTIVE_NO_HISTORY_STANDBY, None),
            (ACCState.SYSTEM_EXIT, ACCCommand.CRUISE_MODE): (ACCState.CRUISE_ONLY, ACCControlMode.CRUISE_MODE_ENGAGE),
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
        处理ACC指令并执行状态转移

        Args:
            command: ACC指令
            ego_speed_kmh: 当前车速 (km/h)
            has_target: 是否存在前车目标
            current_distance: 当前与前车距离 (m, 可选)

        Returns:
            tuple: (新状态, 控制模式, 执行结果消息)
        """
        if not isinstance(command, ACCCommand):
            return self.current_state, None, "无效指令"

        if self.debug:
            print(f"处理指令: {command.value}, 当前状态: {self.current_state.value}, 有前车: {has_target}")

        # 保存当前状态作为历史
        self.previous_state = self.current_state
        self.previous_control_mode = self.current_control_mode

        # === 特殊逻辑：增速/减速指令的条件检查 ===
        if command in [ACCCommand.INCREASE_SPEED, ACCCommand.DECREASE_SPEED]:
            # 在纯定速巡航模式下，增速/减速始终有效
            if self.current_state == ACCState.CRUISE_ONLY:
                pass  # 允许执行
            elif has_target and not self.force_cruise_mode:
                return self.current_state, None, f"存在前车时，{command.value}指令无效（由两模式控制器自动控制）。使用定速巡航模式可忽略前车。"

        # === 特殊逻辑：增距/减距指令的条件检查 ===
        if command in [ACCCommand.INCREASE_DISTANCE, ACCCommand.DECREASE_DISTANCE]:
            # 在纯定速巡航模式下，增距/减距指令无效
            if self.current_state == ACCState.CRUISE_ONLY:
                return self.current_state, None, f"纯定速巡航模式下，{command.value}指令无效（忽略前车距离）"
            elif not has_target:
                # 无前车时，先存储调整量
                adjustment = self.distance_step if command == ACCCommand.INCREASE_DISTANCE else -self.distance_step
                self.pending_distance_adjustment += adjustment
                return self.current_state, None, f"无前车时，{command.value}指令已存储（调整量: {self.pending_distance_adjustment:+.1f}m）"

        # 查找状态转移
        transition_key = (self.current_state, command)

        if transition_key in self.transition_table:
            new_state, control_mode = self.transition_table[transition_key]

            # 保存原状态用于调试输出
            old_state_for_debug = self.current_state

            # === 新增：更新当前控制模式 ===
            self.current_control_mode = control_mode

            # 执行状态转移
            success_msg = self._execute_state_transition(new_state, control_mode, command, ego_speed_kmh, has_target,
                                                         current_distance)

            if self.debug:
                print(f"状态转移: {old_state_for_debug.value} + {command.value} -> {new_state.value}")
                if control_mode:
                    print(f"控制模式: {control_mode.value}")

            return new_state, control_mode, success_msg
        else:
            # 无效的状态转移
            return self.current_state, None, f"当前状态 {self.current_state.value} 下不支持指令 {command.value}"

    def _execute_state_transition(self, new_state, control_mode, command, ego_speed_kmh, has_target=False,
                                  current_distance=None):
        """执行具体的状态转移逻辑"""
        old_state = self.current_state
        self.current_state = new_state

        if self.debug:
            print(f"执行状态转移: {old_state.value} -> {new_state.value}")

        if new_state == ACCState.IN_CONTROL:
            if control_mode == ACCControlMode.CONTINUE_CONTROL:
                # 从定速巡航切换回自适应模式时，需要关闭强制定速标志
                if old_state == ACCState.CRUISE_ONLY:
                    if self.debug:
                        print("从定速巡航切换回自适应模式，清除定速标志")
                    self.cruise_mode_active = False
                    self.force_cruise_mode = False
                    msg_suffix = "（已退出定速巡航模式）"
                else:
                    msg_suffix = ""

                # 恢复历史设定
                if self.history_V_target_kmh:
                    self.V_target_kmh = self.history_V_target_kmh
                if self.history_G2_s:
                    self.G2_s = self.history_G2_s

                # 应用待存储的距离调整（直接调整G2_s）
                if self.pending_distance_adjustment != 0:
                    # 将距离调整转换为时距调整（假设当前速度为50km/h作为基准）
                    base_speed_ms = 50.0 / 3.6
                    time_adjustment = self.pending_distance_adjustment / base_speed_ms
                    self.G2_s = max(1.0, self.G2_s + time_adjustment)
                    adjustment_msg = f"应用待存储距离调整: {self.pending_distance_adjustment:+.1f}m -> G2调整{time_adjustment:+.1f}s"
                    self.pending_distance_adjustment = 0.0
                else:
                    adjustment_msg = ""

                self._update_two_mode_parameters()
                return f"继续控制: 恢复目标速度={self.V_target_kmh:.1f}km/h, G2={self.G2_s:.1f}s {adjustment_msg}{msg_suffix}"

            elif control_mode == ACCControlMode.NO_CONTINUE_CONTROL:
                # 使用当前设定开始新的控制
                # 应用待存储的距离调整（直接调整G2_s）
                if self.pending_distance_adjustment != 0:
                    # 将距离调整转换为时距调整（假设当前速度为50km/h作为基准）
                    base_speed_ms = 50.0 / 3.6
                    time_adjustment = self.pending_distance_adjustment / base_speed_ms
                    self.G2_s = max(1.0, self.G2_s + time_adjustment)
                    adjustment_msg = f"应用待存储距离调整: {self.pending_distance_adjustment:+.1f}m -> G2调整{time_adjustment:+.1f}s"
                    self.pending_distance_adjustment = 0.0
                else:
                    adjustment_msg = ""

                self._update_two_mode_parameters()
                return f"无继控制: 使用目标速度={self.V_target_kmh:.1f}km/h, G2={self.G2_s:.1f}s {adjustment_msg}"

            elif control_mode == ACCControlMode.TARGET_DECREASE:
                # 减速：直接调整目标速度
                self.V_target_kmh = max(self.V_min_kmh + 1, self.V_target_kmh - self.speed_step)
                self._update_two_mode_parameters()
                return f"减速: 目标速度调至{self.V_target_kmh:.1f}km/h"

            elif control_mode == ACCControlMode.TARGET_INCREASE:
                # 增速：直接调整目标速度
                self.V_target_kmh = min(120.0, self.V_target_kmh + self.speed_step)
                self._update_two_mode_parameters()
                return f"增速: 目标速度调至{self.V_target_kmh:.1f}km/h"

            elif control_mode == ACCControlMode.DISTANCE_DECREASE:
                # 减距：直接调整时距参数G2_s（减少时距）
                # 将距离调整转换为时距调整（假设当前速度为50km/h作为基准）
                base_speed_ms = 50.0 / 3.6
                time_adjustment = -self.distance_step / base_speed_ms  # 减距对应减少时距
                self.G2_s = max(1.0, self.G2_s + time_adjustment)
                self._update_two_mode_parameters()
                return f"距离降低: G2调整{time_adjustment:.2f}s，新G2={self.G2_s:.2f}s（时距控制）"

            elif control_mode == ACCControlMode.DISTANCE_INCREASE:
                # 增距：直接调整时距参数G2_s（增加时距）
                # 将距离调整转换为时距调整（假设当前速度为50km/h作为基准）
                base_speed_ms = 50.0 / 3.6
                time_adjustment = self.distance_step / base_speed_ms  # 增距对应增加时距
                self.G2_s = max(1.0, self.G2_s + time_adjustment)
                self._update_two_mode_parameters()
                return f"距离增加: G2调整{time_adjustment:.2f}s，新G2={self.G2_s:.2f}s（时距控制）"

            # 记录控制开始时间
            self.last_control_time = time.time()

        elif new_state == ACCState.CRUISE_ONLY:
            if control_mode == ACCControlMode.CRUISE_MODE_ENGAGE:
                # 进入纯定速巡航状态
                self.cruise_mode_active = True
                self.force_cruise_mode = True
                # 应用待存储的距离调整（转换为时距调整）
                if self.pending_distance_adjustment != 0:
                    base_speed_ms = 50.0 / 3.6
                    time_adjustment = self.pending_distance_adjustment / base_speed_ms
                    self.G2_s = max(1.0, self.G2_s + time_adjustment)
                    self.pending_distance_adjustment = 0.0
                self._update_two_mode_parameters()
                return f"纯定速巡航: 忽略前车，按V_target={self.V_target_kmh:.1f}km/h巡航"
            elif control_mode == ACCControlMode.TARGET_INCREASE:
                # 定速巡航模式下增速
                self.V_target_kmh = min(120.0, self.V_target_kmh + self.speed_step)
                self._update_two_mode_parameters()
                return f"目标增量: 新V_target={self.V_target_kmh:.1f}km/h（纯定速巡航）"
            elif control_mode == ACCControlMode.TARGET_DECREASE:
                # 定速巡航模式下减速
                self.V_target_kmh = max(self.V_min_kmh + 1, self.V_target_kmh - self.speed_step)
                self._update_two_mode_parameters()
                return f"目标减量: 新V_target={self.V_target_kmh:.1f}km/h（纯定速巡航）"

            # 记录控制开始时间
            self.last_control_time = time.time()

        elif new_state == ACCState.ADAPTIVE_HISTORY_STANDBY or new_state == ACCState.ADAPTIVE_NO_HISTORY_STANDBY:
            # 进入待命状态（通常是从系统退出状态转移而来）
            if new_state == ACCState.ADAPTIVE_NO_HISTORY_STANDBY:
                # 从退出状态开启，进入无史待命
                return f"系统激活: 进入适速无史待命状态"
            else:
                # 从其他状态进入有史待命
                return f"进入待命状态: 保持历史设定"

        elif new_state == ACCState.SYSTEM_STANDBY:
            # 保存当前设定作为历史
            self._save_history()
            # 清除控制模式
            self.current_control_mode = None
            return "系统待命: 人工操作优先，ACC暂停"

        elif new_state == ACCState.SYSTEM_EXIT:
            # 清除历史和待存储调整
            self._clear_history()
            # 清除控制模式
            self.current_control_mode = None
            return "系统退出: ACC完全关闭"

        return f"转移到状态: {new_state.value}"

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
        self.pending_distance_adjustment = 0.0  # 清除待存储调整
        self.cruise_mode_active = False
        self.force_cruise_mode = False
        self.current_control_mode = None  # 清除控制模式

        if self.debug:
            print("清除历史设定、待存储调整、定速模式标志和控制模式")

    def update_state_by_speed(self, ego_speed_kmh):
        """
        根据车速自动更新状态（仅在非控制状态下）

        Args:
            ego_speed_kmh: 当前车速 (km/h)
        """
        if self.current_state not in [ACCState.IN_CONTROL, ACCState.CRUISE_ONLY, ACCState.SYSTEM_EXIT]:
            ideal_state = self.determine_state_from_speed(ego_speed_kmh)

            # 只有当状态确实需要改变时才更新
            if self.current_state != ideal_state:
                self.previous_state = self.current_state
                self.current_state = ideal_state

                if self.debug:
                    print(f"根据车速更新状态: {ego_speed_kmh:.1f}km/h -> {ideal_state.value}")

        # 重要：如果不在巡航状态，确保巡航标志为False
        if self.current_state != ACCState.CRUISE_ONLY:
            if self.cruise_mode_active or self.force_cruise_mode:
                self.cruise_mode_active = False
                self.force_cruise_mode = False
                if self.debug:
                    print("退出巡航状态，清除巡航模式标志")

    def get_current_parameters(self):
        """
        获取当前ACC参数

        Returns:
            dict: 当前参数字典
        """
        return {
            'V_target_kmh': self.V_target_kmh,
            'V_min_kmh': self.V_min_kmh,
            'G2_s': self.G2_s,
            'state': self.current_state.value,
            'has_history': self.has_history,
            'is_active': self.current_state in [ACCState.IN_CONTROL, ACCState.CRUISE_ONLY],
            'pending_distance_adjustment': self.pending_distance_adjustment,
            'cruise_mode_active': self.cruise_mode_active,
            'force_cruise_mode': self.force_cruise_mode,
            'current_control_mode': self.current_control_mode.value if self.current_control_mode else None
        }

    def get_decision_output(self, ego_speed_kmh, current_distance=None):
        """
        获取决策输出，供控制模块使用

        Args:
            ego_speed_kmh: 当前车速 (km/h)
            current_distance: 当前目标距离 (m, 可选)

        Returns:
            dict: 决策输出
        """
        # 根据速度自动更新状态
        self.update_state_by_speed(ego_speed_kmh)

        # 检查是否有前车（但在强制定速模式下忽略）
        has_target = current_distance is not None and current_distance < 100.0
        effective_has_target = has_target and not self.force_cruise_mode

        decision = {
            'acc_active': self.current_state in [ACCState.IN_CONTROL, ACCState.CRUISE_ONLY],
            'V_target_kmh': self.V_target_kmh,
            'V_target_ms': self.V_target_kmh / 3.6,
            'V_min_kmh': self.V_min_kmh,
            'G2_s': self.G2_s,
            'state': self.current_state.value,
            'state_description': self._get_state_description(),
            'control_enabled': self.current_state in [ACCState.IN_CONTROL, ACCState.CRUISE_ONLY],
            'has_target': has_target,
            'effective_has_target': effective_has_target,  # 考虑强制定速模式后的有效前车状态
            'cruise_mode_active': self.cruise_mode_active,
            'force_cruise_mode': self.force_cruise_mode,
            'pending_distance_adjustment': self.pending_distance_adjustment,
            'current_control_mode': self.current_control_mode.value if self.current_control_mode else None
        }

        return decision

    def _get_state_description(self):
        """获取状态描述"""
        descriptions = {
            ACCState.IN_CONTROL: "ACC主动控制中",
            ACCState.ADAPTIVE_HISTORY_STANDBY: "适速有史待命",
            ACCState.ADAPTIVE_NO_HISTORY_STANDBY: "适速无史待命",
            ACCState.LOW_SPEED: "低速状态",
            ACCState.SYSTEM_STANDBY: "系统待命",
            ACCState.SYSTEM_EXIT: "系统退出",
            ACCState.CRUISE_ONLY: "纯定速巡航"
        }
        return descriptions.get(self.current_state, "未知状态")

    def is_in_active_control_mode(self):
        """
        判断是否处于主动控制模式

        Returns:
            bool: 是否处于需要执行控制的模式
        """
        if self.current_control_mode is None:
            return False

        active_modes = [
            ACCControlMode.CONTINUE_CONTROL,
            ACCControlMode.NO_CONTINUE_CONTROL,
            ACCControlMode.TARGET_DECREASE,
            ACCControlMode.TARGET_INCREASE,
            ACCControlMode.DISTANCE_DECREASE,
            ACCControlMode.DISTANCE_INCREASE,
            ACCControlMode.CRUISE_MODE_ENGAGE
        ]

        return self.current_control_mode in active_modes

    def set_debug(self, enable):
        """启用/禁用调试模式"""
        self.debug = enable

    def reset(self):
        """重置ACC决策模块"""
        self.current_state = ACCState.SYSTEM_EXIT
        self.previous_state = None
        self.current_control_mode = None
        self.previous_control_mode = None
        self._clear_history()
        print("ACC决策模块已重置")

    def get_status_info(self):
        """获取详细状态信息"""
        return {
            'current_state': self.current_state.value,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'current_control_mode': self.current_control_mode.value if self.current_control_mode else None,
            'previous_control_mode': self.previous_control_mode.value if self.previous_control_mode else None,
            'V_target_kmh': self.V_target_kmh,
            'V_min_kmh': self.V_min_kmh,
            'G2_s': self.G2_s,
            'has_history': self.has_history,
            'history_V_target_kmh': self.history_V_target_kmh,
            'history_G2_s': self.history_G2_s,
            'last_control_time': self.last_control_time,
            'state_description': self._get_state_description(),
            'pending_distance_adjustment': self.pending_distance_adjustment,
            'cruise_mode_active': self.cruise_mode_active,
            'force_cruise_mode': self.force_cruise_mode,
            'is_in_active_control_mode': self.is_in_active_control_mode()
        }


def test_acc_decision_logic():
    """测试ACC决策模块的逻辑"""
    print("=== ACC决策模块逻辑测试（包含控制模式追踪） ===")

    # 创建决策模块
    acc_decision = ACCDecisionModule(initial_target_speed_kmh=50.0, initial_time_gap=2.0)
    acc_decision.set_debug(True)

    ego_speed = 35.0  # km/h

    print(f"\n初始状态: {acc_decision.current_state.value}")
    print(f"初始控制模式: {acc_decision.current_control_mode}")
    print(f"初始参数: V_target={acc_decision.V_target_kmh}km/h, G2={acc_decision.G2_s}s")

    # 测试1: 从退出状态开启ACC
    print("\n=== 测试1: 从退出状态开启ACC ===")
    state, mode, msg = acc_decision.process_command(ACCCommand.ENGAGE, ego_speed, has_target=False)
    print(f"结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")
    print(f"当前控制模式: {acc_decision.current_control_mode}")

    # 测试2: 进入控制状态
    print("\n=== 测试2: 再次开启进入控制状态 ===")
    state, mode, msg = acc_decision.process_command(ACCCommand.ENGAGE, ego_speed, has_target=False)
    print(f"结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")
    print(f"当前控制模式: {acc_decision.current_control_mode.value if acc_decision.current_control_mode else None}")
    print(f"是否处于主动控制模式: {acc_decision.is_in_active_control_mode()}")

    # 测试3: 增速指令
    print("\n=== 测试3: 增速指令 ===")
    state, mode, msg = acc_decision.process_command(ACCCommand.INCREASE_SPEED, ego_speed, has_target=False)
    print(f"结果: {msg}")
    print(f"当前控制模式: {acc_decision.current_control_mode.value if acc_decision.current_control_mode else None}")
    print(f"是否处于主动控制模式: {acc_decision.is_in_active_control_mode()}")

    # 测试4: 切换到定速巡航模式
    print("\n=== 测试4: 切换到定速巡航模式 ===")
    state, mode, msg = acc_decision.process_command(ACCCommand.CRUISE_MODE, ego_speed, has_target=True,
                                                    current_distance=20.0)
    print(f"结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")
    print(f"当前控制模式: {acc_decision.current_control_mode.value if acc_decision.current_control_mode else None}")
    print(f"是否处于主动控制模式: {acc_decision.is_in_active_control_mode()}")

    # 测试5: 检查决策输出
    print("\n=== 测试5: 检查决策输出 ===")
    decision = acc_decision.get_decision_output(ego_speed, 20.0)
    print(f"控制模式: {decision['current_control_mode']}")
    print(f"控制启用: {decision['control_enabled']}")
    print(f"强制定速模式: {decision['force_cruise_mode']}")

    # 测试6: 距离调整指令
    print("\n=== 测试6: 距离调整指令（有前车） ===")
    # 先切换回自适应模式
    acc_decision.process_command(ACCCommand.ENGAGE, ego_speed, has_target=True, current_distance=20.0)
    state, mode, msg = acc_decision.process_command(ACCCommand.INCREASE_DISTANCE, ego_speed, has_target=True,
                                                    current_distance=20.0)
    print(f"结果: {msg}")
    print(f"当前控制模式: {acc_decision.current_control_mode.value if acc_decision.current_control_mode else None}")
    print(f"是否处于主动控制模式: {acc_decision.is_in_active_control_mode()}")

    # 测试7: 人工干预
    print("\n=== 测试7: 人工刹车干预 ===")
    state, mode, msg = acc_decision.process_command(ACCCommand.BRAKE, ego_speed)
    print(f"结果: {msg}")
    print(f"当前状态: {acc_decision.current_state.value}")
    print(f"当前控制模式: {acc_decision.current_control_mode}")
    print(f"是否处于主动控制模式: {acc_decision.is_in_active_control_mode()}")

    # 获取最终状态
    print("\n=== 最终状态 ===")
    status = acc_decision.get_status_info()
    for key, value in status.items():
        if value is not None:
            print(f"{key}: {value}")


if __name__ == "__main__":
    test_acc_decision_logic()