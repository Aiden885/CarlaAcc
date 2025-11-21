"""
ACC核心控制器 - 高内聚设计
整合决策逻辑、输入验证、状态管理
使用外化的状态转移表 (state_transitions.json)
"""
import json
import os
import numpy as np
import time
from enum import Enum
from typing import Dict, Any, Tuple, Optional


class ACCState(Enum):
    """ACC状态枚举 - 用于类型安全的状态表示"""
    ACTIVE_CONTROL = 0           # S0: 在控状态
    ADAPTIVE_HISTORY_STANDBY = 1 # S1: 适速有史待命
    ADAPTIVE_NO_HISTORY_STANDBY = 2 # S2: 适速无史待命
    LOW_SPEED = 3                # S3: 低速状态

    # 兼容性别名（用于向后兼容）
    IN_CONTROL = 0


class ACCController:
    """
    ACC核心控制器
    - 整合决策状态机 + 输入验证
    - 管理ACC状态（S0-S3）和决策（R1-R8）
    - 处理键盘指令和参数调整
    """
    
    def __init__(self, debug: bool = False, max_target_speed_kmh: float = 150.0,
                 transition_table_path: str = None):
        self.debug = debug
        self.max_target_speed_kmh = max_target_speed_kmh

        # ACC状态定义
        self.STATES = {
            'ACTIVE_CONTROL': 0,           # S0: 在控状态
            'ADAPTIVE_HISTORY_STANDBY': 1, # S1: 适速有史待命
            'ADAPTIVE_NO_HISTORY_STANDBY': 2, # S2: 适速无史待命
            'LOW_SPEED': 3                 # S3: 低速状态
        }

        # 决策定义
        self.DECISIONS = {
            'DECREASE_SPEED': 1,           # R1: 速度降低
            'INCREASE_SPEED': 2,           # R2: 速度增加
            'DECREASE_DISTANCE': 3,        # R3: 时距降低
            'INCREASE_DISTANCE': 4,        # R4: 时距增加
            'ACTIVATE_CURRENT_SPEED': 5,   # R5: 无继控制
            'ACTIVATE_INHERITED_SPEED': 6, # R6: 继承控制
            'TORQUE_ARBITRATION': 7,       # R7: 扭矩仲裁
            'SYSTEM_STANDBY': 8            # R8: 系统待命
        }

        # 加载状态转移表
        if transition_table_path is None:
            # 默认路径：与当前文件同目录下的state_transitions.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            transition_table_path = os.path.join(current_dir, 'state_transitions.json')

        try:
            with open(transition_table_path, 'r', encoding='utf-8') as f:
                self.transition_table = json.load(f)
            if self.debug:
                print(f"✅ 成功加载状态转移表: {transition_table_path}")
        except FileNotFoundError:
            raise RuntimeError(f"状态转移表文件未找到: {transition_table_path}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"状态转移表JSON解析失败: {e}")

        # ACC状态
        self.current_state = self.STATES['ADAPTIVE_NO_HISTORY_STANDBY']
        self.has_history = False
        self.last_active_decision = self.DECISIONS['SYSTEM_STANDBY']

        # 控制参数
        self.params = {
            'V_target_kmh': 50.0,
            'V_min_kmh': 30.0,
            'G2_s': 2.0
        }

        # 调试计数器
        self.debug_counter = 0
        
    def reset(self):
        """重置ACC状态"""
        self.current_state = self.STATES['ADAPTIVE_NO_HISTORY_STANDBY']
        self.has_history = False
        self.last_active_decision = self.DECISIONS['SYSTEM_STANDBY']
        
        if self.debug:
            print("🔄 ACC控制器已重置")
    
    def validate_and_process_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入验证和预处理 - 整合Input_Validator功能
        
        Args:
            raw_input: 原始输入数据
            
        Returns:
            Dict: 验证后的输入数据
        """
        # 复制并验证数值范围
        validated = raw_input.copy()
        
        # 核心数值验证
        validated['ego_speed_kmh'] = np.clip(raw_input['ego_speed_kmh'], 0, 200)
        validated['ego_speed_ms'] = np.clip(raw_input['ego_speed_ms'], 0, 60)
        validated['V_target_kmh'] = np.clip(raw_input['V_target_kmh'], 30, self.max_target_speed_kmh)
        validated['V_min_kmh'] = np.clip(raw_input['V_min_kmh'], 20, 50)
        validated['G2_s'] = np.clip(raw_input['G2_s'], 1.0, 8.0)
        
        # 确保参数一致性
        if validated['V_min_kmh'] > validated['V_target_kmh']:
            validated['V_min_kmh'] = validated['V_target_kmh'] - 10
            
        # 处理NaN和无效值
        for key in ['ego_speed_kmh', 'ego_speed_ms', 'control_error']:
            if not np.isfinite(validated.get(key, 0)):
                validated[key] = 0.0
                
        return validated
    
    def process_keyboard_command(self, command_type: int, ego_speed_kmh: float) -> Tuple[bool, int, Dict[str, float]]:
        """
        处理键盘指令 - 整合Decision_Function的键盘逻辑
        
        Args:
            command_type: 指令类型 (0=NONE, 1-7=I0-I6)
            ego_speed_kmh: 当前车速
            
        Returns:
            Tuple[control_enabled, decision, updated_params]
        """
        # 先处理参数调整（在状态机之前）
        updated_params = {}
        if command_type in [1, 2, 3, 4]:  # 参数调整指令
            speed_step = 5.0
            time_gap_step = 0.2
            in_active_control = (self.current_state == self.STATES['ACTIVE_CONTROL'])
            
            if command_type == 1:  # E键: 降速 or 当速启控
                if in_active_control:
                    new_target = max(self.params['V_min_kmh'], self.params['V_target_kmh'] - speed_step)
                    updated_params['V_target_kmh'] = new_target
                    if self.debug:
                        print(f"⌨️ E键降速: {self.params['V_target_kmh']:.1f} → {new_target:.1f} km/h")
                # 未在控时，E键用于启控，不调整速度
                    
            elif command_type == 2:  # Q键: 增速 or 继承启控
                if in_active_control:
                    new_target = min(self.max_target_speed_kmh, self.params['V_target_kmh'] + speed_step)
                    updated_params['V_target_kmh'] = new_target
                    if self.debug:
                        print(f"⌨️ Q键增速: {self.params['V_target_kmh']:.1f} → {new_target:.1f} km/h")
                # 未在控时，Q键用于启控，不调整速度
                    
            elif command_type == 3:  # T键: 降距
                new_gap = max(1.0, self.params['G2_s'] - time_gap_step)
                updated_params['G2_s'] = new_gap
                if self.debug:
                    print(f"⌨️ T键降距: {self.params['G2_s']:.1f} → {new_gap:.1f} s")
                    
            elif command_type == 4:  # R键: 增距
                new_gap = min(5.0, self.params['G2_s'] + time_gap_step)
                updated_params['G2_s'] = new_gap
                if self.debug:
                    print(f"⌨️ R键增距: {self.params['G2_s']:.1f} → {new_gap:.1f} s")
        
        # 更新参数
        self.params.update(updated_params)
        
        # 基于车速的自动状态转移
        if ego_speed_kmh < self.params['V_min_kmh']:
            if self.current_state != self.STATES['LOW_SPEED']:
                self.current_state = self.STATES['LOW_SPEED']
                if self.debug:
                    print(f"🚗 车速过低({ego_speed_kmh:.1f} < {self.params['V_min_kmh']:.1f})，转入S3低速状态")
        else:
            # 从S3恢复
            if self.current_state == self.STATES['LOW_SPEED']:
                if self.has_history:
                    self.current_state = self.STATES['ADAPTIVE_HISTORY_STANDBY']
                    if self.debug:
                        print("🚗 车速恢复，转入S1有史待命")
                else:
                    self.current_state = self.STATES['ADAPTIVE_NO_HISTORY_STANDBY']
                    if self.debug:
                        print("🚗 车速恢复，转入S2无史待命")
        
        # 状态机处理
        control_enabled, decision = self._handle_state_machine(command_type)
        
        # 更新有效决策历史
        if decision >= 1 and decision <= 7:
            self.last_active_decision = decision
            
        return control_enabled, decision, updated_params
    
    def _handle_state_machine(self, command_type: int) -> Tuple[bool, int]:
        """
        内部状态机逻辑 - 使用外化的状态转移表

        Returns:
            Tuple[control_enabled, decision]
        """
        # 从转移表中查找当前状态的转移规则
        state_key = str(self.current_state)
        # 确保command_type为整数类型（防止float导致键查找失败）
        command_key = str(int(command_type))

        # 获取当前状态的转移表
        if state_key not in self.transition_table['transitions']:
            # 未知状态，使用错误状态的默认转移
            if self.debug:
                print(f"⚠️ 未知状态 {self.current_state}，重置到S2")
            self.current_state = self.STATES['ADAPTIVE_NO_HISTORY_STANDBY']
            self.has_history = False
            return False, self.DECISIONS['SYSTEM_STANDBY']

        state_transitions = self.transition_table['transitions'][state_key]

        # 查找匹配的转移规则
        if command_key in state_transitions:
            transition = state_transitions[command_key]
        elif 'default' in state_transitions:
            transition = state_transitions['default']
        else:
            # 如果没有找到转移规则，返回待命状态
            if self.debug:
                print(f"⚠️ 状态S{self.current_state}未找到指令{command_type}的转移规则")
            return False, self.DECISIONS['SYSTEM_STANDBY']

        # 提取转移信息
        next_state = transition['next_state']
        decision = transition['decision']
        control_enabled = transition['control_enabled']
        description = transition.get('description', '')
        side_effect = transition.get('side_effect', None)

        # 处理特殊决策值 -1 (使用last_active_decision)
        if decision == -1:
            decision = self.last_active_decision

        # 执行状态转移（只在状态改变时）
        if next_state != self.current_state:
            self.current_state = next_state
            if self.debug and description:
                print(description)

        # 处理副作用
        if side_effect == 'set_has_history_true':
            self.has_history = True

        return control_enabled, decision
    
    def get_state_info(self) -> Dict[str, Any]:
        """获取当前状态信息"""
        state_names = {v: k for k, v in self.STATES.items()}
        decision_names = {v: k for k, v in self.DECISIONS.items()}
        
        return {
            'current_state': self.current_state,
            'state_name': state_names.get(self.current_state, 'UNKNOWN'),
            'has_history': self.has_history,
            'last_active_decision': self.last_active_decision,
            'last_decision_name': decision_names.get(self.last_active_decision, 'UNKNOWN'),
            'params': self.params.copy()
        }
    
    def update_debug_counter(self) -> int:
        """更新调试计数器并返回调试代码"""
        self.debug_counter += 1
        
        if self.debug_counter % 20 == 0:
            # 生成复合调试代码
            debug_code = 1000 + self.current_state * 100 + self.last_active_decision * 10
            if self.debug:
                print(f"🔍 ACC Debug: State=S{self.current_state}, Decision=R{self.last_active_decision}, Code={debug_code}")
            return debug_code
        else:
            return self.debug_counter
