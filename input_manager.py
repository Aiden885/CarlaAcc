"""
输入管理器
统一处理所有输入（键盘、手动控制）
从acc_updated.py的handle_events和handle_keyboard_input方法中提取
"""
import time
from typing import Tuple, Optional

from pygame.locals import *

from acc_config import ACCConfig
from manual_input_controller import ManualSteeringController
from system_state import ManualInputState, KeyboardCommand


class InputManager:
    """
    输入管理器
    职责：
    1. 处理键盘事件
    2. 管理按键状态
    3. 计算手动输入值（油门/刹车/转向）
    4. 生成键盘指令
    """

    def __init__(self, config: ACCConfig):
        self.config = config

        # 手动转向控制器
        self.manual_steering_controller = ManualSteeringController()

        # 按键去抖动
        self.last_space_press_time = 0.0
        self.last_manual_steer_update = time.time()

        # 手动输入状态（使用状态对象）
        self.manual_input = ManualInputState()

    def process_keydown_event(self, key_code: int, acc_enabled: bool) -> Optional[dict]:
        """
        处理按键按下事件

        Args:
            key_code: pygame按键码
            acc_enabled: ACC系统是否开启

        Returns:
            dict: 包含事件类型和相关数据，例如：
                  {'type': 'quit'}
                  {'type': 'acc_toggle', 'enabled': True}
                  {'type': 'keyboard_command', 'code': 1, 'desc': 'E降速(I0)'}
                  {'type': 'debug_toggle'}
                  None: 无特殊事件
        """
        # ESC: 退出
        if key_code == K_ESCAPE:
            return {'type': 'quit'}

        # BACKSPACE: 强制清理并退出
        elif key_code == K_BACKSPACE:
            return {'type': 'force_quit'}

        # 空格: ACC主开关（带去抖动）
        elif key_code == K_SPACE:
            current_time = time.time()
            if current_time - self.last_space_press_time > 0.5:
                self.last_space_press_time = current_time
                return {'type': 'acc_toggle'}
            else:
                return {'type': 'debounced_space'}

        # P: 调试模式切换
        elif key_code == K_p:
            return {'type': 'debug_toggle'}

        # W: 油门键按下
        elif key_code == K_w:
            self.manual_input.w_pressed = True
            if acc_enabled:
                return {'type': 'keyboard_command', 'code': 5, 'desc': 'W油门(I4)'}

        # S: 刹车键按下
        elif key_code == K_s:
            self.manual_input.s_pressed = True
            if acc_enabled:
                return {'type': 'keyboard_command', 'code': 6, 'desc': 'S刹车(I5)'}

        # A/D: 转向键
        elif key_code == K_a:
            self.manual_input.a_pressed = True

        elif key_code == K_d:
            self.manual_input.d_pressed = True

        # ACC指令键（需要ACC开启）
        elif key_code in (K_q, K_e, K_r, K_t, K_c):
            if not acc_enabled:
                return {'type': 'acc_disabled_warning'}
            else:
                mapping = {
                    K_q: (2, 'Q增速(I1)'),
                    K_e: (1, 'E降速(I0)'),
                    K_r: (4, 'R增距(I3)'),
                    K_t: (3, 'T降距(I2)'),
                    K_c: (7, 'C取消(I6)')
                }
                code, desc = mapping[key_code]
                return {'type': 'keyboard_command', 'code': code, 'desc': desc}

        # F: 触发前车斜坡速度
        elif key_code == K_f:
            return {'type': 'trigger_ramp'}

        # Z/X: 前车换道
        elif key_code == K_z:
            return {'type': 'lane_change_left'}

        elif key_code == K_x:
            return {'type': 'lane_change_right'}

        return None

    def process_keyup_event(self, key_code: int) -> Optional[dict]:
        """
        处理按键释放事件

        Returns:
            dict: 包含事件类型和相关数据
        """
        if key_code == K_w:
            self.manual_input.w_pressed = False
            self.manual_input.throttle = 0.0
            return {'type': 'throttle_released'}

        elif key_code == K_s:
            self.manual_input.s_pressed = False
            self.manual_input.brake = 0.0
            return {'type': 'brake_released'}

        elif key_code == K_a:
            self.manual_input.a_pressed = False
            if not self.manual_input.d_pressed:
                self.manual_steering_controller.reset()
                self.manual_input.steer = 0.0

        elif key_code == K_d:
            self.manual_input.d_pressed = False
            if not self.manual_input.a_pressed:
                self.manual_steering_controller.reset()
                self.manual_input.steer = 0.0

        return None

    def update_manual_inputs(self) -> ManualInputState:
        """
        更新手动输入值（每帧调用）

        Returns:
            ManualInputState: 更新后的手动输入状态
        """
        # 油门累加（W键持续按下）
        if self.manual_input.w_pressed:
            self.manual_input.throttle = min(
                1.0,
                self.manual_input.throttle + self.config.manual_throttle_step
            )
        else:
            self.manual_input.throttle = 0.0

        # 刹车累加（S键持续按下）
        if self.manual_input.s_pressed:
            self.manual_input.brake = min(
                1.0,
                self.manual_input.brake + self.config.manual_brake_step
            )
        else:
            self.manual_input.brake = 0.0

        # 转向累加（A/D键持续按下，基于真实时间间隔）
        current_time = time.time()
        dt = current_time - self.last_manual_steer_update
        self.manual_input.steer = self.manual_steering_controller.update(
            steer_left=self.manual_input.a_pressed,
            steer_right=self.manual_input.d_pressed,
            dt_seconds=dt
        )
        self.last_manual_steer_update = current_time

        return self.manual_input

    def get_manual_input_state(self) -> ManualInputState:
        """获取当前手动输入状态"""
        return self.manual_input

    def reset(self):
        """重置所有输入状态"""
        self.manual_input = ManualInputState()
        self.manual_steering_controller.reset()
