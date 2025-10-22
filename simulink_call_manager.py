#!/usr/bin/env python
"""
Simulink智能调用管理器
实现基于时间间隔和事件触发的Simulink调用控制，避免阻塞横向控制
"""

import time


class SimulinkCallManager:
    """
    Simulink智能调用管理器

    设计原则：
    1. 横向控制每帧执行（不受影响）
    2. Simulink按需调用（降频但不丢失输入）
    3. 关键事件立即触发
    4. 缓存结果用于跳过帧
    """

    def __init__(self, min_interval=0.1, max_wait_time=0.5):
        """
        初始化管理器

        Args:
            min_interval (float): 最小调用间隔（秒），默认0.1s = 10Hz
            max_wait_time (float): 最大等待时间（秒），超过强制调用，默认0.5s
        """
        self.min_interval = min_interval
        self.max_wait_time = max_wait_time

        # 时间跟踪
        self.last_call_time = 0.0  # 上次调用Simulink的时间
        self.pending_since = None  # pending_input首次记录的时间（用于max_wait_time检测）

        # 输入输出缓存
        self.pending_input = None  # 待处理的最新输入
        self.cached_output = None  # 缓存的Simulink输出
        self.output_timestamp = 0.0  # 输出的时间戳

        # 事件触发标志
        self.force_update = False  # 强制更新标志
        self.force_reason = ""  # 强制更新原因

        # 状态跟踪（用于检测变化）
        self.last_acc_enabled = False
        self.last_command_type = 0
        self.last_V_target = 0.0
        self.last_G2 = 0.0

        # 统计信息
        self.total_frames = 0  # 总帧数
        self.simulink_calls = 0  # Simulink调用次数
        self.skipped_frames = 0  # 跳过的帧数
        self.forced_calls = 0  # 强制调用次数

        print("✅ Simulink智能调用管理器初始化完成")
        print(f"   最小间隔: {min_interval}s ({1/min_interval:.1f}Hz)")
        print(f"   最大等待: {max_wait_time}s")

    def update_input(self, unified_input, acc_enabled):
        """
        更新输入状态（每帧调用）

        Args:
            unified_input (dict): 新的Simulink输入
            acc_enabled (bool): ACC是否启用

        Returns:
            None
        """
        self.total_frames += 1

        # 只在pending_input从None变为非空时记录pending_since
        if self.pending_input is None and unified_input is not None:
            self.pending_since = time.time()

        self.pending_input = unified_input

        # 检测关键事件
        self._detect_critical_events(unified_input, acc_enabled)

    def _detect_critical_events(self, unified_input, acc_enabled):
        """
        检测关键事件，设置强制更新标志

        关键事件：
        1. ACC开关变化
        2. 键盘指令（非NONE）
        3. 手动刹车/油门介入
        4. 参数调整
        """
        # 事件1：ACC开关变化
        if acc_enabled != self.last_acc_enabled:
            self.force_update = True
            self.force_reason = f"ACC开关: {self.last_acc_enabled}→{acc_enabled}"
            self.last_acc_enabled = acc_enabled
            return

        # 事件2：键盘指令
        command_type = unified_input.get('command_type', 0)

        # command_type回到0时重置last_command_type，允许重复触发同一指令
        if command_type == 0:
            self.last_command_type = 0
        elif command_type != self.last_command_type:
            # 新的非零指令触发强制更新
            self.force_update = True
            self.force_reason = f"键盘指令: {command_type}"
            self.last_command_type = command_type
            return

        # 事件3：手动刹车/油门介入
        manual_throttle_active = unified_input.get('manual_throttle_active', False)
        manual_brake_active = unified_input.get('manual_brake_active', False)

        if manual_throttle_active:
            self.force_update = True
            self.force_reason = "手动油门介入"
            return

        if manual_brake_active:
            self.force_update = True
            self.force_reason = "手动刹车介入"
            return

        # 事件4：参数调整
        V_target = unified_input.get('V_target_kmh', 0.0)
        G2 = unified_input.get('G2_s', 0.0)
        if abs(V_target - self.last_V_target) > 0.1:
            self.force_update = True
            self.force_reason = f"V_target变化: {self.last_V_target:.1f}→{V_target:.1f}"
            self.last_V_target = V_target
            return
        if abs(G2 - self.last_G2) > 0.01:
            self.force_update = True
            self.force_reason = f"G2变化: {self.last_G2:.2f}→{G2:.2f}"
            self.last_G2 = G2
            return

    def should_call_simulink(self):
        """
        判断是否应该调用Simulink

        Returns:
            tuple: (should_call: bool, reason: str)
        """
        current_time = time.time()

        # 情况1：强制更新标志
        if self.force_update:
            reason = f"强制更新: {self.force_reason}"
            return True, reason

        # 情况2：没有待处理输入
        if self.pending_input is None:
            return False, "无待处理输入"

        # 情况3：距离上次调用超过最小间隔
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call >= self.min_interval:
            reason = f"达到最小间隔: {time_since_last_call:.3f}s >= {self.min_interval}s"
            return True, reason

        # 情况4：待处理输入等待时间超过最大等待时间（防止遗漏）
        if self.pending_since is not None:
            wait_time = current_time - self.pending_since
            if wait_time >= self.max_wait_time:
                reason = f"超过最大等待: {wait_time:.3f}s >= {self.max_wait_time}s"
                return True, reason

        # 情况5：首次调用（没有缓存输出）
        if self.cached_output is None:
            reason = "首次调用"
            return True, reason

        # 默认：跳过此帧
        return False, f"跳过（距上次{time_since_last_call:.3f}s < {self.min_interval}s）"

    def get_pending_input(self):
        """
        获取待处理的输入

        Returns:
            dict: 待处理输入
        """
        return self.pending_input

    def mark_called(self, output):
        """
        标记Simulink已调用，缓存输出

        Args:
            output (dict): Simulink输出
        """
        self.last_call_time = time.time()
        self.cached_output = output
        self.output_timestamp = self.last_call_time
        self.pending_input = None  # 清空待处理输入
        self.pending_since = None  # 重置pending时间戳
        self.simulink_calls += 1

        # 重置强制更新标志（保留reason用于日志记录）
        if self.force_update:
            self.forced_calls += 1
            self.force_update = False
            self.force_reason = ""

    def mark_skipped(self):
        """标记跳过此帧"""
        self.skipped_frames += 1

    def get_cached_output(self):
        """
        获取缓存的Simulink输出

        Returns:
            dict: 缓存输出
        """
        return self.cached_output

    def get_output_age(self):
        """
        获取缓存输出的年龄（秒）

        Returns:
            float: 输出年龄（秒）
        """
        if self.output_timestamp == 0.0:
            return float('inf')
        return time.time() - self.output_timestamp

    def get_statistics(self):
        """
        获取统计信息

        Returns:
            dict: 统计数据
        """
        if self.total_frames == 0:
            return {
                'total_frames': 0,
                'simulink_calls': 0,
                'skipped_frames': 0,
                'forced_calls': 0,
                'call_ratio': 0.0,
                'effective_frequency': 0.0
            }

        call_ratio = self.simulink_calls / self.total_frames
        elapsed_time = time.time() - (self.last_call_time - self.simulink_calls * self.min_interval)
        if elapsed_time > 0:
            effective_frequency = self.simulink_calls / elapsed_time
        else:
            effective_frequency = 0.0

        return {
            'total_frames': self.total_frames,
            'simulink_calls': self.simulink_calls,
            'skipped_frames': self.skipped_frames,
            'forced_calls': self.forced_calls,
            'call_ratio': call_ratio,
            'effective_frequency': effective_frequency
        }

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        output_age = self.get_output_age()

        print(f"\n{'='*80}")
        print(f"Simulink调用统计")
        print(f"{'='*80}")
        print(f"总帧数:       {stats['total_frames']}")
        print(f"Simulink调用: {stats['simulink_calls']} 次")
        print(f"跳过帧数:     {stats['skipped_frames']} 帧")
        print(f"强制调用:     {stats['forced_calls']} 次")
        print(f"调用比例:     {stats['call_ratio']*100:.1f}%")
        print(f"实际频率:     {stats['effective_frequency']:.1f} Hz")
        print(f"输出年龄:     {output_age:.3f}s")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    # 测试用例
    print("=== Simulink调用管理器测试 ===\n")

    manager = SimulinkCallManager(min_interval=0.1, max_wait_time=0.5)

    # 模拟主循环
    import time

    for frame in range(100):
        # 模拟输入
        unified_input = {
            'ego_speed_kmh': 50.0,
            'command_type': 0,
            'manual_throttle_active': False,
            'V_target_kmh': 50.0,
            'G2_s': 2.0
        }

        # 第50帧模拟ACC开关
        if frame == 50:
            acc_enabled = True
        else:
            acc_enabled = False

        # 更新输入
        manager.update_input(unified_input, acc_enabled)

        # 判断是否调用
        should_call, reason = manager.should_call_simulink()

        if should_call:
            print(f"帧{frame:3d}: 调用Simulink - {reason}")
            # 模拟Simulink输出
            output = {'target_accel': 0.5}
            manager.mark_called(output)
        else:
            print(f"帧{frame:3d}: 跳过 - {reason}")
            manager.mark_skipped()

        # 模拟50Hz主循环
        time.sleep(0.02)

    # 打印统计
    manager.print_statistics()

    print("✅ 测试完成")