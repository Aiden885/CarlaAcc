"""
为acc_updated.py添加实时模式补丁

使用方法：
在acc_updated.py的主循环中，在world.tick()之后添加速率限制
"""
import time


class RealtimeRateLimiter:
    """实时速率限制器"""

    def __init__(self, target_fps=20):
        """
        Args:
            target_fps: 目标帧率（应与fixed_delta_seconds匹配）
        """
        self.target_dt = 1.0 / target_fps
        self.last_tick_time = None

    def wait(self):
        """等待到下一帧时间"""
        current_time = time.time()

        if self.last_tick_time is not None:
            elapsed = current_time - self.last_tick_time
            sleep_time = self.target_dt - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        self.last_tick_time = time.time()


print(__doc__)
