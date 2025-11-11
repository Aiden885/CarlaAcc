"""
前车斜坡速度控制器
实现前车速度的线性斜坡变化，用于测试ACC系统的跟随响应
"""

import time


class RampSpeedController:
    """
    斜坡速度控制器

    功能：控制前车速度按照线性斜坡变化
    - 触发后从起始速度线性增加/减少到目标速度
    - 达到目标速度后保持恒定

    参数：
        start_speed_kmh: 起始速度 (km/h)
        target_speed_kmh: 目标速度 (km/h)
        duration_s: 斜坡持续时间 (秒)
    """

    def __init__(self, start_speed_kmh=50.0, target_speed_kmh=90.0, duration_s=10.0):
        """
        初始化斜坡速度控制器

        Args:
            start_speed_kmh: 起始速度 (km/h)
            target_speed_kmh: 目标速度 (km/h)
            duration_s: 斜坡持续时间 (秒)
        """
        self.start_speed_kmh = start_speed_kmh
        self.target_speed_kmh = target_speed_kmh
        self.duration_s = duration_s

        # 计算速度斜率 (km/h per second)
        self.speed_slope = (target_speed_kmh - start_speed_kmh) / duration_s

        # 状态变量
        self.is_active = False
        self.ramp_start_time = None

    def trigger(self):
        """触发斜坡速度变化"""
        self.is_active = True
        self.ramp_start_time = time.time()

        direction = "加速" if self.speed_slope > 0 else "减速"
        print(f"\n{'='*60}")
        print(f"🚗 前车斜坡速度已触发 ({direction})")
        print(f"   起始速度: {self.start_speed_kmh:.1f} km/h")
        print(f"   目标速度: {self.target_speed_kmh:.1f} km/h")
        print(f"   斜坡时间: {self.duration_s:.1f} 秒")
        print(f"   速度斜率: {self.speed_slope:.2f} km/h/s")
        print(f"{'='*60}\n")

    def get_target_speed(self):
        """
        获取当前目标速度

        Returns:
            float: 当前目标速度 (km/h)
        """
        if not self.is_active:
            return None

        elapsed_time = time.time() - self.ramp_start_time

        # 斜坡阶段：线性变化
        if elapsed_time < self.duration_s:
            current_speed = self.start_speed_kmh + self.speed_slope * elapsed_time
            return current_speed

        # 斜坡结束：保持目标速度
        else:
            # 只在刚完成时打印一次
            if elapsed_time < self.duration_s + 0.1:  # 容差0.1秒
                print(f"\n✅ 前车斜坡速度已完成，保持 {self.target_speed_kmh:.1f} km/h 恒定\n")
            return self.target_speed_kmh

    def reset(self):
        """重置控制器状态"""
        self.is_active = False
        self.ramp_start_time = None
        print("\n🔄 前车斜坡速度已重置\n")

    def is_ramp_active(self):
        """
        检查斜坡是否激活

        Returns:
            bool: True表示斜坡激活，False表示未激活
        """
        return self.is_active

    def update_parameters(self, start_speed_kmh=None, target_speed_kmh=None, duration_s=None):
        """
        更新斜坡参数（仅在未激活时可用）

        Args:
            start_speed_kmh: 新的起始速度 (km/h)
            target_speed_kmh: 新的目标速度 (km/h)
            duration_s: 新的斜坡时间 (秒)

        Returns:
            bool: True表示更新成功，False表示更新失败（斜坡正在运行）
        """
        if self.is_active:
            print("⚠️ 斜坡正在运行，无法更新参数")
            return False

        if start_speed_kmh is not None:
            self.start_speed_kmh = start_speed_kmh
        if target_speed_kmh is not None:
            self.target_speed_kmh = target_speed_kmh
        if duration_s is not None:
            self.duration_s = duration_s

        # 重新计算斜率
        self.speed_slope = (self.target_speed_kmh - self.start_speed_kmh) / self.duration_s

        print(f"✅ 斜坡参数已更新: {self.start_speed_kmh:.1f}→{self.target_speed_kmh:.1f} km/h, 时长{self.duration_s:.1f}秒")
        return True
