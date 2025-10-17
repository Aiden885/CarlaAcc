#!/usr/bin/env python
"""
横向PID控制器
用于ACC系统的车道保持控制
  - 左右摇摆震荡 → 减小Kp，增加Kd
  - 持续偏离中心 → 增加Ki
  - 转向反应迟钝 → 增加Kp
  - 过冲后回摆 → 增加Kd，减小Kp
"""

import time


class LateralPIDController:
    """
    横向PID控制器

    用于根据车道偏移计算转向角度
    """

    def __init__(self, kp=0.3, ki=0.01, kd=0.1, output_limits=(-0.4, 0.4)):
        """
        初始化PID控制器

        Args:
            kp (float): 比例增益
            ki (float): 积分增益
            kd (float): 微分增益
            output_limits (tuple): 输出限制 (min, max)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        # PID内部状态
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = None

        # 积分饱和限制
        self.integral_limit = 2.0  # 防止积分饱和

    def reset(self):
        """重置PID状态"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = None

    def update(self, error, dt=None):
        """
        更新PID控制器

        Args:
            error (float): 当前误差（车道偏移量，米）
                          符号约定：车偏左为负，车偏右为正
            dt (float): 时间步长（秒），如果为None则自动计算

        Returns:
            float: 转向控制输出（-1到1之间，负值向左，正值向右）
        """
        current_time = time.time()

        # 计算时间步长
        if dt is None:
            if self.last_time is None:
                dt = 0.05  # 默认50ms
            else:
                dt = current_time - self.last_time

        self.last_time = current_time

        # 防止dt过大（可能是第一次调用或长时间暂停）
        if dt > 1.0:
            dt = 0.05

        # 比例项
        p_term = self.kp * error

        # 积分项（带抗饱和）
        self.integral += error * dt
        # 限制积分项，防止积分饱和
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i_term = self.ki * self.integral

        # 微分项
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative

        # 保存当前误差供下次使用
        self.previous_error = error

        # 计算总输出
        output = p_term + i_term + d_term

        # 输出限幅
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        return output

    def set_gains(self, kp=None, ki=None, kd=None):
        """
        动态调整PID增益

        Args:
            kp (float): 比例增益
            ki (float): 积分增益
            kd (float): 微分增益
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd

    def get_state(self):
        """
        获取PID状态（用于调试）

        Returns:
            dict: PID内部状态
        """
        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'integral': self.integral,
            'previous_error': self.previous_error
        }


if __name__ == '__main__':
    # 简单测试
    print("=== 横向PID控制器测试 ===\n")

    # 创建PID控制器
    pid = LateralPIDController(kp=0.3, ki=0.01, kd=0.1)

    # 模拟场景：车辆偏左0.5米
    print("场景1：车辆持续偏左0.5米")
    for i in range(5):
        error = -0.5  # 偏左为负
        output = pid.update(error, dt=0.05)
        print(f"  步骤{i+1}: 误差={error:.2f}m, 转向输出={output:.3f} (应向右转纠正)")

    print("\n场景2：车辆逐渐回到中心")
    pid.reset()
    errors = [-0.5, -0.3, -0.1, 0.0, 0.0]
    for i, error in enumerate(errors):
        output = pid.update(error, dt=0.05)
        print(f"  步骤{i+1}: 误差={error:.2f}m, 转向输出={output:.3f}")

    print("\n场景3：车辆偏右后回中")
    pid.reset()
    errors = [0.5, 0.3, 0.1, 0.0, 0.0]
    for i, error in enumerate(errors):
        output = pid.update(error, dt=0.05)
        print(f"  步骤{i+1}: 误差={error:.2f}m, 转向输出={output:.3f}")

    print("\n✅ 测试完成")