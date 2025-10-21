#!/usr/bin/env python
"""
Stanley横向控制器
用于ACC系统的车道保持控制（全速域优化，特别适用于高速>100km/h场景）

算法特点：
- 结合航向误差和横向误差的双重校正
- 速度自适应增益（高速时修正更平缓）
- 前轴参考点（比后轴响应更快）
- arctan限制输出范围，避免过冲

控制律：δ(t) = ψe(t) + arctan(k * e(t) / (v(t) + k_soft))

参数说明：
- δ(t): 前轮转向角（弧度）
- ψe(t): 航向误差 = 路径航向 - 车辆航向（弧度）
- e(t): 横向误差，前轴中心到路径的距离（米）
- k: 横向误差增益（推荐：1.0-3.0）
- k_soft: 软化因子，防止低速时增益过大（推荐：0.5-2.0）
- v(t): 车速（m/s）

调优建议：
- 高速震荡 → 减小k值
- 修正太慢 → 增大k值
- 低速转向过猛 → 增大k_soft值
"""

import math
import numpy as np


class LateralStanleyController:
    """
    Stanley横向控制器

    基于Stanley方法的路径跟踪控制器，
    2005年DARPA挑战赛获胜车辆Stanley使用的算法
    """

    def __init__(self, k=2.0, k_soft=1.0, output_limits=(-0.4, 0.4)):
        """
        初始化Stanley控制器

        Args:
            k (float): 横向误差增益
                      推荐值：
                      - 低速(<80km/h): 1.0-1.5
                      - 中速(80-100km/h): 1.5-2.0
                      - 高速(>100km/h): 2.0-3.0
            k_soft (float): 软化因子，防止低速时增益过大
                           推荐值：0.5-2.0（通常设为1.0）
            output_limits (tuple): 转向角输出限制（弧度）
                                  默认±0.4弧度（约±23度）
        """
        self.k = k
        self.k_soft = k_soft
        self.output_limits = output_limits

        # 速度限制（防止除零）
        self.min_speed = 0.5  # m/s

        # 调试信息
        self.last_heading_error = 0.0
        self.last_crosstrack_error = 0.0
        self.last_velocity = 0.0
        self.last_heading_term = 0.0
        self.last_crosstrack_term = 0.0

    def reset(self):
        """重置控制器状态（调试用）"""
        self.last_heading_error = 0.0
        self.last_crosstrack_error = 0.0
        self.last_velocity = 0.0
        self.last_heading_term = 0.0
        self.last_crosstrack_term = 0.0

    def update(self, heading_error, cross_track_error, velocity, dt=None):
        """
        计算Stanley转向控制输出

        Args:
            heading_error (float): 航向误差（弧度）
                                  计算方式：路径航向 - 车辆航向
                                  正值表示需要逆时针转向（向左）
            cross_track_error (float): 横向误差（米）
                                      前轴中心到车道中心的距离
                                      符号约定：车偏左为负，车偏右为正
            velocity (float): 当前车速（m/s）
            dt (float): 时间步长（秒），Stanley算法不使用，保留用于接口兼容

        Returns:
            float: 转向角输出（弧度）
                  负值表示向左转，正值表示向右转
        """
        # 速度限制（防止除零或负速度）
        v = max(abs(velocity), self.min_speed)

        # Stanley控制律
        # 第一项：航向误差校正
        # 直接补偿航向偏差，使车辆朝向与路径切线对齐
        heading_term = heading_error

        # 第二项：横向误差校正（速度自适应）
        # arctan函数将输出限制在(-π/2, π/2)，避免过度转向
        # 分母中的k_soft防止低速时增益过大导致震荡
        crosstrack_term = math.atan(self.k * cross_track_error / (v + self.k_soft))

        # 总转向角
        steering_angle = heading_term + crosstrack_term

        # 输出限幅（确保不超过车辆物理限制）
        steering_angle = np.clip(steering_angle, self.output_limits[0], self.output_limits[1])

        # 保存调试信息
        self.last_heading_error = heading_error
        self.last_crosstrack_error = cross_track_error
        self.last_velocity = v
        self.last_heading_term = heading_term
        self.last_crosstrack_term = crosstrack_term

        return steering_angle

    def set_gains(self, k=None, k_soft=None):
        """
        动态调整Stanley增益参数

        Args:
            k (float): 横向误差增益
            k_soft (float): 软化因子
        """
        if k is not None:
            self.k = k
            print(f"Stanley控制器：k增益已更新为 {k}")
        if k_soft is not None:
            self.k_soft = k_soft
            print(f"Stanley控制器：k_soft已更新为 {k_soft}")

    def get_state(self):
        """
        获取控制器状态（调试用）

        Returns:
            dict: 控制器参数和最近一次计算的详细信息
        """
        return {
            'k': self.k,
            'k_soft': self.k_soft,
            'min_speed': self.min_speed,
            'output_limits': self.output_limits,
            'last_heading_error_deg': math.degrees(self.last_heading_error),
            'last_crosstrack_error_m': self.last_crosstrack_error,
            'last_velocity_ms': self.last_velocity,
            'last_heading_term_deg': math.degrees(self.last_heading_term),
            'last_crosstrack_term_deg': math.degrees(self.last_crosstrack_term)
        }

    def print_debug_info(self):
        """打印调试信息"""
        print(f"\n=== Stanley控制器调试信息 ===")
        print(f"增益参数：k={self.k:.2f}, k_soft={self.k_soft:.2f}")
        print(f"航向误差：{math.degrees(self.last_heading_error):.2f}°")
        print(f"横向误差：{self.last_crosstrack_error:.3f}m")
        print(f"当前车速：{self.last_velocity:.2f}m/s ({self.last_velocity*3.6:.1f}km/h)")
        print(f"航向修正项：{math.degrees(self.last_heading_term):.2f}°")
        print(f"横向修正项：{math.degrees(self.last_crosstrack_term):.2f}°")
        print(f"总转向角：{math.degrees(self.last_heading_term + self.last_crosstrack_term):.2f}°")
        print(f"================================\n")
