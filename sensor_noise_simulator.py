"""
传感器噪音模拟器
噪音频率匹配滤波器截止频率
"""
import numpy as np
from typing import Dict, Optional


class SensorNoiseSimulator:
    """
    传感器噪音模拟器

    设计目标：
    - 产生高频噪音（5-10 Hz）来测试Simulink滤波器（fc=1 Hz, 0.64 Hz）
    - 70%高频成分会被滤波器抑制
    - 30%低频成分会穿过滤波器（真实信号）
    """

    def __init__(
        self,
        enable_noise: bool = False,
        noise_level: str = 'medium',  # 'low', 'medium', 'high'
        random_seed: Optional[int] = None
    ):
        """
        初始化噪音模拟器

        Args:
            enable_noise: 是否启用噪音
            noise_level: 噪音等级 ('low', 'medium', 'high')
            random_seed: 随机种子
        """
        self.enable_noise = enable_noise
        self.noise_level = noise_level

        if random_seed is not None:
            np.random.seed(random_seed)

        # 根据噪音等级设置参数
        self._setup_noise_params()

        # 低频噪音状态（用于生成低频成分）
        self._low_freq_distance_noise = 0.0
        self._low_freq_speed_noise = 0.0
        self._low_freq_lane_noise = 0.0

        # 低频噪音更新系数（模拟低频漂移）
        self._low_freq_alpha = 0.05  # 对应约0.16 Hz的低频成分

        # 统计
        self.stats = {
            'calls': 0,
            'distance_noise_rms': [],
            'speed_noise_rms': [],
        }

    def _setup_noise_params(self):
        """根据噪音等级设置参数"""
        if self.noise_level == 'low':
            # 低噪音：理想传感器
            self.distance_std = 0.1      # ±0.1m
            self.speed_std = 0.1         # ±0.1 km/h
            self.lane_offset_std = 0.03  # ±0.03m

        elif self.noise_level == 'medium':
            # 中等噪音：匹配滤波器设计（推荐）
            self.distance_std = 0.25     # ±0.25m
            self.speed_std = 0.15        # ±0.15 km/h
            self.lane_offset_std = 0.05  # ±0.05m

        elif self.noise_level == 'high':
            # 高噪音：恶劣条件
            self.distance_std = 0.4      # ±0.4m
            self.speed_std = 0.3         # ±0.3 km/h
            self.lane_offset_std = 0.08  # ±0.08m
        else:
            raise ValueError(f"Unknown noise level: {self.noise_level}")

        # 高频/低频噪音比例（固定）
        self.high_freq_ratio = 0.7   # 70%高频（会被滤波器抑制）
        self.low_freq_ratio = 0.3    # 30%低频（会穿过滤波器）

    def add_distance_noise(self, true_distance: float) -> float:
        """
        添加距离测量噪音

        噪音组成：
        - 70%高频白噪音（5-10 Hz）→ 会被velocity滤波器抑制
        - 30%低频漂移（0.1-0.5 Hz）→ 会穿过滤波器
        """
        if not self.enable_noise:
            return true_distance

        # 高频噪音（每帧独立）
        high_freq_noise = np.random.normal(0, self.distance_std * self.high_freq_ratio)

        # 低频噪音（缓慢变化）
        self._low_freq_distance_noise = (
            self._low_freq_alpha * np.random.normal(0, self.distance_std * self.low_freq_ratio) +
            (1 - self._low_freq_alpha) * self._low_freq_distance_noise
        )

        # 总噪音 = 高频 + 低频
        total_noise = high_freq_noise + self._low_freq_distance_noise

        # 记录统计
        self.stats['distance_noise_rms'].append(total_noise**2)

        return max(0.0, true_distance + total_noise)

    def add_speed_noise(self, true_speed: float) -> float:
        """添加速度测量噪音"""
        if not self.enable_noise:
            return true_speed

        # 高频噪音
        high_freq_noise = np.random.normal(0, self.speed_std * self.high_freq_ratio)

        # 低频噪音
        self._low_freq_speed_noise = (
            self._low_freq_alpha * np.random.normal(0, self.speed_std * self.low_freq_ratio) +
            (1 - self._low_freq_alpha) * self._low_freq_speed_noise
        )

        total_noise = high_freq_noise + self._low_freq_speed_noise

        # 记录统计
        self.stats['speed_noise_rms'].append(total_noise**2)

        return max(0.0, true_speed + total_noise)

    def add_lane_offset_noise(self, true_offset: float) -> float:
        """添加车道偏移噪音"""
        if not self.enable_noise:
            return true_offset

        # 高频噪音
        high_freq_noise = np.random.normal(0, self.lane_offset_std * self.high_freq_ratio)

        # 低频噪音
        self._low_freq_lane_noise = (
            self._low_freq_alpha * np.random.normal(0, self.lane_offset_std * self.low_freq_ratio) +
            (1 - self._low_freq_alpha) * self._low_freq_lane_noise
        )

        total_noise = high_freq_noise + self._low_freq_lane_noise

        return true_offset + total_noise

    def get_statistics(self) -> Dict:
        """获取噪音统计"""
        stats = {}

        if self.stats['distance_noise_rms']:
            distance_rms = np.sqrt(np.mean(self.stats['distance_noise_rms']))
            stats['distance_noise_rms'] = distance_rms
            stats['distance_noise_std_actual'] = distance_rms  # RMS ≈ STD for zero-mean
        else:
            stats['distance_noise_rms'] = 0.0
            stats['distance_noise_std_actual'] = 0.0

        if self.stats['speed_noise_rms']:
            speed_rms = np.sqrt(np.mean(self.stats['speed_noise_rms']))
            stats['speed_noise_rms'] = speed_rms
            stats['speed_noise_std_actual'] = speed_rms
        else:
            stats['speed_noise_rms'] = 0.0
            stats['speed_noise_std_actual'] = 0.0

        stats['distance_noise_std_config'] = self.distance_std
        stats['speed_noise_std_config'] = self.speed_std
        stats['calls'] = self.stats['calls']

        return stats

    def print_config(self):
        """打印噪音配置"""
        print("\n" + "=" * 70)
        print("📡 传感器噪音模拟器配置")
        print("=" * 70)
        print(f"噪音状态: {'启用' if self.enable_noise else '禁用'}")
        print(f"噪音等级: {self.noise_level}")
        print()
        print("【噪音参数】")
        print(f"  距离测量标准差:   {self.distance_std:.3f} m")
        print(f"  速度测量标准差:   {self.speed_std:.3f} km/h")
        print(f"  车道偏移标准差:   {self.lane_offset_std:.3f} m")
        print()
        print("【噪音频率成分】")
        print(f"  高频成分 (5-10 Hz):  {self.high_freq_ratio*100:.0f}% → 会被滤波器抑制")
        print(f"  低频成分 (0.1-0.5 Hz): {self.low_freq_ratio*100:.0f}% → 会穿过滤波器")
        print()
        print("【对应的误差噪音估算】")
        # 假设speed=15m/s, time_gap=distance/speed
        error_noise_std = self.distance_std / 15.0  # 简化估算
        print(f"  误差噪音 (error):     ~{error_noise_std:.3f} s")

        # 数值微分后的velocity噪音（无滤波）
        dt = 0.05
        velocity_noise_raw = error_noise_std / dt
        print(f"  Velocity噪音 (无滤波): ~{velocity_noise_raw:.1f} s/s")

        # 滤波后的velocity噪音（alpha=0.3）
        alpha_v = 0.3
        velocity_noise_filtered = velocity_noise_raw * alpha_v
        print(f"  Velocity噪音 (滤波后): ~{velocity_noise_filtered:.1f} s/s (抑制{(1-alpha_v)*100:.0f}%)")

        # acceleration噪音（滤波后，alpha=0.2）
        alpha_a = 0.2
        accel_noise_filtered = (velocity_noise_filtered / dt) * alpha_a
        print(f"  Acceleration噪音 (滤波后): ~{accel_noise_filtered:.1f} s/s²")
        print("=" * 70)