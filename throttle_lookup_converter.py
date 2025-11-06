#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于实验标定数据的油门-扭矩查找表转换器

使用双变量标定数据（速度×油门）实现：
1. 正向查找：(速度, 油门) → 发动机扭矩
2. 反向查找：(速度, 期望扭矩) → 油门值
"""

import json
import numpy as np
from scipy.interpolate import griddata, interp1d
import sys
import io

# 设置控制台输出为UTF-8（Windows）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class ThrottleLookupConverter:
    """
    基于实验标定数据的油门-扭矩转换器

    使用2D查找表和插值实现双向转换：
    - 正向：(当前速度, 油门值) → 发动机扭矩
    - 反向：(当前速度, 期望扭矩) → 油门值
    """

    def __init__(self, calibration_file='throttle_torque_map.json'):
        """
        初始化转换器

        参数:
        calibration_file - 标定数据文件路径
        """
        self.calibration_file = calibration_file
        self.calibration_data = None

        # 查找表数据结构
        self.speed_grid = None      # 速度网格点
        self.throttle_grid = None   # 油门网格点
        self.torque_table = None    # 扭矩查找表 [speed_idx, throttle_idx]

        # 速度范围
        self.min_speed = None
        self.max_speed = None
        self.min_throttle = None
        self.max_throttle = None

        # 加载标定数据
        self._load_calibration_data()

        # 构建查找表
        self._build_lookup_table()

        print(f"\n{'='*60}")
        print(f"ThrottleLookupConverter 初始化完成")
        print(f"{'='*60}")
        print(f"标定数据文件: {calibration_file}")
        print(f"数据点数: {len(self.calibration_data)}")
        print(f"速度范围: {self.min_speed:.1f} - {self.max_speed:.1f} km/h")
        print(f"油门范围: {self.min_throttle:.2f} - {self.max_throttle:.2f}")
        print(f"扭矩范围: {self.torque_table.min():.1f} - {self.torque_table.max():.1f} N·m")
        print(f"{'='*60}\n")

    def _load_calibration_data(self):
        """加载标定数据"""
        try:
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.calibration_data = data.get('calibration_data', [])

            if not self.calibration_data:
                raise ValueError("标定数据为空！")

            print(f"✓ 成功加载标定数据: {len(self.calibration_data)} 个数据点")

        except FileNotFoundError:
            print(f"❌ 标定文件未找到: {self.calibration_file}")
            print(f"   请先运行 calibrate_throttle_torque_safe.py 生成标定数据")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            raise

    def _build_lookup_table(self):
        """构建2D查找表"""
        # 提取所有唯一的速度和油门值
        speeds = sorted(set(d['speed_kmh'] for d in self.calibration_data))
        throttles = sorted(set(d['throttle'] for d in self.calibration_data))

        self.speed_grid = np.array(speeds)
        self.throttle_grid = np.array(throttles)

        self.min_speed = self.speed_grid.min()
        self.max_speed = self.speed_grid.max()
        self.min_throttle = self.throttle_grid.min()
        self.max_throttle = self.throttle_grid.max()

        # 创建2D扭矩表
        self.torque_table = np.zeros((len(speeds), len(throttles)))

        # 填充查找表
        for data_point in self.calibration_data:
            speed = data_point['speed_kmh']
            throttle = data_point['throttle']
            torque = data_point['engine_torque_nm']

            # 找到对应的索引
            speed_idx = np.where(self.speed_grid == speed)[0][0]
            throttle_idx = np.where(self.throttle_grid == throttle)[0][0]

            self.torque_table[speed_idx, throttle_idx] = torque

        print(f"✓ 查找表构建完成")
        print(f"  速度网格点: {len(speeds)} 个")
        print(f"  油门网格点: {len(throttles)} 个")
        print(f"  总数据点: {len(speeds) * len(throttles)}")

    def get_torque_from_throttle(self, current_speed_kmh, throttle_value):
        """
        正向查找：给定速度和油门，返回发动机扭矩

        参数:
        current_speed_kmh - 当前车速 (km/h)
        throttle_value - 油门值 (0-1)

        返回:
        发动机扭矩 (N·m)
        """
        # 边界检查
        speed = np.clip(current_speed_kmh, self.min_speed, self.max_speed)
        throttle = np.clip(throttle_value, self.min_throttle, self.max_throttle)

        # 2D线性插值
        # 使用scipy的griddata进行插值
        points = []
        values = []

        for i, s in enumerate(self.speed_grid):
            for j, t in enumerate(self.throttle_grid):
                points.append([s, t])
                values.append(self.torque_table[i, j])

        points = np.array(points)
        values = np.array(values)

        # 插值计算
        torque = griddata(points, values, (speed, throttle), method='linear')

        # 如果超出范围，使用最近邻插值
        if np.isnan(torque):
            torque = griddata(points, values, (speed, throttle), method='nearest')

        return float(torque)

    def get_throttle_from_torque(self, current_speed_kmh, desired_torque_nm):
        """
        反向查找：给定速度和期望扭矩，返回所需油门值

        参数:
        current_speed_kmh - 当前车速 (km/h)
        desired_torque_nm - 期望发动机扭矩 (N·m)

        返回:
        (throttle, brake) - 油门和刹车值 (0-1)
        """
        # 边界检查：速度
        speed = np.clip(current_speed_kmh, self.min_speed, self.max_speed)

        # 找到最接近的速度索引
        speed_idx = np.argmin(np.abs(self.speed_grid - speed))
        actual_speed = self.speed_grid[speed_idx]

        # 获取该速度下的扭矩曲线（扭矩 vs 油门）
        torque_curve = self.torque_table[speed_idx, :]

        # 如果需要精确插值（速度不在网格点上）
        if speed != actual_speed:
            # 找到相邻的两个速度点
            if speed < actual_speed and speed_idx > 0:
                speed_idx_low = speed_idx - 1
                speed_idx_high = speed_idx
            elif speed > actual_speed and speed_idx < len(self.speed_grid) - 1:
                speed_idx_low = speed_idx
                speed_idx_high = speed_idx + 1
            else:
                # 已经在边界，使用单个点
                speed_idx_low = speed_idx_high = speed_idx

            # 线性插值两条扭矩曲线
            speed_low = self.speed_grid[speed_idx_low]
            speed_high = self.speed_grid[speed_idx_high]

            if speed_low != speed_high:
                weight = (speed - speed_low) / (speed_high - speed_low)
                torque_curve = (1 - weight) * self.torque_table[speed_idx_low, :] + \
                               weight * self.torque_table[speed_idx_high, :]
            else:
                torque_curve = self.torque_table[speed_idx_low, :]

        # 在扭矩曲线上查找对应的油门值
        # 注意：扭矩随油门单调递增

        # 获取该速度下的最大和最小扭矩
        min_torque = torque_curve.min()
        max_torque = torque_curve.max()

        # 判断加速还是减速
        if desired_torque_nm >= 0:
            # 加速或保持
            # 限制在可达到的扭矩范围内
            target_torque = np.clip(desired_torque_nm, min_torque, max_torque)

            # 使用1D插值找到对应的油门值
            # 创建插值函数：油门 = f(扭矩)
            # 需要确保扭矩是单调的
            if np.all(np.diff(torque_curve) > 0):
                # 严格单调递增，直接插值
                interp_func = interp1d(torque_curve, self.throttle_grid,
                                      bounds_error=False, fill_value='extrapolate')
                throttle = float(interp_func(target_torque))
            else:
                # 非严格单调，使用最近邻
                idx = np.argmin(np.abs(torque_curve - target_torque))
                throttle = float(self.throttle_grid[idx])

            # 限制油门范围
            throttle = np.clip(throttle, 0.0, 1.0)
            brake = 0.0

        else:
            # 减速（负扭矩 → 刹车）
            # 简单映射：将负扭矩映射到刹车
            # 假设最大制动扭矩约为 -max_torque
            brake = np.clip(abs(desired_torque_nm) / max_torque, 0.0, 1.0)
            throttle = 0.0

        return throttle, brake

    def get_max_torque_at_speed(self, current_speed_kmh):
        """
        获取指定速度下的最大可用扭矩

        参数:
        current_speed_kmh - 当前车速 (km/h)

        返回:
        最大发动机扭矩 (N·m)
        """
        speed = np.clip(current_speed_kmh, self.min_speed, self.max_speed)

        # 找到最接近的速度索引
        speed_idx = np.argmin(np.abs(self.speed_grid - speed))

        # 返回该速度下的最大扭矩（油门=1.0时的扭矩）
        max_torque = self.torque_table[speed_idx, -1]  # 最后一列是油门1.0

        return float(max_torque)

    def validate_calibration(self):
        """
        验证标定数据的质量

        返回:
        验证报告字典
        """
        print(f"\n{'='*60}")
        print(f"标定数据验证")
        print(f"{'='*60}\n")

        issues = []

        # 1. 检查数据完整性
        expected_points = len(self.speed_grid) * len(self.throttle_grid)
        actual_points = len(self.calibration_data)

        print(f"数据完整性:")
        print(f"  期望数据点: {expected_points}")
        print(f"  实际数据点: {actual_points}")

        if actual_points < expected_points:
            issues.append(f"缺失数据点: {expected_points - actual_points} 个")
            print(f"  ⚠️  缺失 {expected_points - actual_points} 个数据点")
        else:
            print(f"  ✓ 数据完整")

        # 2. 检查单调性（油门增加 → 扭矩增加）
        print(f"\n单调性检查:")
        non_monotonic_speeds = []

        for i, speed in enumerate(self.speed_grid):
            torques = self.torque_table[i, :]
            if not np.all(np.diff(torques) >= 0):
                non_monotonic_speeds.append(speed)
                issues.append(f"速度 {speed} km/h 下扭矩非单调")

        if non_monotonic_speeds:
            print(f"  ⚠️  以下速度点的扭矩非单调递增:")
            for s in non_monotonic_speeds:
                print(f"      {s:.1f} km/h")
        else:
            print(f"  ✓ 所有速度点的扭矩单调递增")

        # 3. 检查合理性（扭矩范围）
        print(f"\n扭矩范围检查:")
        print(f"  最小扭矩: {self.torque_table.min():.1f} N·m")
        print(f"  最大扭矩: {self.torque_table.max():.1f} N·m")

        # 检查是否有异常的负扭矩（除了极低油门）
        negative_torques = self.torque_table < 0
        if np.any(negative_torques):
            count = np.sum(negative_torques)
            print(f"  ⚠️  发现 {count} 个负扭矩数据点")
            issues.append(f"发现 {count} 个负扭矩数据点")
        else:
            print(f"  ✓ 无异常负扭矩")

        # 4. 检查数据平滑性（相邻点差异）
        print(f"\n平滑性检查:")
        max_speed_jump = 0
        max_throttle_jump = 0

        for i in range(len(self.speed_grid) - 1):
            for j in range(len(self.throttle_grid)):
                jump = abs(self.torque_table[i+1, j] - self.torque_table[i, j])
                max_speed_jump = max(max_speed_jump, jump)

        for i in range(len(self.speed_grid)):
            for j in range(len(self.throttle_grid) - 1):
                jump = abs(self.torque_table[i, j+1] - self.torque_table[i, j])
                max_throttle_jump = max(max_throttle_jump, jump)

        print(f"  速度方向最大跳变: {max_speed_jump:.1f} N·m")
        print(f"  油门方向最大跳变: {max_throttle_jump:.1f} N·m")

        if max_speed_jump > 200 or max_throttle_jump > 200:
            issues.append(f"数据跳变过大（速度:{max_speed_jump:.1f}, 油门:{max_throttle_jump:.1f}）")
            print(f"  ⚠️  数据跳变较大，可能存在测量误差")
        else:
            print(f"  ✓ 数据平滑")

        # 总结
        print(f"\n{'='*60}")
        if issues:
            print(f"⚠️  发现 {len(issues)} 个问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"✓ 标定数据验证通过，质量良好")
        print(f"{'='*60}\n")

        return {
            'total_issues': len(issues),
            'issues': issues,
            'expected_points': expected_points,
            'actual_points': actual_points,
            'torque_range': (float(self.torque_table.min()), float(self.torque_table.max())),
            'non_monotonic_speeds': non_monotonic_speeds
        }

    def test_conversion(self):
        """测试转换功能"""
        print(f"\n{'='*60}")
        print(f"转换功能测试")
        print(f"{'='*60}\n")

        # 测试用例
        test_cases = [
            # (速度, 期望扭矩, 描述)
            (30, 400, "中速中等扭矩"),
            (30, 600, "中速大扭矩"),
            (50, 300, "高速中等扭矩"),
            (10, 500, "低速大扭矩"),
            (30, 0, "零扭矩"),
            (30, -100, "负扭矩（制动）")
        ]

        print(f"{'速度':<10} {'期望扭矩':<15} {'计算油门':<12} {'计算刹车':<12} {'实际扭矩':<15} {'描述':<20}")
        print(f"{'-'*90}")

        for speed, desired_torque, description in test_cases:
            # 反向查找：扭矩 → 油门
            throttle, brake = self.get_throttle_from_torque(speed, desired_torque)

            # 正向验证：油门 → 扭矩
            if throttle > 0:
                actual_torque = self.get_torque_from_throttle(speed, throttle)
            else:
                actual_torque = 0.0

            print(f"{speed:<10.1f} {desired_torque:<15.1f} {throttle:<12.3f} {brake:<12.3f} "
                  f"{actual_torque:<15.1f} {description:<20}")

        print(f"\n{'='*60}\n")


# ============================================================================
# 测试代码
# ============================================================================

def test_converter():
    """测试转换器"""
    try:
        # 创建转换器（需要先有标定文件）
        converter = ThrottleLookupConverter('throttle_torque_map.json')

        # 验证标定数据
        validation_result = converter.validate_calibration()

        # 测试转换
        converter.test_conversion()

        # 显示一些关键速度点的最大扭矩
        print(f"各速度下的最大扭矩:")
        for speed in [10, 20, 30, 40, 50, 60]:
            max_torque = converter.get_max_torque_at_speed(speed)
            print(f"  {speed:3.0f} km/h: {max_torque:6.1f} N·m")

    except FileNotFoundError:
        print(f"\n⚠️  尚未生成标定文件")
        print(f"   请先运行 calibrate_throttle_torque_safe.py 进行标定")
        print(f"   标定完成后会生成 throttle_torque_map.json 文件")


if __name__ == "__main__":
    test_converter()
