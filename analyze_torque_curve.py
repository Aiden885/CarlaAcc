#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析CARLA扭矩曲线的详细信息
检查是否可以获取更多数据点或通过实验标定
"""

import carla
import sys
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置控制台输出为UTF-8（Windows）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def analyze_torque_curve_source(vehicle):
    """分析扭矩曲线的来源和详细信息"""
    print("\n" + "="*60)
    print("CARLA扭矩曲线详细分析")
    print("="*60)

    physics = vehicle.get_physics_control()

    # 提取扭矩曲线
    torque_curve = physics.torque_curve
    print(f"\n【从CARLA API获取的扭矩曲线】")
    print(f"数据点数量: {len(torque_curve)}")
    print(f"\nRPM值分布:")
    rpm_values = [p.x for p in torque_curve]
    print(f"  最小RPM: {min(rpm_values)}")
    print(f"  最大RPM: {max(rpm_values)}")
    print(f"  RPM间隔: {np.diff(sorted(rpm_values))}")

    print(f"\n完整数据点:")
    print(f"{'索引':<6} {'RPM':<10} {'扭矩(N·m)':<12}")
    print("-" * 30)
    for i, point in enumerate(torque_curve):
        print(f"{i:<6} {point.x:<10.1f} {point.y:<12.1f}")

    # 检查是否有其他相关API
    print(f"\n【检查VehiclePhysicsControl的其他属性】")
    attrs = dir(physics)
    relevant_attrs = [a for a in attrs if not a.startswith('_') and
                     ('torque' in a.lower() or 'curve' in a.lower() or
                      'rpm' in a.lower() or 'power' in a.lower())]
    print(f"扭矩/功率相关属性: {relevant_attrs}")

    # 尝试访问这些属性
    for attr in relevant_attrs:
        try:
            value = getattr(physics, attr)
            print(f"  {attr}: {value}")
        except Exception as e:
            print(f"  {attr}: 无法访问 ({e})")

    return torque_curve


def visualize_torque_curve(torque_curve):
    """可视化扭矩曲线及插值效果"""
    print(f"\n【可视化扭矩曲线】")

    # 提取数据
    rpm_points = np.array([p.x for p in torque_curve])
    torque_points = np.array([p.y for p in torque_curve])

    # 创建密集的RPM范围用于插值
    rpm_dense = np.linspace(min(rpm_points), max(rpm_points), 1000)

    # 线性插值
    torque_linear = np.interp(rpm_dense, rpm_points, torque_points)

    # 样条插值（如果scipy可用）
    try:
        from scipy.interpolate import interp1d

        # 三次样条插值
        spline_cubic = interp1d(rpm_points, torque_points, kind='cubic')
        torque_cubic = spline_cubic(rpm_dense)

        # 二次样条插值
        spline_quadratic = interp1d(rpm_points, torque_points, kind='quadratic')
        torque_quadratic = spline_quadratic(rpm_dense)

        has_scipy = True
    except ImportError:
        print("  提示: 安装scipy可以使用更高级的插值方法")
        has_scipy = False

    # 绘图
    plt.figure(figsize=(12, 8))

    # 子图1: 扭矩曲线
    plt.subplot(2, 1, 1)
    plt.plot(rpm_points, torque_points, 'ro', markersize=10, label='CARLA原始数据点', zorder=3)
    plt.plot(rpm_dense, torque_linear, 'b-', linewidth=2, label='线性插值', alpha=0.7)

    if has_scipy:
        plt.plot(rpm_dense, torque_cubic, 'g--', linewidth=2, label='三次样条插值', alpha=0.7)
        plt.plot(rpm_dense, torque_quadratic, 'm:', linewidth=2, label='二次样条插值', alpha=0.7)

    plt.xlabel('发动机转速 (RPM)', fontsize=12)
    plt.ylabel('扭矩 (N·m)', fontsize=12)
    plt.title('发动机扭矩曲线对比', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # 子图2: 插值误差分析（如果有样条）
    if has_scipy:
        plt.subplot(2, 1, 2)
        diff_cubic = torque_cubic - torque_linear
        diff_quadratic = torque_quadratic - torque_linear

        plt.plot(rpm_dense, diff_cubic, 'g-', linewidth=2, label='三次样条 - 线性')
        plt.plot(rpm_dense, diff_quadratic, 'm-', linewidth=2, label='二次样条 - 线性')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        plt.xlabel('发动机转速 (RPM)', fontsize=12)
        plt.ylabel('扭矩差异 (N·m)', fontsize=12)
        plt.title('不同插值方法的差异', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        print(f"  最大插值差异（三次样条 vs 线性）: {np.max(np.abs(diff_cubic)):.2f} N·m")
        print(f"  最大插值差异（二次样条 vs 线性）: {np.max(np.abs(diff_quadratic)):.2f} N·m")

    plt.tight_layout()
    plt.savefig('torque_curve_analysis.png', dpi=150)
    print(f"  图像已保存: torque_curve_analysis.png")


def check_rpm_density_requirement(torque_curve):
    """检查当前数据点密度是否足够"""
    print(f"\n【数据点密度评估】")

    rpm_points = np.array([p.x for p in torque_curve])
    torque_points = np.array([p.y for p in torque_curve])

    # 计算相邻点之间的斜率变化
    rpm_sorted_idx = np.argsort(rpm_points)
    rpm_sorted = rpm_points[rpm_sorted_idx]
    torque_sorted = torque_points[rpm_sorted_idx]

    print(f"\n各段斜率分析:")
    print(f"{'起始RPM':<12} {'结束RPM':<12} {'RPM跨度':<12} {'扭矩变化':<12} {'平均斜率':<15}")
    print("-" * 75)

    for i in range(len(rpm_sorted) - 1):
        rpm_span = rpm_sorted[i+1] - rpm_sorted[i]
        torque_change = torque_sorted[i+1] - torque_sorted[i]
        slope = torque_change / rpm_span if rpm_span > 0 else 0

        print(f"{rpm_sorted[i]:<12.0f} {rpm_sorted[i+1]:<12.0f} "
              f"{rpm_span:<12.0f} {torque_change:<12.1f} {slope:<15.4f}")

    # 评估
    max_span = np.max(np.diff(rpm_sorted))
    print(f"\n最大RPM跨度: {max_span:.0f} RPM")

    if max_span > 2000:
        print("  ⚠️  警告: 存在跨度超过2000 RPM的区间，建议增加数据点")
    elif max_span > 1000:
        print("  ⚠️  注意: 存在跨度超过1000 RPM的区间，线性插值可能有较大误差")
    else:
        print("  ✓  数据点密度较好")


def suggest_additional_points(torque_curve):
    """建议在哪些RPM增加测量点"""
    print(f"\n【建议增加的测量点】")

    rpm_points = np.array([p.x for p in torque_curve])
    rpm_sorted = np.sort(rpm_points)

    print(f"\n推荐在以下RPM进行实验测量（在大跨度区间的中点）:")

    for i in range(len(rpm_sorted) - 1):
        rpm_span = rpm_sorted[i+1] - rpm_sorted[i]
        if rpm_span > 1000:
            # 计算对应的车速
            # 假设传动比9.204，车轮半径0.37m
            mid_rpm = (rpm_sorted[i] + rpm_sorted[i+1]) / 2
            wheel_rpm = mid_rpm / 9.204
            wheel_angular_vel = wheel_rpm * 2 * np.pi / 60
            speed_ms = wheel_angular_vel * 0.37
            speed_kmh = speed_ms * 3.6

            print(f"  RPM区间 [{rpm_sorted[i]:.0f}, {rpm_sorted[i+1]:.0f}], "
                  f"跨度 {rpm_span:.0f} RPM")
            print(f"    → 建议测量点: {mid_rpm:.0f} RPM (对应车速约 {speed_kmh:.1f} km/h)")


def main():
    # 连接CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 生成测试车辆
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    vehicle.set_simulate_physics(True)

    for _ in range(20):
        world.tick()

    try:
        # 分析扭矩曲线
        torque_curve = analyze_torque_curve_source(vehicle)

        # 可视化
        visualize_torque_curve(torque_curve)

        # 密度评估
        check_rpm_density_requirement(torque_curve)

        # 建议
        suggest_additional_points(torque_curve)

        print("\n" + "="*60)
        print("分析完成")
        print("="*60)

    finally:
        vehicle.destroy()


if __name__ == "__main__":
    main()
