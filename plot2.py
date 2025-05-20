import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_speed_comparison(csv_file_path='speed_data.csv'):
    """
    图1：绘制两车速度随时间变化对比图
    """
    try:
        data = pd.read_csv(csv_file_path)

        # 查找对应的列
        time_col = [col for col in data.columns if 'Time' in col or 'time' in col][0]
        ego_speed_col = [col for col in data.columns if 'Ego_Speed' in col][0]
        target_speed_col = [col for col in data.columns if 'Target_Speed' in col][0]

        time_data = data[time_col]
        ego_speed = data[ego_speed_col]
        target_speed = data[target_speed_col]

        # 创建图像
        plt.figure(figsize=(12, 8))
        plt.plot(time_data, ego_speed, 'b-', linewidth=2.5, label='自车速度 (Ego Vehicle)', alpha=0.8)
        plt.plot(time_data, target_speed, 'r-', linewidth=2.5, label='目标车速度 (Target Vehicle)', alpha=0.8)

        plt.xlabel('时间 (秒)', fontsize=14)
        plt.ylabel('速度 (km/h)', fontsize=14)
        plt.title('车辆速度随时间变化对比', fontsize=16, fontweight='bold')
        plt.legend(fontsize=13)
        plt.grid(True, alpha=0.3)

        # 添加速度统计信息
        ego_avg = np.mean(ego_speed)
        target_avg = np.mean(target_speed)
        ego_max = np.max(ego_speed)
        ego_min = np.min(ego_speed)
        target_max = np.max(target_speed)
        target_min = np.min(target_speed)

        stats_text = f'''速度统计信息:
自车平均速度: {ego_avg:.1f} km/h
自车速度范围: {ego_min:.1f} - {ego_max:.1f} km/h
目标车平均速度: {target_avg:.1f} km/h
目标车速度范围: {target_min:.1f} - {target_max:.1f} km/h'''

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig('speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ 速度对比图绘制完成")

    except Exception as e:
        print(f"绘制速度对比图时出错: {e}")


def plot_following_distance_comparison(csv_file_path='speed_data.csv'):
    """
    图2：绘制实际跟车距离 vs 期望跟车距离对比图 (纵坐标从零开始)
    """
    try:
        data = pd.read_csv(csv_file_path)

        # 查找对应的列
        time_col = [col for col in data.columns if 'Time' in col][0]

        # 寻找距离列
        actual_distance_col = None
        desired_distance_col = None
        ego_speed_col = None

        for col in data.columns:
            if 'Actual_Distance' in col:
                actual_distance_col = col
            elif 'Desired_Distance' in col:
                desired_distance_col = col
            elif 'Ego_Speed' in col:
                ego_speed_col = col
            elif actual_distance_col is None and 'Distance' in col and 'Lane' not in col:
                actual_distance_col = col

        # 如果没有期望距离，计算它
        if desired_distance_col is None and ego_speed_col is not None:
            ego_speed = data[ego_speed_col]
            ego_speed_ms = ego_speed / 3.6
            desired_distance = 2.0 + ego_speed_ms * 2.0
        else:
            desired_distance = data[desired_distance_col]

        time_data = data[time_col]
        actual_distance = data[actual_distance_col]

        # 创建图像
        plt.figure(figsize=(12, 8))

        # 绘制距离线
        plt.plot(time_data, actual_distance, 'g-', linewidth=2.5, label='实际跟车距离', alpha=0.9)
        plt.plot(time_data, desired_distance, 'orange', linewidth=2.5, linestyle='--',
                 label='期望跟车距离', alpha=0.9)

        # 填充误差区域
        plt.fill_between(time_data, actual_distance, desired_distance,
                         alpha=0.2, color='red', label='距离误差')

        # 设置纵坐标从零开始
        all_distances = np.concatenate([actual_distance, desired_distance])
        max_distance = np.max(all_distances)

        # 留出一些上边距，让图像更美观
        y_max = max_distance * 1.1
        plt.ylim(0, y_max)

        plt.xlabel('时间 (秒)', fontsize=14)
        plt.ylabel('距离 (米)', fontsize=14)
        plt.title('实际跟车距离 vs 期望跟车距离对比', fontsize=16, fontweight='bold')
        plt.legend(fontsize=13)
        plt.grid(True, alpha=0.3)

        # 计算统计信息
        distance_error = actual_distance - desired_distance
        mae = np.mean(np.abs(distance_error))
        rmse = np.sqrt(np.mean(distance_error ** 2))
        max_error = np.max(np.abs(distance_error))

        # 计算跟车性能指标
        tolerance = 0.2  # 20%容差
        within_tolerance = np.abs(distance_error) <= (desired_distance * tolerance)
        performance_percentage = np.sum(within_tolerance) / len(within_tolerance) * 100

        # 统计信息
        stats_text = f'''跟车性能统计:
平均绝对误差: {mae:.2f} m
均方根误差: {rmse:.2f} m
最大误差: {max_error:.2f} m
±20%容差内时间: {performance_percentage:.1f}%
平均实际距离: {np.mean(actual_distance):.1f} m
平均期望距离: {np.mean(desired_distance):.1f} m'''

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig('following_distance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ 跟车距离对比图绘制完成 (纵坐标从零开始)")

        # 返回统计信息以供后续使用
        return {
            'mae': mae,
            'rmse': rmse,
            'max_error': max_error,
            'performance_percentage': performance_percentage
        }

    except Exception as e:
        print(f"绘制跟车距离对比图时出错: {e}")
        return None


def plot_lane_offset(csv_file_path='speed_data.csv'):
    """
    图3：绘制车道中心偏移量随时间变化图
    """
    try:
        data = pd.read_csv(csv_file_path)

        # 查找对应的列
        time_col = [col for col in data.columns if 'Time' in col][0]
        lane_offset_col = [col for col in data.columns if 'Lane_Offset' in col or 'offset' in col][0]

        time_data = data[time_col]
        lane_offset = data[lane_offset_col]

        # 创建图像
        plt.figure(figsize=(12, 8))
        plt.plot(time_data, lane_offset, 'purple', linewidth=2.5, label='车道中心偏移', alpha=0.8)

        # 添加车道边界参考线
        lane_width = 3.5  # 典型车道宽度的一半
        plt.axhline(y=lane_width / 2, color='red', linestyle='--', alpha=0.7,
                    linewidth=2, label=f'车道边界 (+{lane_width / 2:.1f}m)')
        plt.axhline(y=-lane_width / 2, color='red', linestyle='--', alpha=0.7,
                    linewidth=2, label=f'车道边界 (-{lane_width / 2:.1f}m)')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5,
                    linewidth=1.5, label='车道中心线')

        plt.xlabel('时间 (秒)', fontsize=14)
        plt.ylabel('偏移量 (米)', fontsize=14)
        plt.title('车道中心偏移量随时间变化', fontsize=16, fontweight='bold')
        plt.legend(fontsize=13)
        plt.grid(True, alpha=0.3)

        # 添加偏移统计信息
        offset_avg = np.mean(np.abs(lane_offset))
        offset_max = np.max(np.abs(lane_offset))
        offset_std = np.std(lane_offset)

        # 计算车道保持性能
        within_lane = np.abs(lane_offset) <= lane_width / 2
        lane_keeping_percentage = np.sum(within_lane) / len(within_lane) * 100

        stats_text = f'''车道保持统计:
平均偏移: {offset_avg:.3f} m
最大偏移: {offset_max:.3f} m
偏移标准差: {offset_std:.3f} m
车道内行驶时间: {lane_keeping_percentage:.1f}%
偏移范围: {np.min(lane_offset):.3f} - {np.max(lane_offset):.3f} m'''

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.savefig('lane_offset.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ 车道偏移图绘制完成")

    except Exception as e:
        print(f"绘制车道偏移图时出错: {e}")


def plot_all_individual_graphs(csv_file_path='speed_data.csv'):
    """
    主函数：分别绘制三个独立的图像
    """
    print(f"开始分析CSV文件: {csv_file_path}")
    print("=" * 50)

    # 检查文件是否存在
    try:
        data = pd.read_csv(csv_file_path)
        print(f"✓ 成功读取数据，共 {len(data)} 行数据")
        print(f"✓ 数据列: {list(data.columns)}")
        print("=" * 50)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {csv_file_path}")
        return
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return

    # 分别绘制三个图
    print("开始绘制图像...")

    # 图1：速度对比
    plot_speed_comparison(csv_file_path)

    # 图2：跟车距离对比
    stats = plot_following_distance_comparison(csv_file_path)

    # 图3：车道偏移
    plot_lane_offset(csv_file_path)

    print("=" * 50)
    print("✅ 所有图像绘制完成！")
    print("生成的图像文件:")
    print("  - speed_comparison.png")
    print("  - following_distance_comparison.png")
    print("  - lane_offset.png")

    # 如果有跟车性能统计，打印摘要
    if stats:
        print("\n📊 跟车性能摘要:")
        print(f"  - 平均绝对误差: {stats['mae']:.2f} m")
        print(f"  - 均方根误差: {stats['rmse']:.2f} m")
        print(f"  - 最大误差: {stats['max_error']:.2f} m")
        print(f"  - 跟车精度: {stats['performance_percentage']:.1f}%")


def generate_comprehensive_report(csv_file_path='speed_data.csv'):
    """
    生成综合性能报告
    """
    try:
        data = pd.read_csv(csv_file_path)

        # 提取基本信息
        time_data = data[[col for col in data.columns if 'Time' in col][0]]
        duration = time_data.iloc[-1] - time_data.iloc[0]

        print("\n" + "=" * 60)
        print("🚗 ACC系统性能综合报告")
        print("=" * 60)
        print(f"测试时长: {duration:.1f} 秒")
        print(f"数据点数: {len(data)} 个")
        print(f"平均采样率: {len(data) / duration:.1f} Hz")
        print("=" * 60)

        # 调用各个绘图函数并生成报告
        plot_all_individual_graphs(csv_file_path)

    except Exception as e:
        print(f"生成报告时出错: {e}")


# 主函数
if __name__ == "__main__":
    # 设置CSV文件路径
    csv_file_path = 'speed_data.csv'

    # 生成所有图像和报告
    generate_comprehensive_report(csv_file_path)

    # 如果只想绘制特定图像，可以直接调用：
    # plot_speed_comparison(csv_file_path)
    # plot_following_distance_comparison(csv_file_path)
    # plot_lane_offset(csv_file_path)