#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版两模式控制数据分析脚本
避免Windows控制台显示问题
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def load_csv_data(csv_file='speed_data_integrated.csv'):
    """加载CSV数据，支持多种编码"""
    if not os.path.exists(csv_file):
        print(f"[ERROR] 数据文件不存在: {csv_file}")
        return None
    
    # 尝试不同编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            data = pd.read_csv(csv_file, encoding=encoding)
            print(f"[OK] 使用 {encoding} 编码加载数据成功")
            print(f"[INFO] 数据行数: {len(data)}")
            print(f"[INFO] 数据列数: {len(data.columns)}")
            return data
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[ERROR] 使用 {encoding} 编码失败: {e}")
            continue
    
    # 最后尝试忽略错误
    try:
        data = pd.read_csv(csv_file, encoding='utf-8', errors='ignore')
        print("[WARNING] 使用UTF-8忽略错误模式加载")
        return data
    except Exception as e:
        print(f"[ERROR] 所有方法都失败: {e}")
        return None


def analyze_data(data):
    """分析控制数据"""
    if data is None:
        return
    
    print("\n" + "="*50)
    print("数据分析结果")
    print("="*50)
    
    # 显示列名
    print("\n[INFO] 数据列:")
    for i, col in enumerate(data.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # 基本统计
    if 'Time(s)' in data.columns:
        duration = data['Time(s)'].max()
        print(f"\n[TIME] 数据时长: {duration:.1f} 秒")
        print(f"[TIME] 数据点数: {len(data)} 个")
        print(f"[TIME] 采样频率: {len(data)/duration:.1f} Hz")
    
    # 速度分析
    if 'Ego_Speed(km/h)' in data.columns and 'Target_Speed(km/h)' in data.columns:
        ego_speeds = data['Ego_Speed(km/h)']
        target_speeds = data['Target_Speed(km/h)']
        
        speed_errors = np.abs(ego_speeds - target_speeds)
        
        print(f"\n[SPEED] 速度控制分析:")
        print(f"  自车速度范围: {ego_speeds.min():.1f} - {ego_speeds.max():.1f} km/h")
        print(f"  目标速度范围: {target_speeds.min():.1f} - {target_speeds.max():.1f} km/h")
        print(f"  平均速度误差: {speed_errors.mean():.2f} km/h")
        print(f"  最大速度误差: {speed_errors.max():.2f} km/h")
        print(f"  速度误差标准差: {speed_errors.std():.2f} km/h")
        
        # 速度误差评估
        if speed_errors.mean() < 3.0:
            print("  [OK] 速度控制精度良好")
        else:
            print("  [WARNING] 速度控制精度需要改进")
    
    # 距离分析
    if 'Actual_Distance(m)' in data.columns and 'Desired_Distance(m)' in data.columns:
        actual_distances = data['Actual_Distance(m)']
        desired_distances = data['Desired_Distance(m)']
        
        # 过滤有效数据
        valid_mask = (actual_distances > 0) & (desired_distances > 0)
        valid_actual = actual_distances[valid_mask]
        valid_desired = desired_distances[valid_mask]
        
        if len(valid_actual) > 0:
            distance_errors = np.abs(valid_actual - valid_desired)
            
            print(f"\n[DISTANCE] 距离控制分析:")
            print(f"  有效数据点: {len(valid_actual)}/{len(actual_distances)} ({len(valid_actual)/len(actual_distances)*100:.1f}%)")
            print(f"  实际距离范围: {valid_actual.min():.1f} - {valid_actual.max():.1f} m")
            print(f"  期望距离范围: {valid_desired.min():.1f} - {valid_desired.max():.1f} m")
            print(f"  平均距离误差: {distance_errors.mean():.2f} m")
            print(f"  最大距离误差: {distance_errors.max():.2f} m")
            print(f"  距离误差标准差: {distance_errors.std():.2f} m")
            
            # 距离误差评估
            if distance_errors.mean() < 2.0:
                print("  [OK] 距离控制精度良好")
            else:
                print("  [WARNING] 距离控制精度需要改进")
        else:
            print(f"\n[DISTANCE] 无有效距离控制数据")
    
    # 控制模式分析
    if 'Control_Mode' in data.columns:
        modes = data['Control_Mode'].value_counts()
        
        print(f"\n[MODE] 控制模式分布:")
        for mode, count in modes.items():
            percentage = count / len(data) * 100
            print(f"  {mode}: {count} 次 ({percentage:.1f}%)")
        
        # 模式切换分析
        mode_changes = 0
        for i in range(1, len(data)):
            if data['Control_Mode'].iloc[i] != data['Control_Mode'].iloc[i-1]:
                mode_changes += 1
        
        switch_frequency = mode_changes / data['Time(s)'].max() if 'Time(s)' in data.columns else 0
        print(f"  模式切换次数: {mode_changes}")
        print(f"  切换频率: {switch_frequency:.2f} 次/秒")
        
        if 0.1 <= switch_frequency <= 0.5:
            print("  [OK] 模式切换频率合理")
        elif switch_frequency < 0.1:
            print("  [INFO] 模式切换较少，可能控制条件稳定")
        else:
            print("  [WARNING] 模式切换过于频繁")
    
    # ACC状态分析
    if 'ACC_Active' in data.columns:
        acc_active = data['ACC_Active']
        active_time = acc_active.sum()
        total_time = len(acc_active)
        active_ratio = active_time / total_time * 100
        
        print(f"\n[ACC] ACC状态分析:")
        print(f"  ACC激活时间: {active_time}/{total_time} ({active_ratio:.1f}%)")
        
        if active_ratio > 80:
            print("  [OK] ACC工作时间充足")
        else:
            print("  [WARNING] ACC工作时间偏少，可能存在频繁切换")
    
    # 车道保持分析
    if 'Lane_Offset' in data.columns:
        lane_offsets = data['Lane_Offset']
        valid_offsets = lane_offsets[np.isfinite(lane_offsets)]
        
        if len(valid_offsets) > 0:
            print(f"\n[LANE] 车道保持分析:")
            print(f"  平均车道偏移: {valid_offsets.mean():.3f} m")
            print(f"  最大车道偏移: {valid_offsets.abs().max():.3f} m")
            print(f"  车道偏移标准差: {valid_offsets.std():.3f} m")
            
            if valid_offsets.abs().max() < 0.5:
                print("  [OK] 车道保持性能良好")
            else:
                print("  [WARNING] 车道偏移较大")


def generate_simple_report(data):
    """生成简单的分析报告"""
    if data is None:
        return
    
    print("\n" + "="*50)
    print("系统性能评估")
    print("="*50)
    
    scores = []
    
    # 速度控制评分
    if 'Ego_Speed(km/h)' in data.columns and 'Target_Speed(km/h)' in data.columns:
        speed_errors = np.abs(data['Ego_Speed(km/h)'] - data['Target_Speed(km/h)'])
        speed_score = max(0, min(100, 100 - speed_errors.mean() * 20))  # 每1km/h误差扣20分
        scores.append(('速度控制', speed_score))
        print(f"速度控制: {speed_score:.1f}/100")
    
    # 距离控制评分
    if 'Actual_Distance(m)' in data.columns and 'Desired_Distance(m)' in data.columns:
        actual_distances = data['Actual_Distance(m)']
        desired_distances = data['Desired_Distance(m)']
        valid_mask = (actual_distances > 0) & (desired_distances > 0)
        
        if valid_mask.any():
            distance_errors = np.abs(actual_distances[valid_mask] - desired_distances[valid_mask])
            distance_score = max(0, min(100, 100 - distance_errors.mean() * 10))  # 每1m误差扣10分
            scores.append(('距离控制', distance_score))
            print(f"距离控制: {distance_score:.1f}/100")
    
    # 综合评分
    if scores:
        overall_score = np.mean([score for _, score in scores])
        print(f"\n综合评分: {overall_score:.1f}/100")
        
        if overall_score >= 90:
            grade = "优秀"
        elif overall_score >= 80:
            grade = "良好"
        elif overall_score >= 70:
            grade = "合格"
        else:
            grade = "需要改进"
        
        print(f"评估等级: {grade}")
        
        # 保存简单报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"simple_analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("两模式控制系统数据分析报告\n")
            f.write("="*40 + "\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据点数: {len(data)}\n\n")
            
            for name, score in scores:
                f.write(f"{name}: {score:.1f}/100\n")
            
            f.write(f"\n综合评分: {overall_score:.1f}/100\n")
            f.write(f"评估等级: {grade}\n")
        
        print(f"\n[INFO] 简单报告已保存至: {report_file}")


def main():
    """主分析函数"""
    print("两模式控制数据分析工具 - 简化版")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    csv_file = 'speed_data_integrated.csv'
    data = load_csv_data(csv_file)
    
    if data is not None:
        # 执行分析
        analyze_data(data)
        generate_simple_report(data)
        
        print("\n[SUCCESS] 数据分析完成")
        print("\n建议:")
        print("1. 检查速度和距离控制的误差是否在合理范围内")
        print("2. 观察模式切换频率是否适中")
        print("3. 验证ACC激活比例是否满足期望")
        print("4. 对比三模式控制的历史数据（如果有的话）")
        
        return True
    else:
        print("\n[ERROR] 数据加载失败，无法进行分析")
        print("\n建议:")
        print("1. 确认CSV文件存在且格式正确")
        print("2. 检查文件是否被其他程序占用")
        print("3. 重新运行CARLA仿真生成新数据")
        
        return False


if __name__ == "__main__":
    main()