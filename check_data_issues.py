#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查CARLA仿真数据问题并提供修复建议
"""

import pandas as pd
import numpy as np


def diagnose_data_issues():
    """诊断数据问题"""
    print("两模式控制数据问题诊断")
    print("="*50)
    
    try:
        # 加载数据
        data = pd.read_csv('speed_data_integrated.csv', encoding='gbk')
        print(f"[OK] 数据加载成功: {len(data)} 行")
        
        # 1. 检查ACC激活状态
        acc_states = data['ACC_State'].value_counts()
        print(f"\n[ISSUE 1] ACC状态分布:")
        for state, count in acc_states.items():
            percentage = count / len(data) * 100
            print(f"  {state}: {count} 次 ({percentage:.1f}%)")
        
        if '系统退出' in acc_states and acc_states['系统退出'] > len(data) * 0.8:
            print("  [PROBLEM] ACC主要处于系统退出状态，未正确激活")
            print("  [SOLUTION] 在CARLA仿真中按数字键'1'激活ACC")
        
        # 2. 检查速度控制问题
        ego_speeds = data['Ego_Speed(km/h)']
        target_speeds = data['Target_Speed(km/h)']
        speed_errors = np.abs(ego_speeds - target_speeds)
        
        print(f"\n[ISSUE 2] 速度控制分析:")
        print(f"  自车速度: {ego_speeds.min():.1f} - {ego_speeds.max():.1f} km/h")
        print(f"  目标速度: {target_speeds.min():.1f} - {target_speeds.max():.1f} km/h")
        print(f"  平均速度误差: {speed_errors.mean():.1f} km/h")
        
        if speed_errors.mean() > 5:
            print("  [PROBLEM] 速度跟踪误差过大")
            print("  [SOLUTION] 检查ACC是否激活，调整控制参数")
        
        # 3. 检查距离控制问题
        actual_distances = data['Actual_Distance(m)']
        desired_distances = data['Desired_Distance(m)']
        
        print(f"\n[ISSUE 3] 距离控制分析:")
        print(f"  实际距离: {actual_distances.min():.1f} - {actual_distances.max():.1f} m")
        print(f"  期望距离: {desired_distances.min():.1f} - {desired_distances.max():.1f} m")
        
        # 检查期望距离是否合理
        if desired_distances.max() < 10:
            print("  [PROBLEM] 期望距离异常小，可能参数设置有问题")
            print("  [SOLUTION] 检查两模式控制参数设置")
        
        # 4. 检查控制模式分布
        if 'Control_Mode' in data.columns:
            mode_counts = data['Control_Mode'].value_counts()
            print(f"\n[ISSUE 4] 控制模式分布:")
            for mode, count in mode_counts.items():
                percentage = count / len(data) * 100
                print(f"  {mode}: {count} 次 ({percentage:.1f}%)")
            
            if 'TIME' in mode_counts and mode_counts['TIME'] > len(data) * 0.9:
                print("  [INFO] 主要使用时距控制模式（低速）")
            elif 'SPEED' in mode_counts and mode_counts['SPEED'] > len(data) * 0.9:
                print("  [INFO] 主要使用定速控制模式（高速）")
        
        # 5. 提供具体的操作建议
        print(f"\n" + "="*50)
        print("修复建议和操作指南")
        print("="*50)
        
        print("\n[步骤1] 确保CARLA环境正确:")
        print("  1. 启动CARLA服务器: ./CarlaUE4.sh -quality-level=Low")
        print("  2. 等待CARLA完全加载后再运行Python脚本")
        
        print("\n[步骤2] 正确激活ACC系统:")
        print("  1. 运行: python acc_updated.py")
        print("  2. 等待Pygame窗口出现")
        print("  3. 按键'1'激活ACC (看到'ACC开启'消息)")
        print("  4. 观察ACC_State是否变为'在控'或'适速'状态")
        
        print("\n[步骤3] 测试不同场景:")
        print("  1. 低速测试 (20-40 km/h): 验证时距控制")
        print("  2. 高速测试 (50-70 km/h): 验证定速控制")
        print("  3. 模式切换 (48-52 km/h): 验证切换平滑性")
        
        print("\n[步骤4] 参数调整建议:")
        print("  如果控制效果不理想，可以调整:")
        print("  - V_threshold_kmh: 模式切换阈值（默认50km/h）")
        print("  - G2_s: 时距参数（默认2.0秒）")
        print("  - target_speed_kmh: 目标速度（默认50km/h）")
        
        print("\n[步骤5] 数据验证:")
        print("  运行新的测试后，检查以下指标:")
        print("  - ACC_Active 应该大部分为 True")
        print("  - 速度误差应该 < 3 km/h")
        print("  - 距离误差应该 < 2 m")
        print("  - 应该看到TIME和SPEED两种模式切换")
        
        # 6. 生成快速测试脚本建议
        print(f"\n[快速测试] 建议的测试序列:")
        test_sequence = [
            "1. 启动仿真: python acc_updated.py",
            "2. 按'1'激活ACC",
            "3. 按'Q'增速到60km/h (观察模式切换)",
            "4. 按'E'减速到40km/h (观察模式切换)",
            "5. 按'3'切换定速巡航模式",
            "6. 运行5-10分钟收集数据",
            "7. 按ESC退出",
            "8. 运行分析: python simple_analyze.py"
        ]
        
        for i, step in enumerate(test_sequence, 1):
            print(f"  {step}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 诊断失败: {e}")
        return False


if __name__ == "__main__":
    diagnose_data_issues()