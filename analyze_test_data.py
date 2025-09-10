#!/usr/bin/env python3
"""
两模式控制数据分析脚本
分析CARLA仿真生成的CSV数据，验证控制效果
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# 设置matplotlib后端，避免GUI相关问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns


class TwoModeDataAnalyzer:
    """两模式控制数据分析器"""
    
    def __init__(self, csv_file='speed_data_integrated.csv'):
        self.csv_file = csv_file
        self.data = None
        self.analysis_results = {}
        
        # 设置字体支持（优先使用系统字体，避免中文显示警告）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            # 字体设置失败时使用默认字体
            pass
        
    def load_data(self):
        """加载CSV数据"""
        try:
            if not os.path.exists(self.csv_file):
                print(f"[ERROR] 数据文件不存在: {self.csv_file}")
                return False
            
            # 尝试不同的编码方式加载CSV文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.csv_file, encoding=encoding)
                    print(f"[OK] 使用 {encoding} 编码成功加载数据: {len(self.data)} 行记录")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print("[ERROR] 尝试所有编码都失败，使用错误处理模式")
                self.data = pd.read_csv(self.csv_file, encoding='utf-8', errors='ignore')
            
            print(f"[INFO] 数据列: {list(self.data.columns)}")
            
            # 显示数据基本信息
            if 'Time(s)' in self.data.columns:
                print(f"[INFO] 数据时长: {self.data['Time(s)'].max():.1f} 秒")
            if 'Ego_Speed(km/h)' in self.data.columns:
                print(f"[INFO] 速度范围: {self.data['Ego_Speed(km/h)'].min():.1f} - {self.data['Ego_Speed(km/h)'].max():.1f} km/h")
            if 'Actual_Distance(m)' in self.data.columns:
                print(f"[INFO] 距离范围: {self.data['Actual_Distance(m)'].min():.1f} - {self.data['Actual_Distance(m)'].max():.1f} m")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 加载数据失败: {e}")
            return False
    
    def analyze_control_performance(self):
        """分析控制性能"""
        if self.data is None:
            print("❌ 请先加载数据")
            return
            
        print("\n" + "="*60)
        print("📊 控制性能分析")
        print("="*60)
        
        # 1. 速度跟踪性能
        ego_speeds = self.data['Ego_Speed(km/h)']
        target_speeds = self.data['Target_Speed(km/h)']
        
        # 计算速度误差
        speed_errors = np.abs(ego_speeds - target_speeds)
        speed_rmse = np.sqrt(np.mean(speed_errors**2))
        
        print(f"[SPEED] 速度跟踪性能:")
        print(f"   平均速度误差: {speed_errors.mean():.2f} km/h")
        print(f"   最大速度误差: {speed_errors.max():.2f} km/h")
        print(f"   速度RMSE: {speed_rmse:.2f} km/h")
        print(f"   速度误差标准差: {speed_errors.std():.2f} km/h")
        
        # 2. 距离控制性能
        actual_distances = self.data['Actual_Distance(m)']
        desired_distances = self.data['Desired_Distance(m)']
        
        # 计算距离误差（排除无效数据）
        valid_mask = (actual_distances > 0) & (desired_distances > 0)
        valid_actual = actual_distances[valid_mask]
        valid_desired = desired_distances[valid_mask]
        
        if len(valid_actual) > 0:
            distance_errors = np.abs(valid_actual - valid_desired)
            distance_rmse = np.sqrt(np.mean(distance_errors**2))
            
            print(f"\n📏 距离控制性能:")
            print(f"   平均距离误差: {distance_errors.mean():.2f} m")
            print(f"   最大距离误差: {distance_errors.max():.2f} m")
            print(f"   距离RMSE: {distance_rmse:.2f} m")
            print(f"   距离误差标准差: {distance_errors.std():.2f} m")
            print(f"   有效距离数据: {len(valid_actual)}/{len(actual_distances)} ({len(valid_actual)/len(actual_distances)*100:.1f}%)")
        else:
            print(f"\n📏 距离控制性能: 无有效数据")
        
        # 3. 控制模式分析
        if 'Control_Mode' in self.data.columns:
            mode_counts = self.data['Control_Mode'].value_counts()
            total_count = len(self.data)
            
            print(f"\n🔄 控制模式分布:")
            for mode, count in mode_counts.items():
                percentage = count / total_count * 100
                print(f"   {mode}: {count} 次 ({percentage:.1f}%)")
        
        # 4. ACC状态分析
        if 'ACC_State' in self.data.columns:
            acc_states = self.data['ACC_State'].value_counts()
            
            print(f"\n🤖 ACC状态分布:")
            for state, count in acc_states.items():
                percentage = count / len(self.data) * 100
                print(f"   {state}: {count} 次 ({percentage:.1f}%)")
        
        # 5. 车道保持性能
        if 'Lane_Offset' in self.data.columns:
            lane_offsets = self.data['Lane_Offset']
            valid_offsets = lane_offsets[np.isfinite(lane_offsets)]
            
            if len(valid_offsets) > 0:
                print(f"\n🛣️  车道保持性能:")
                print(f"   平均车道偏移: {valid_offsets.mean():.3f} m")
                print(f"   最大车道偏移: {valid_offsets.abs().max():.3f} m")
                print(f"   车道偏移标准差: {valid_offsets.std():.3f} m")
        
        # 保存分析结果
        self.analysis_results['speed_tracking'] = {
            'mean_error': speed_errors.mean(),
            'max_error': speed_errors.max(),
            'rmse': speed_rmse,
            'std': speed_errors.std()
        }
        
        if len(valid_actual) > 0:
            self.analysis_results['distance_control'] = {
                'mean_error': distance_errors.mean(),
                'max_error': distance_errors.max(),
                'rmse': distance_rmse,
                'std': distance_errors.std(),
                'data_validity': len(valid_actual) / len(actual_distances)
            }
    
    def analyze_mode_switching(self):
        """分析模式切换行为"""
        if self.data is None or 'Control_Mode' not in self.data.columns:
            print("❌ 无控制模式数据")
            return
            
        print("\n" + "="*60)
        print("🔄 模式切换分析")
        print("="*60)
        
        # 检测模式切换
        modes = self.data['Control_Mode']
        speeds = self.data['Ego_Speed(km/h)']
        
        # 统计切换次数
        mode_changes = 0
        switch_speeds = []
        
        for i in range(1, len(modes)):
            if modes.iloc[i] != modes.iloc[i-1]:
                mode_changes += 1
                switch_speeds.append(speeds.iloc[i])
        
        print(f"📊 模式切换统计:")
        print(f"   总切换次数: {mode_changes}")
        print(f"   切换频率: {mode_changes / self.data['Time(s)'].max():.2f} 次/秒")
        
        if switch_speeds:
            print(f"   切换时平均速度: {np.mean(switch_speeds):.1f} km/h")
            print(f"   切换速度范围: {np.min(switch_speeds):.1f} - {np.max(switch_speeds):.1f} km/h")
        
        # 分析切换速度分布
        V_threshold = 50.0  # 假设阈值为50km/h
        threshold_switches = [s for s in switch_speeds if abs(s - V_threshold) < 5.0]
        
        print(f"   阈值附近切换({V_threshold}±5km/h): {len(threshold_switches)} 次")
        
        self.analysis_results['mode_switching'] = {
            'total_switches': mode_changes,
            'switch_frequency': mode_changes / self.data['Time(s)'].max(),
            'threshold_switches': len(threshold_switches),
            'switch_speeds': switch_speeds
        }
    
    def analyze_stability(self):
        """分析系统稳定性"""
        if self.data is None:
            print("❌ 请先加载数据")
            return
            
        print("\n" + "="*60)
        print("📈 系统稳定性分析")
        print("="*60)
        
        # 1. 速度稳定性
        ego_speeds = self.data['Ego_Speed(km/h)']
        speed_variance = ego_speeds.var()
        speed_smoothness = np.mean(np.abs(np.diff(ego_speeds)))
        
        print(f"🚗 速度稳定性:")
        print(f"   速度方差: {speed_variance:.2f}")
        print(f"   速度平滑度: {speed_smoothness:.2f} km/h/step")
        
        # 2. 控制输出稳定性
        if 'Manual_Throttle' in self.data.columns and 'Manual_Brake' in self.data.columns:
            throttle = self.data['Manual_Throttle']
            brake = self.data['Manual_Brake']
            
            throttle_variance = throttle.var()
            brake_variance = brake.var()
            
            print(f"\n⚙️ 控制输出稳定性:")
            print(f"   油门方差: {throttle_variance:.4f}")
            print(f"   刹车方差: {brake_variance:.4f}")
        
        # 3. 震荡检测
        window_size = 50  # 5秒窗口（假设10Hz采样）
        oscillation_count = 0
        
        for i in range(window_size, len(ego_speeds) - window_size):
            window = ego_speeds.iloc[i-window_size:i+window_size]
            if window.max() - window.min() > 10:  # 速度变化超过10km/h
                oscillation_count += 1
        
        oscillation_ratio = oscillation_count / (len(ego_speeds) - 2*window_size)
        
        print(f"\n🌊 震荡分析:")
        print(f"   震荡点数: {oscillation_count}")
        print(f"   震荡比例: {oscillation_ratio:.2%}")
        
        self.analysis_results['stability'] = {
            'speed_variance': speed_variance,
            'speed_smoothness': speed_smoothness,
            'oscillation_ratio': oscillation_ratio
        }
    
    def plot_comprehensive_analysis(self):
        """绘制综合分析图表"""
        if self.data is None:
            print("❌ 请先加载数据")
            return
            
        print("\n📈 正在生成综合分析图表...")
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Two Mode Control System Performance Analysis', fontsize=16, fontweight='bold')
        
        time_data = self.data['Time(s)']
        
        # 1. 速度跟踪
        axes[0, 0].plot(time_data, self.data['Target_Speed(km/h)'], 'r-', label='Target Speed', linewidth=2)
        axes[0, 0].plot(time_data, self.data['Ego_Speed(km/h)'], 'b-', label='Ego Speed', linewidth=1.5)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Speed (km/h)')
        axes[0, 0].set_title('Speed Tracking Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 距离控制
        valid_mask = (self.data['Actual_Distance(m)'] > 0) & (self.data['Desired_Distance(m)'] > 0)
        if valid_mask.any():
            valid_time = time_data[valid_mask]
            axes[0, 1].plot(valid_time, self.data['Actual_Distance(m)'][valid_mask], 'g-', label='Actual Distance', linewidth=2)
            axes[0, 1].plot(valid_time, self.data['Desired_Distance(m)'][valid_mask], 'r--', label='Desired Distance', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Distance (m)')
        axes[0, 1].set_title('Distance Control Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 控制模式分布
        if 'Control_Mode' in self.data.columns:
            mode_counts = self.data['Control_Mode'].value_counts()
            axes[0, 2].pie(mode_counts.values, labels=mode_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0, 2].set_title('Control Mode Distribution')
        else:
            axes[0, 2].text(0.5, 0.5, 'No Control Mode Data', ha='center', va='center')
            axes[0, 2].set_title('Control Mode Distribution')
        
        # 4. 误差分析
        speed_errors = np.abs(self.data['Ego_Speed(km/h)'] - self.data['Target_Speed(km/h)'])
        axes[1, 0].plot(time_data, speed_errors, 'r-', linewidth=1.5)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Speed Error (km/h)')
        axes[1, 0].set_title('Speed Tracking Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. ACC状态分析
        if 'ACC_Active' in self.data.columns:
            acc_active = self.data['ACC_Active'].astype(int)
            axes[1, 1].plot(time_data, acc_active, 'b-', linewidth=2)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('ACC Status')
            axes[1, 1].set_title('ACC Activation Status')
            axes[1, 1].set_yticks([0, 1])
            axes[1, 1].set_yticklabels(['Off', 'Active'])
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No ACC Status Data', ha='center', va='center')
            axes[1, 1].set_title('ACC Activation Status')
        
        # 6. 性能指标雷达图
        if self.analysis_results:
            categories = []
            values = []
            
            if 'speed_tracking' in self.analysis_results:
                st = self.analysis_results['speed_tracking']
                categories.extend(['速度精度', '速度稳定性'])
                # 转换为0-1评分（误差越小分数越高）
                speed_score = max(0, 1 - st['mean_error'] / 10)  # 假设10km/h为最差
                stability_score = max(0, 1 - st['std'] / 5)  # 假设5km/h标准差为最差
                values.extend([speed_score, stability_score])
            
            if 'distance_control' in self.analysis_results:
                dc = self.analysis_results['distance_control']
                categories.append('距离精度')
                distance_score = max(0, 1 - dc['mean_error'] / 5)  # 假设5m为最差
                values.append(distance_score)
            
            if 'stability' in self.analysis_results:
                stab = self.analysis_results['stability']
                categories.append('系统稳定性')
                system_score = max(0, 1 - stab['oscillation_ratio'])
                values.append(system_score)
            
            if categories:
                # 绘制雷达图
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]  # 闭合图形
                angles += angles[:1]
                
                axes[1, 2].plot(angles, values, 'o-', linewidth=2, color='blue')
                axes[1, 2].fill(angles, values, alpha=0.25, color='blue')
                axes[1, 2].set_xticks(angles[:-1])
                axes[1, 2].set_xticklabels(['Speed', 'Stability', 'Distance', 'System'], rotation=0)
                axes[1, 2].set_ylim(0, 1)
                axes[1, 2].set_title('Overall Performance')
                axes[1, 2].grid(True)
            else:
                axes[1, 2].text(0.5, 0.5, 'No Performance Data', ha='center', va='center')
                axes[1, 2].set_title('Overall Performance')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"two_mode_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 分析图表已保存至: {plot_file}")
        
        # 关闭图形以释放内存
        plt.close()
    
    def generate_analysis_report(self):
        """生成分析报告"""
        print("\n" + "="*60)
        print("📋 两模式控制数据分析报告")
        print("="*60)
        
        if not self.analysis_results:
            print("❌ 无分析结果")
            return
        
        # 计算综合评分
        total_score = 0
        score_count = 0
        
        print("📊 性能评估:")
        
        if 'speed_tracking' in self.analysis_results:
            st = self.analysis_results['speed_tracking']
            speed_score = max(0, min(100, 100 - st['mean_error'] * 10))  # 误差每1km/h扣10分
            print(f"   速度跟踪: {speed_score:.1f}/100 (平均误差: {st['mean_error']:.2f}km/h)")
            total_score += speed_score
            score_count += 1
        
        if 'distance_control' in self.analysis_results:
            dc = self.analysis_results['distance_control']
            distance_score = max(0, min(100, 100 - dc['mean_error'] * 5))  # 误差每1m扣5分
            print(f"   距离控制: {distance_score:.1f}/100 (平均误差: {dc['mean_error']:.2f}m)")
            total_score += distance_score
            score_count += 1
        
        if 'stability' in self.analysis_results:
            stab = self.analysis_results['stability']
            stability_score = max(0, min(100, 100 - stab['oscillation_ratio'] * 100))
            print(f"   系统稳定性: {stability_score:.1f}/100 (震荡比例: {stab['oscillation_ratio']:.1%})")
            total_score += stability_score
            score_count += 1
        
        if 'mode_switching' in self.analysis_results:
            ms = self.analysis_results['mode_switching']
            # 合理的切换频率应该在0.1-0.5次/秒之间
            freq = ms['switch_frequency']
            if 0.1 <= freq <= 0.5:
                switch_score = 100
            elif freq < 0.1:
                switch_score = 80  # 切换太少可能不够灵敏
            else:
                switch_score = max(0, 100 - (freq - 0.5) * 50)  # 切换太频繁扣分
            
            print(f"   模式切换: {switch_score:.1f}/100 (切换频率: {freq:.2f}次/秒)")
            total_score += switch_score
            score_count += 1
        
        # 综合评分
        if score_count > 0:
            overall_score = total_score / score_count
            print(f"\n🎯 综合评分: {overall_score:.1f}/100")
            
            if overall_score >= 90:
                grade = "优秀"
                emoji = "🎉"
            elif overall_score >= 80:
                grade = "良好"
                emoji = "✅"
            elif overall_score >= 70:
                grade = "合格"
                emoji = "⚠️"
            else:
                grade = "需要改进"
                emoji = "❌"
            
            print(f"{emoji} 评级: {grade}")
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"two_mode_data_analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("两模式控制系统数据分析报告\n")
            f.write("="*50 + "\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {self.csv_file}\n")
            f.write(f"数据记录数: {len(self.data)}\n")
            f.write(f"数据时长: {self.data['Time(s)'].max():.1f} 秒\n\n")
            
            f.write("性能指标:\n")
            for key, value in self.analysis_results.items():
                f.write(f"{key}: {value}\n")
            
            if score_count > 0:
                f.write(f"\n综合评分: {overall_score:.1f}/100\n")
                f.write(f"评级: {grade}\n")
        
        print(f"\n📄 分析报告已保存至: {report_file}")
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("🔍 开始两模式控制数据分析")
        print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 加载数据
        if not self.load_data():
            return False
        
        # 执行各项分析
        try:
            self.analyze_control_performance()
            self.analyze_mode_switching()
            self.analyze_stability()
            self.plot_comprehensive_analysis()
            self.generate_analysis_report()
            
            print("\n✅ 数据分析完成！")
            return True
            
        except Exception as e:
            print(f"❌ 分析过程出错: {e}")
            return False


def main():
    """主分析函数"""
    print("📊 两模式控制数据分析工具")
    print("   用于分析CARLA仿真产生的控制数据\n")
    
    # 检查数据文件
    csv_file = 'speed_data_integrated.csv'
    if not os.path.exists(csv_file):
        print(f"❌ 找不到数据文件: {csv_file}")
        print("💡 请先运行CARLA仿真生成数据")
        return False
    
    analyzer = TwoModeDataAnalyzer(csv_file)
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎯 分析建议:")
        print("1. 检查综合评分，重点关注低于80分的项目")
        print("2. 观察模式切换是否在合理的速度范围内")
        print("3. 检查系统稳定性，减少不必要的震荡")
        print("4. 对比历史数据验证改进效果")
    
    return success


if __name__ == "__main__":
    main()