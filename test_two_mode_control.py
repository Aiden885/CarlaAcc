#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两模式控制系统测试脚本
用于验证从三模式到两模式的改进效果
包含单元测试、集成测试和性能分析
"""

import time
import csv
import numpy as np
from datetime import datetime
import os
import sys

# 设置输出编码，避免Windows控制台显示问题
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# 设置matplotlib后端，避免GUI问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 导入项目模块
try:
    from two_mode_controller import TwoModeController, calculate_two_mode_desired_distance, two_mode_control
    from acc_decision import ACCDecisionModule, ACCCommand, ACCState
    print("[OK] 成功导入两模式控制模块")
except ImportError as e:
    print(f"[ERROR] 导入模块失败: {e}")
    exit(1)


class TwoModeControlTester:
    """两模式控制测试器"""
    
    def __init__(self):
        self.test_results = []
        self.performance_data = []
        
        # 测试参数
        self.V_threshold_kmh = 50.0  # 模式切换阈值
        self.G2_s = 2.0              # 时距参数
        self.target_speed_kmh = 60.0 # 目标速度
        
        # 创建控制器实例
        self.controller = TwoModeController(
            V_threshold_kmh=self.V_threshold_kmh,
            G2_s=self.G2_s,
            target_speed_kmh=self.target_speed_kmh
        )
        self.controller.debug = True
        
        # 创建ACC决策模块
        self.acc_decision = ACCDecisionModule(
            initial_target_speed_kmh=self.target_speed_kmh,
            initial_time_gap=self.G2_s
        )
        self.acc_decision.set_debug(True)
        
    def test_mode_switching_logic(self):
        """测试1: 模式切换逻辑验证"""
        print("\n" + "="*60)
        print("🧪 测试1: 两模式切换逻辑验证")
        print("="*60)
        
        test_speeds = [20, 35, 45, 49, 50, 51, 55, 70, 80]  # km/h
        results = []
        
        for speed_kmh in test_speeds:
            speed_ms = speed_kmh / 3.6
            
            # 测试模式判断
            mode = self.controller.determine_control_mode(speed_ms)
            
            # 测试期望距离计算
            desired_distance, control_mode = self.controller.calculate_desired_distance(speed_ms)
            
            # 验证逻辑正确性
            expected_mode = 'TIME' if speed_kmh <= self.V_threshold_kmh else 'SPEED'
            is_correct = (mode == expected_mode) and (mode == control_mode)
            
            result = {
                'speed_kmh': speed_kmh,
                'detected_mode': mode,
                'expected_mode': expected_mode,
                'desired_distance': desired_distance,
                'is_correct': is_correct
            }
            results.append(result)
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} 速度: {speed_kmh:2.0f}km/h -> 模式: {mode:5s} -> 期望距离: {desired_distance:5.1f}m")
        
        # 统计结果
        correct_count = sum(1 for r in results if r['is_correct'])
        success_rate = correct_count / len(results) * 100
        
        print(f"\n📊 模式切换测试结果: {correct_count}/{len(results)} 正确, 成功率: {success_rate:.1f}%")
        
        self.test_results.append({
            'test_name': '模式切换逻辑',
            'success_rate': success_rate,
            'details': results
        })
        
        return success_rate > 95  # 95%以上为通过
    
    def test_distance_calculation(self):
        """测试2: 期望距离计算验证"""
        print("\n" + "="*60)
        print("🧪 测试2: 期望距离计算验证")
        print("="*60)
        
        test_cases = [
            # (speed_kmh, expected_distance_formula)
            (30, 30/3.6 * self.G2_s),  # 时距控制: v*G2
            (40, 40/3.6 * self.G2_s),  # 时距控制: v*G2
            (50, 50/3.6 * self.G2_s),  # 边界值: v*G2
            (60, self.V_threshold_kmh/3.6 * self.G2_s),  # 定速控制: V_threshold*G2
            (80, self.V_threshold_kmh/3.6 * self.G2_s),  # 定速控制: V_threshold*G2
        ]
        
        results = []
        
        for speed_kmh, expected_distance in test_cases:
            speed_ms = speed_kmh / 3.6
            calculated_distance, mode = self.controller.calculate_desired_distance(speed_ms)
            
            # 允许小误差
            error = abs(calculated_distance - expected_distance)
            is_correct = error < 0.1  # 10cm误差范围
            
            result = {
                'speed_kmh': speed_kmh,
                'mode': mode,
                'calculated_distance': calculated_distance,
                'expected_distance': expected_distance,
                'error': error,
                'is_correct': is_correct
            }
            results.append(result)
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} {speed_kmh:2.0f}km/h ({mode:5s}): 计算={calculated_distance:5.1f}m, "
                  f"期望={expected_distance:5.1f}m, 误差={error:4.2f}m")
        
        correct_count = sum(1 for r in results if r['is_correct'])
        success_rate = correct_count / len(results) * 100
        
        print(f"\n📊 距离计算测试结果: {correct_count}/{len(results)} 正确, 成功率: {success_rate:.1f}%")
        
        self.test_results.append({
            'test_name': '距离计算',
            'success_rate': success_rate,
            'details': results
        })
        
        return success_rate > 95
    
    def test_acc_decision_integration(self):
        """测试3: ACC决策模块集成测试"""
        print("\n" + "="*60)
        print("🧪 测试3: ACC决策模块集成测试")
        print("="*60)
        
        # 模拟典型ACC操作序列
        test_sequence = [
            (ACCCommand.ENGAGE, 40.0, False, "ACC开启"),
            (ACCCommand.INCREASE_SPEED, 45.0, False, "增速操作"),
            (ACCCommand.INCREASE_DISTANCE, 50.0, True, "增距操作"),
            (ACCCommand.CRUISE_MODE, 55.0, True, "切换定速巡航"),
            (ACCCommand.DECREASE_SPEED, 60.0, True, "定速模式减速"),
            (ACCCommand.ENGAGE, 45.0, True, "退出定速模式"),
            (ACCCommand.BRAKE, 40.0, True, "人工刹车干预"),
            (ACCCommand.ENGAGE, 35.0, False, "重新激活"),
        ]
        
        results = []
        
        for command, speed, has_target, description in test_sequence:
            distance = 25.0 if has_target else None
            
            # 执行指令
            state, mode, msg = self.acc_decision.process_command(
                command, speed, has_target, distance
            )
            
            # 获取决策输出
            decision = self.acc_decision.get_decision_output(speed, distance)
            
            result = {
                'description': description,
                'command': command.value,
                'input_speed': speed,
                'has_target': has_target,
                'resulting_state': state.value,
                'control_mode': mode.value if mode else None,
                'message': msg,
                'control_enabled': decision['control_enabled'],
                'V_target': decision['V_target_kmh'],
                'G2': decision['G2_s']
            }
            results.append(result)
            
            print(f"  📝 {description}:")
            print(f"     指令: {command.value} -> 状态: {state.value}")
            print(f"     消息: {msg}")
            print(f"     控制启用: {decision['control_enabled']}, V_target: {decision['V_target_kmh']:.1f}km/h")
            print()
        
        # 验证最终状态的合理性
        final_params = self.acc_decision.get_current_parameters()
        is_reasonable = (
            20 <= final_params['V_target_kmh'] <= 120 and
            1.0 <= final_params['G2_s'] <= 5.0
        )
        
        success_rate = 100 if is_reasonable else 0
        
        print(f"📊 ACC决策集成测试: {'通过' if is_reasonable else '失败'}")
        print(f"   最终参数: V_target={final_params['V_target_kmh']:.1f}km/h, G2={final_params['G2_s']:.1f}s")
        
        self.test_results.append({
            'test_name': 'ACC决策集成',
            'success_rate': success_rate,
            'details': results
        })
        
        return is_reasonable
    
    def test_performance_simulation(self):
        """测试4: 性能仿真测试"""
        print("\n" + "="*60)
        print("🧪 测试4: 性能仿真测试")
        print("="*60)
        
        # 仿真场景参数
        simulation_time = 60.0  # 60秒仿真
        dt = 0.1  # 100ms时间步长
        steps = int(simulation_time / dt)
        
        # 模拟前车速度变化
        target_speeds = []
        ego_speeds = []
        distances = []
        control_modes = []
        desired_distances = []
        
        # 初始状态
        ego_speed = 30.0 / 3.6  # 30 km/h
        target_speed = 50.0 / 3.6  # 50 km/h
        distance = 30.0  # 30m
        
        print(f"  🚗 仿真参数: 时长={simulation_time}s, 步长={dt}s, 总步数={steps}")
        print(f"  📊 正在运行仿真...")
        
        start_time = time.time()
        
        for step in range(steps):
            t = step * dt
            
            # 模拟前车速度变化（正弦波形）
            target_speed = (50 + 10 * np.sin(0.1 * t)) / 3.6
            
            # 使用两模式控制计算期望距离
            desired_distance, mode = calculate_two_mode_desired_distance(ego_speed)
            
            # 简单的车辆动力学仿真
            distance_error = distance - desired_distance
            speed_error = target_speed - ego_speed
            
            # 简化的控制逻辑
            if mode == 'TIME':
                # 时距控制：主要考虑距离误差
                accel = -0.5 * distance_error + 0.3 * speed_error
            else:
                # 定速控制：主要考虑速度误差
                accel = 0.8 * speed_error - 0.2 * distance_error
            
            # 限制加速度
            accel = np.clip(accel, -3.0, 2.0)
            
            # 更新状态
            ego_speed += accel * dt
            ego_speed = max(0, ego_speed)  # 速度不能为负
            
            # 更新距离（简化模型）
            relative_speed = target_speed - ego_speed
            distance += relative_speed * dt
            distance = max(5.0, distance)  # 最小安全距离
            
            # 记录数据
            if step % 10 == 0:  # 每1秒记录一次
                target_speeds.append(target_speed * 3.6)
                ego_speeds.append(ego_speed * 3.6)
                distances.append(distance)
                control_modes.append(mode)
                desired_distances.append(desired_distance)
        
        simulation_time_elapsed = time.time() - start_time
        
        # 分析性能指标
        speed_errors = [abs(ts - es) for ts, es in zip(target_speeds, ego_speeds)]
        distance_errors = [abs(d - dd) for d, dd in zip(distances, desired_distances)]
        
        avg_speed_error = np.mean(speed_errors)
        max_speed_error = np.max(speed_errors)
        avg_distance_error = np.mean(distance_errors)
        max_distance_error = np.max(distance_errors)
        
        # 统计模式切换次数
        mode_switches = sum(1 for i in range(1, len(control_modes)) 
                           if control_modes[i] != control_modes[i-1])
        
        performance_metrics = {
            'simulation_duration': simulation_time_elapsed,
            'avg_speed_error_kmh': avg_speed_error,
            'max_speed_error_kmh': max_speed_error,
            'avg_distance_error_m': avg_distance_error,
            'max_distance_error_m': max_distance_error,
            'mode_switches': mode_switches,
            'time_mode_ratio': control_modes.count('TIME') / len(control_modes),
            'speed_mode_ratio': control_modes.count('SPEED') / len(control_modes)
        }
        
        print(f"  ⏱️  仿真耗时: {simulation_time_elapsed:.3f}秒")
        print(f"  🎯 速度跟踪: 平均误差 {avg_speed_error:.1f}km/h, 最大误差 {max_speed_error:.1f}km/h")
        print(f"  📏 距离控制: 平均误差 {avg_distance_error:.1f}m, 最大误差 {max_distance_error:.1f}m")
        print(f"  🔄 模式切换: {mode_switches}次")
        print(f"  📊 模式比例: 时距控制 {performance_metrics['time_mode_ratio']:.1%}, 定速控制 {performance_metrics['speed_mode_ratio']:.1%}")
        
        self.performance_data = {
            'time': [i * dt for i in range(0, len(target_speeds))],
            'target_speeds': target_speeds,
            'ego_speeds': ego_speeds,
            'distances': distances,
            'desired_distances': desired_distances,
            'control_modes': control_modes,
            'metrics': performance_metrics
        }
        
        # 判断性能是否合格
        performance_good = (
            avg_speed_error < 5.0 and  # 平均速度误差 < 5km/h
            avg_distance_error < 3.0 and  # 平均距离误差 < 3m
            simulation_time_elapsed < 1.0  # 仿真耗时 < 1秒
        )
        
        success_rate = 100 if performance_good else 0
        
        self.test_results.append({
            'test_name': '性能仿真',
            'success_rate': success_rate,
            'details': performance_metrics
        })
        
        return performance_good
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n" + "="*60)
        print("📋 两模式控制系统测试报告")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success_rate'] >= 95)
        overall_success_rate = np.mean([result['success_rate'] for result in self.test_results])
        
        print(f"🎯 测试概览:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过测试: {passed_tests}")
        print(f"   整体成功率: {overall_success_rate:.1f}%")
        print()
        
        print(f"📊 详细结果:")
        for result in self.test_results:
            status = "✅ PASS" if result['success_rate'] >= 95 else "❌ FAIL"
            print(f"   {status} {result['test_name']}: {result['success_rate']:.1f}%")
        
        # 保存测试报告到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"two_mode_test_report_{timestamp}.csv"
        
        with open(report_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['测试项目', '成功率(%)', '状态', '备注'])
            
            for result in self.test_results:
                status = "PASS" if result['success_rate'] >= 95 else "FAIL"
                writer.writerow([
                    result['test_name'], 
                    f"{result['success_rate']:.1f}", 
                    status,
                    str(result['details'])[:100] + "..." if len(str(result['details'])) > 100 else str(result['details'])
                ])
            
            # 添加总结行
            writer.writerow([])
            writer.writerow(['总体评估', f"{overall_success_rate:.1f}", 
                           "PASS" if overall_success_rate >= 95 else "FAIL", 
                           f"{passed_tests}/{total_tests}项测试通过"])
        
        print(f"\n📄 测试报告已保存至: {report_file}")
        
        return overall_success_rate >= 95
    
    def plot_performance_data(self):
        """绘制性能数据图表"""
        if not self.performance_data:
            print("❌ 无性能数据可绘制")
            return
        
        print("\n📈 正在生成性能分析图表...")
        
        # 设置matplotlib支持中文显示
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            # 如果中文字体设置失败，使用英文
            pass
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        try:
            fig.suptitle('Two Mode Control System Performance Analysis', fontsize=16, fontweight='bold')
        except:
            fig.suptitle('Two Mode Control Performance', fontsize=16, fontweight='bold')
        
        time_data = self.performance_data['time']
        
        # 子图1: 速度跟踪
        axes[0, 0].plot(time_data, self.performance_data['target_speeds'], 'r-', label='Target Speed', linewidth=2)
        axes[0, 0].plot(time_data, self.performance_data['ego_speeds'], 'b-', label='Ego Speed', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Speed (km/h)')
        axes[0, 0].set_title('Speed Tracking Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2: 距离控制
        axes[0, 1].plot(time_data, self.performance_data['distances'], 'g-', label='Actual Distance', linewidth=2)
        axes[0, 1].plot(time_data, self.performance_data['desired_distances'], 'r--', label='Desired Distance', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Distance (m)')
        axes[0, 1].set_title('Distance Control Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3: 控制模式切换
        mode_numeric = [1 if mode == 'TIME' else 2 for mode in self.performance_data['control_modes']]
        axes[1, 0].plot(time_data, mode_numeric, 'o-', markersize=4, linewidth=1)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Control Mode')
        axes[1, 0].set_title('Control Mode Switching')
        axes[1, 0].set_yticks([1, 2])
        axes[1, 0].set_yticklabels(['Time Control', 'Speed Control'])
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: 误差分析
        speed_errors = [abs(ts - es) for ts, es in zip(self.performance_data['target_speeds'], 
                                                      self.performance_data['ego_speeds'])]
        distance_errors = [abs(d - dd) for d, dd in zip(self.performance_data['distances'], 
                                                        self.performance_data['desired_distances'])]
        
        axes[1, 1].plot(time_data, speed_errors, 'r-', label='Speed Error (km/h)', linewidth=2)
        ax2 = axes[1, 1].twinx()
        ax2.plot(time_data, distance_errors, 'b-', label='Distance Error (m)', linewidth=2)
        
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Speed Error (km/h)', color='r')
        ax2.set_ylabel('Distance Error (m)', color='b')
        axes[1, 1].set_title('Control Error Analysis')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加图例
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"two_mode_performance_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 性能图表已保存至: {plot_file}")
        
        # 关闭图形以释放内存
        plt.close()
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始两模式控制系统完整测试")
        print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 运行所有测试
        tests = [
            ("模式切换逻辑", self.test_mode_switching_logic),
            ("期望距离计算", self.test_distance_calculation),
            ("ACC决策集成", self.test_acc_decision_integration),
            ("性能仿真", self.test_performance_simulation)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            try:
                print(f"\n🔄 正在执行: {test_name}")
                result = test_func()
                if not result:
                    all_passed = False
                    print(f"❌ {test_name} 测试失败")
                else:
                    print(f"✅ {test_name} 测试通过")
            except Exception as e:
                print(f"❌ {test_name} 测试异常: {e}")
                all_passed = False
        
        # 生成报告
        report_passed = self.generate_test_report()
        
        # 绘制性能图表
        if self.performance_data:
            try:
                self.plot_performance_data()
            except Exception as e:
                print(f"⚠️  图表生成失败: {e}")
        
        # 最终结果
        print("\n" + "="*60)
        if all_passed and report_passed:
            print("🎉 两模式控制系统测试全部通过！")
            print("✅ 系统已准备好进行CARLA仿真测试")
        else:
            print("⚠️  部分测试未通过，建议检查代码逻辑")
            print("💡 请根据测试报告修复问题后重新测试")
        print("="*60)
        
        return all_passed and report_passed


def main():
    """主测试函数"""
    print("🔧 两模式控制系统测试工具")
    print("   用于验证从三模式到两模式的改进效果\n")
    
    tester = TwoModeControlTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 测试建议:")
        print("1. 运行 CARLA 仿真测试验证实际效果")
        print("2. 检查 CSV 数据文件分析控制性能")
        print("3. 对比三模式控制的历史数据")
        print("4. 监控系统资源使用情况")
    
    return success


if __name__ == "__main__":
    main()