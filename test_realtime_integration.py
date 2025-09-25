#!/usr/bin/env python3
"""
实时Carla-Simulink集成测试验证脚本
测试实时SPPVT状态管理器与ACCDecisionSPPVTInterface的完整集成
"""

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import unittest

# 导入我们的模块
try:
    from realtime_sppvt_state_manager import RealtimeSPPVTStateManager
    from acc_decision_sppvt_interface import create_decision_sppvt_interface
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入模块: {e}")
    MODULES_AVAILABLE = False


class RealtimeIntegrationTest(unittest.TestCase):
    """实时集成测试类"""

    def setUp(self):
        """测试初始化"""
        if not MODULES_AVAILABLE:
            self.skipTest("必需模块不可用")

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 初始化测试参数
        self.test_results = []

    def test_realtime_sppvt_manager_basic(self):
        """测试实时SPPVT状态管理器基本功能"""
        self.logger.info("=== 测试实时SPPVT状态管理器基本功能 ===")

        try:
            manager = RealtimeSPPVTStateManager()

            # 基本功能测试
            test_cases = [
                (0.5, 15.0, True, True, "正常控制"),
                (1.2, 12.0, True, True, "大误差控制"),
                (-0.8, 8.0, True, True, "负误差控制"),
                (0.1, 13.5, True, True, "小误差控制")
            ]

            for i, (error, speed, mode, enabled, desc) in enumerate(test_cases):
                with self.subTest(f"测试案例{i+1}: {desc}"):
                    output = manager.run_single_step_simulation(error, speed, mode, enabled)

                    # 验证输出
                    self.assertIsNotNone(output, f"案例{i+1}输出不应为None")
                    self.assertIsInstance(output, (int, float), f"案例{i+1}输出应为数值")
                    self.assertTrue(-5.0 <= output <= 5.0, f"案例{i+1}输出应在合理范围内")

                    self.logger.info(f"案例{i+1} ({desc}): 输入={error:.2f}, 输出={output:.3f}")

            # 验证状态持久化
            stats = manager.get_performance_stats()
            self.assertEqual(stats['total_simulations'], len(test_cases))
            self.assertGreater(stats['avg_execution_time_ms'], 0)

            manager.cleanup()

        except Exception as e:
            self.fail(f"实时SPPVT管理器测试失败: {e}")

    def test_acc_decision_sppvt_interface(self):
        """测试ACC决策+SPPVT接口 - 14/15-field版本"""
        self.logger.info("=== 测试ACC决策+SPPVT接口 (14/15-field) ===")

        try:
            interface = create_decision_sppvt_interface(debug=False)

            # 测试标准输入格式 - 14-field DecisionSPPVTInputExtended
            test_input = {
                # 原始11个字段
                'ego_speed_kmh': 45.0,
                'ego_speed_ms': 12.5,
                'command_type': 1,  # I0指令
                'command_active': True,
                'manual_throttle_active': False,
                'control_error': -1.5,
                'control_mode_flag': 1,  # distance mode
                'V_target_kmh': 50.0,
                'V_min_kmh': 30.0,
                'G2_s': 2.0,
                'timestamp': time.time(),

                # 新增3个外部状态字段
                'external_stage_offset': 0.0,
                'external_stage_manager_states': [0.0, 0.0, 0.0],  # [stage, error_sign, upgrade_count]
                'external_adapter_states': [0.0, 13.89, 0.1]  # [prev_error, prev_velocity, prev_accel]
            }

            result = interface.process_decision_and_control(test_input)

            # 验证输出结构 - 15-field DecisionSPPVTOutputExtended
            required_fields = [
                'target_accel', 'control_enabled', 'current_state',
                'current_decision', 'torque_arbitration_active',
                'updated_V_target_kmh', 'updated_G2_s', 'sppvt_control_output',
                'sppvt_velocity_output', 'sppvt_acceleration_output', 'sppvt_stage_output',
                'sppvt_status_output', 'debug_message', 'new_stage_offset',
                'new_stage_manager_states', 'new_adapter_states'
            ]

            for field in required_fields:
                self.assertIn(field, result, f"输出应包含字段: {field}")

            # 验证数值范围
            self.assertTrue(-5.0 <= result['target_accel'] <= 5.0)
            self.assertIn(result['current_state'], [0, 1, 2, 3])
            self.assertGreaterEqual(result['current_decision'], 0)

            # 验证新的状态外化字段
            self.assertIsInstance(result['new_stage_offset'], (int, float))
            self.assertIsInstance(result['new_stage_manager_states'], list)
            self.assertIsInstance(result['new_adapter_states'], list)
            self.assertEqual(len(result['new_stage_manager_states']), 3)
            self.assertEqual(len(result['new_adapter_states']), 3)

            self.logger.info(f"接口测试结果 (14/15-field): {result}")

        except Exception as e:
            self.fail(f"ACC决策+SPPVT接口测试失败: {e}")

    def test_state_continuity(self):
        """测试状态连续性"""
        self.logger.info("=== 测试状态连续性 ===")

        try:
            manager = RealtimeSPPVTStateManager()

            # 模拟连续控制场景
            scenarios = []
            stage_offsets = []

            for step in range(10):
                # 模拟逐渐增大的控制误差
                control_error = 0.2 + 0.1 * step
                ego_speed = 15.0 - 0.3 * step  # 逐渐减速

                output = manager.run_single_step_simulation(
                    control_error, ego_speed, True, True
                )

                stats = manager.get_performance_stats()
                stage_offset = stats['current_stage_offset']

                scenarios.append({
                    'step': step,
                    'error': control_error,
                    'speed': ego_speed,
                    'output': output,
                    'stage_offset': stage_offset
                })

                stage_offsets.append(stage_offset)

            # 验证状态连续性
            self.assertEqual(len(scenarios), 10)

            # 验证级差变化合理性
            for i in range(1, len(stage_offsets)):
                prev_offset = stage_offsets[i-1]
                curr_offset = stage_offsets[i]
                # 级差变化不应过于剧烈
                self.assertTrue(abs(curr_offset - prev_offset) <= 0.5)

            self.logger.info("状态连续性测试通过")
            manager.cleanup()

        except Exception as e:
            self.fail(f"状态连续性测试失败: {e}")

    def test_performance_benchmark(self):
        """性能基准测试"""
        self.logger.info("=== 性能基准测试 ===")

        try:
            manager = RealtimeSPPVTStateManager()

            # 性能测试参数
            num_iterations = 50
            start_time = time.time()

            execution_times = []

            for i in range(num_iterations):
                iter_start = time.time()

                # 随机测试参数
                control_error = np.random.uniform(-2.0, 2.0)
                ego_speed = np.random.uniform(8.0, 20.0)

                output = manager.run_single_step_simulation(
                    control_error, ego_speed, True, True
                )

                iter_time = time.time() - iter_start
                execution_times.append(iter_time)

                self.assertIsNotNone(output)

            total_time = time.time() - start_time
            avg_time_ms = np.mean(execution_times) * 1000
            max_time_ms = np.max(execution_times) * 1000

            # 性能要求验证
            self.assertLess(avg_time_ms, 200.0, "平均执行时间应小于200ms")
            self.assertLess(max_time_ms, 500.0, "最大执行时间应小于500ms")

            self.logger.info(f"性能统计: 平均{avg_time_ms:.1f}ms, 最大{max_time_ms:.1f}ms, 总计{total_time:.2f}s")

            manager.cleanup()

        except Exception as e:
            self.fail(f"性能基准测试失败: {e}")


class IntegrationValidationSuite:
    """集成验证测试套件"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run_comprehensive_test(self):
        """运行综合测试"""
        self.logger.info("开始运行综合集成验证测试...")

        test_results = {
            'basic_functionality': False,
            'state_management': False,
            'performance': False,
            'error_handling': False,
            'integration': False
        }

        try:
            # 1. 基础功能测试
            self.logger.info("1. 基础功能测试...")
            test_results['basic_functionality'] = self._test_basic_functionality()

            # 2. 状态管理测试
            self.logger.info("2. 状态管理测试...")
            test_results['state_management'] = self._test_state_management()

            # 3. 性能测试
            self.logger.info("3. 性能测试...")
            test_results['performance'] = self._test_performance()

            # 4. 错误处理测试
            self.logger.info("4. 错误处理测试...")
            test_results['error_handling'] = self._test_error_handling()

            # 5. 完整集成测试
            self.logger.info("5. 完整集成测试...")
            test_results['integration'] = self._test_full_integration()

        except Exception as e:
            self.logger.error(f"测试过程中发生错误: {e}")

        # 生成测试报告
        self._generate_test_report(test_results)

        return test_results

    def _test_basic_functionality(self) -> bool:
        """基础功能测试"""
        try:
            manager = RealtimeSPPVTStateManager()
            output = manager.run_single_step_simulation(1.0, 15.0, True, True)
            manager.cleanup()
            return output is not None
        except Exception as e:
            self.logger.error(f"基础功能测试失败: {e}")
            return False

    def _test_state_management(self) -> bool:
        """状态管理测试"""
        try:
            manager = RealtimeSPPVTStateManager()

            # 运行多步仿真
            for i in range(5):
                output = manager.run_single_step_simulation(0.5 + i * 0.1, 15.0, True, True)
                if output is None:
                    manager.cleanup()
                    return False

            stats = manager.get_performance_stats()
            manager.cleanup()
            return stats['total_simulations'] == 5
        except Exception as e:
            self.logger.error(f"状态管理测试失败: {e}")
            return False

    def _test_performance(self) -> bool:
        """性能测试"""
        try:
            manager = RealtimeSPPVTStateManager()

            start_time = time.time()
            for i in range(20):
                output = manager.run_single_step_simulation(1.0, 15.0, True, True)
                if output is None:
                    manager.cleanup()
                    return False

            avg_time = (time.time() - start_time) / 20
            manager.cleanup()
            return avg_time < 0.2  # 每次调用少于200ms
        except Exception as e:
            self.logger.error(f"性能测试失败: {e}")
            return False

    def _test_error_handling(self) -> bool:
        """错误处理测试"""
        try:
            manager = RealtimeSPPVTStateManager()

            # 测试异常参数
            output1 = manager.run_single_step_simulation(float('inf'), 15.0, True, True)
            output2 = manager.run_single_step_simulation(1.0, -10.0, True, True)

            manager.cleanup()
            # 应该有错误恢复机制，不应该崩溃
            return True
        except Exception as e:
            self.logger.error(f"错误处理测试失败: {e}")
            return False

    def _test_full_integration(self) -> bool:
        """完整集成测试 - 14/15-field版本"""
        try:
            interface = create_decision_sppvt_interface(debug=False)

            # 14-field输入测试
            test_input = {
                # 原始11个字段
                'ego_speed_kmh': 45.0,
                'ego_speed_ms': 12.5,
                'command_type': 1,
                'command_active': True,
                'manual_throttle_active': False,
                'control_error': -1.5,
                'control_mode_flag': 1,
                'V_target_kmh': 50.0,
                'V_min_kmh': 30.0,
                'G2_s': 2.0,
                'timestamp': time.time(),

                # 新增3个外部状态字段
                'external_stage_offset': 0.0,
                'external_stage_manager_states': [0.0, 0.0, 0.0],
                'external_adapter_states': [0.0, 13.89, 0.1]
            }

            result = interface.process_decision_and_control(test_input)

            # 验证15-field输出
            return ('target_accel' in result and result['target_accel'] is not None and
                    'new_stage_offset' in result and 'new_stage_manager_states' in result and
                    'new_adapter_states' in result)
        except Exception as e:
            self.logger.error(f"完整集成测试失败: {e}")
            return False

    def _generate_test_report(self, test_results: Dict[str, bool]):
        """生成测试报告"""
        print("\n" + "="*60)
        print("实时Carla-Simulink集成测试报告")
        print("="*60)

        total_tests = len(test_results)
        passed_tests = sum(test_results.values())

        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<25}: {status}")

        print("-"*60)
        print(f"总计测试: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {total_tests - passed_tests}")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%")

        if passed_tests == total_tests:
            print("\n🎉 所有测试通过！系统可用于生产环境。")
        else:
            print(f"\n⚠️ 有{total_tests - passed_tests}个测试失败，请检查配置。")

        print("="*60)


def create_visualization_test():
    """创建可视化测试 - 14/15-field版本"""
    if not MODULES_AVAILABLE:
        print("模块不可用，跳过可视化测试")
        return

    try:
        # 使用14/15-field接口进行测试
        interface = create_decision_sppvt_interface(debug=False)

        # 收集测试数据
        steps = []
        errors = []
        outputs = []
        stage_offsets = []
        state_continuity = []  # 记录状态连续性

        print("运行可视化测试 (14/15-field版本)...")

        # 初始状态
        current_stage_offset = 0.0
        current_stage_manager_states = [0.0, 0.0, 0.0]
        current_adapter_states = [0.0, 13.89, 0.1]

        for step in range(50):
            # 模拟变化的控制场景
            t = step * 0.05  # 时间
            control_error = 1.0 * np.sin(t) + 0.5 * np.cos(2*t)  # 周期性误差
            ego_speed_kmh = 45.0 + 10.0 * np.sin(0.1*t)  # 缓慢变化的速度
            ego_speed_ms = ego_speed_kmh / 3.6

            # 构建14-field输入
            test_input = {
                'ego_speed_kmh': ego_speed_kmh,
                'ego_speed_ms': ego_speed_ms,
                'command_type': 1,
                'command_active': True,
                'manual_throttle_active': False,
                'control_error': control_error,
                'control_mode_flag': 1,
                'V_target_kmh': 50.0,
                'V_min_kmh': 30.0,
                'G2_s': 2.0,
                'timestamp': time.time(),
                'external_stage_offset': current_stage_offset,
                'external_stage_manager_states': current_stage_manager_states,
                'external_adapter_states': current_adapter_states
            }

            result = interface.process_decision_and_control(test_input)

            if result is not None:
                target_accel = result.get('target_accel', 0.0)

                # 更新状态用于下一步
                current_stage_offset = result.get('new_stage_offset', current_stage_offset)
                current_stage_manager_states = result.get('new_stage_manager_states', current_stage_manager_states)
                current_adapter_states = result.get('new_adapter_states', current_adapter_states)

                steps.append(step)
                errors.append(control_error)
                outputs.append(target_accel)
                stage_offsets.append(current_stage_offset)

                # 记录状态连续性数据
                state_continuity.append({
                    'state': result.get('current_state', 0),
                    'decision': result.get('current_decision', 0),
                    'control_enabled': result.get('control_enabled', False)
                })

        # 创建可视化图表 - 扩展为3x2布局以显示更多信息
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))

        # 控制误差
        ax1.plot(steps, errors, 'b-', label='控制误差', linewidth=2)
        ax1.set_title('控制误差变化 (14/15-field版本)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('步数')
        ax1.set_ylabel('误差 (m)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # SPPVT输出
        ax2.plot(steps, outputs, 'r-', label='SPPVT目标加速度', linewidth=2)
        ax2.set_title('SPPVT控制输出')
        ax2.set_xlabel('步数')
        ax2.set_ylabel('加速度 (m/s²)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 级差变化
        ax3.plot(steps, stage_offsets, 'g-', label='级差 (Stage Offset)', linewidth=2)
        ax3.set_title('SPPVT级差变化 (状态外化)')
        ax3.set_xlabel('步数')
        ax3.set_ylabel('级差')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 误差vs输出散点图
        ax4.scatter(errors, outputs, alpha=0.6, c=steps, cmap='viridis')
        ax4.set_title('控制误差 vs SPPVT输出')
        ax4.set_xlabel('控制误差 (m)')
        ax4.set_ylabel('SPPVT输出 (m/s²)')
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax4.scatter(errors, outputs, alpha=0.6, c=steps, cmap='viridis'), ax=ax4)
        cbar.set_label('步数')

        # 状态连续性 - 决策状态
        if state_continuity:
            states = [s['state'] for s in state_continuity]
            decisions = [s['decision'] for s in state_continuity]

            ax5.plot(steps, states, 'o-', label='状态 (S0-S3)', markersize=4)
            ax5.set_title('决策状态连续性')
            ax5.set_xlabel('步数')
            ax5.set_ylabel('状态编号')
            ax5.set_ylim(-0.5, 3.5)
            ax5.grid(True, alpha=0.3)
            ax5.legend()

            # 控制使能状态
            control_enabled = [1 if s['control_enabled'] else 0 for s in state_continuity]
            ax6.plot(steps, control_enabled, 's-', label='控制使能', markersize=4, color='purple')
            ax6.plot(steps, decisions, '^-', label='决策 (R0-R8)', markersize=3, color='orange', alpha=0.7)
            ax6.set_title('控制使能和决策状态')
            ax6.set_xlabel('步数')
            ax6.set_ylabel('状态值')
            ax6.grid(True, alpha=0.3)
            ax6.legend()

        plt.tight_layout()
        plt.savefig('realtime_sppvt_14_15_field_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("✅ 可视化测试完成 (14/15-field版本)，结果保存为 realtime_sppvt_14_15_field_test_results.png")
        print(f"   📊 测试步数: {len(steps)}")
        print(f"   🎯 级差变化范围: {min(stage_offsets):.3f} ~ {max(stage_offsets):.3f}")
        print(f"   🚀 输出范围: {min(outputs):.3f} ~ {max(outputs):.3f} m/s²")
        if state_continuity:
            unique_states = set(s['state'] for s in state_continuity)
            unique_decisions = set(s['decision'] for s in state_continuity)
            print(f"   🔄 涉及状态: {sorted(unique_states)}")
            print(f"   🎯 涉及决策: {sorted(unique_decisions)}")

    except Exception as e:
        print(f"可视化测试失败: {e}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("实时Carla-Simulink集成测试")
    print("="*50)

    if not MODULES_AVAILABLE:
        print("❌ 必需模块不可用，无法运行测试")
        exit(1)

    # 选择测试模式
    print("选择测试模式:")
    print("1. 单元测试 (unittest)")
    print("2. 综合验证测试")
    print("3. 可视化测试")
    print("4. 运行所有测试")

    choice = input("请输入选择 (1-4): ").strip()

    if choice == "1":
        # 运行单元测试
        unittest.main(argv=[''], exit=False, verbosity=2)

    elif choice == "2":
        # 运行综合验证测试
        suite = IntegrationValidationSuite()
        suite.run_comprehensive_test()

    elif choice == "3":
        # 运行可视化测试
        create_visualization_test()

    elif choice == "4":
        # 运行所有测试
        print("\n1. 运行单元测试...")
        unittest.main(argv=[''], exit=False, verbosity=1)

        print("\n2. 运行综合验证测试...")
        suite = IntegrationValidationSuite()
        suite.run_comprehensive_test()

        print("\n3. 运行可视化测试...")
        create_visualization_test()

    else:
        print("无效选择，退出")

    print("\n测试完成！")