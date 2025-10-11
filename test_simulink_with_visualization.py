#!/usr/bin/env python3
"""
Simulink决策+控制可视化测试
生成详细的测试报告和可视化图表
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from test_simulink_decision_control import SimulinkDecisionControlTester

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class VisualizedSimulinkTester(SimulinkDecisionControlTester):
    """带可视化的Simulink测试器"""

    def __init__(self, debug=True):
        super().__init__(debug)
        self.state_history = []
        self.decision_history = []
        self.control_output_history = []
        self.time_history = []
        self.test_sequence = []

    def run_single_test(self, test_name, test_input, expected_state=None, expected_decision=None,
                       check_control_enabled=None, description="", preserve_state=False):
        """重写run_single_test以记录历史数据"""
        result = super().run_single_test(
            test_name, test_input, expected_state, expected_decision,
            check_control_enabled, description, preserve_state
        )

        # 判断测试是否成功
        success = True
        if expected_state is not None and result['current_state'] != expected_state:
            success = False
        if expected_decision is not None and result['current_decision'] != expected_decision:
            success = False
        if check_control_enabled is not None and result['control_enabled'] != check_control_enabled:
            success = False

        # 记录历史数据用于可视化
        self.state_history.append(result['current_state'])
        self.decision_history.append(result['current_decision'])
        self.control_output_history.append(result['target_accel'])
        self.time_history.append(len(self.time_history))
        self.test_sequence.append({
            'name': test_name,
            'input': test_input,
            'expected_state': expected_state,
            'expected_decision': expected_decision,
            'actual_state': result['current_state'],
            'actual_decision': result['current_decision'],
            'success': success
        })

        return result

    def generate_state_transition_diagram(self, output_file='simulink_state_transitions.png'):
        """生成状态转移图"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # 定义状态和决策的颜色映射
        state_colors = {0: 'green', 1: 'orange', 2: 'blue', 3: 'red'}
        state_names = {0: 'S0-在控', 1: 'S1-有史待命', 2: 'S2-无史待命', 3: 'S3-低速'}
        decision_names = {
            0: 'NONE', 1: 'R1-降速', 2: 'R2-增速', 3: 'R3-降距',
            4: 'R4-增距', 5: 'R5-无继控制', 6: 'R6-继承控制',
            7: 'R7-扭矩仲裁', 8: 'R8-待命'
        }

        # 绘制状态转移路径
        x = np.arange(len(self.state_history))
        ax.plot(x, self.state_history, 'o-', linewidth=2, markersize=8,
                label='状态序列', color='darkblue')

        # 标注每个测试点
        for i, (state, test) in enumerate(zip(self.state_history, self.test_sequence)):
            color = 'green' if test['success'] else 'red'
            ax.annotate(f"{state_names[state]}\n{decision_names[test['actual_decision']]}",
                       xy=(i, state), xytext=(0, 20),
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # 设置y轴为离散状态
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['S0-在控', 'S1-有史待命', 'S2-无史待命', 'S3-低速'])
        ax.set_xlabel('测试序列', fontsize=12)
        ax.set_ylabel('状态', fontsize=12)
        ax.set_title('Simulink状态转移测试序列', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"[OK] State transition diagram saved: {output_file}")
        plt.close()

    def generate_decision_distribution(self, output_file='simulink_decision_distribution.png'):
        """生成决策分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 决策分布柱状图
        decision_names = {
            0: 'NONE', 1: 'R1-降速', 2: 'R2-增速', 3: 'R3-降距',
            4: 'R4-增距', 5: 'R5-无继', 6: 'R6-继承',
            7: 'R7-仲裁', 8: 'R8-待命'
        }

        decision_counts = {}
        for decision in self.decision_history:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        decisions = list(decision_counts.keys())
        counts = list(decision_counts.values())

        colors = plt.cm.Set3(np.linspace(0, 1, len(decisions)))
        bars = ax1.bar([decision_names[d] for d in decisions], counts, color=colors)
        ax1.set_xlabel('决策类型', fontsize=12)
        ax1.set_ylabel('触发次数', fontsize=12)
        ax1.set_title('决策输出分布统计', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)

        # 控制输出时间序列
        ax2.plot(self.time_history, self.control_output_history, 'o-',
                linewidth=2, markersize=6, color='darkgreen', label='控制输出')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='上限 2.0 m/s²')
        ax2.axhline(y=-3.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='下限 -3.0 m/s²')

        ax2.set_xlabel('测试序列', fontsize=12)
        ax2.set_ylabel('加速度 (m/s²)', fontsize=12)
        ax2.set_title('SPPVT控制输出序列', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"✅ 决策分布图已保存: {output_file}")
        plt.close()

    def generate_test_coverage_matrix(self, output_file='simulink_test_coverage.png'):
        """生成测试覆盖矩阵"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # 创建状态转移矩阵
        states = [0, 1, 2, 3]
        state_names = ['S0-在控', 'S1-有史待命', 'S2-无史待命', 'S3-低速']
        transition_matrix = np.zeros((4, 4))
        transition_labels = [['' for _ in range(4)] for _ in range(4)]

        # 统计状态转移
        for i in range(len(self.state_history) - 1):
            from_state = self.state_history[i]
            to_state = self.state_history[i+1]
            transition_matrix[from_state][to_state] += 1

            # 记录触发的指令
            test_info = self.test_sequence[i+1]
            command_type = test_info['input']['command_type']
            command_name = f"I{command_type-1}" if command_type > 0 else "NONE"
            if transition_labels[from_state][to_state]:
                transition_labels[from_state][to_state] += f"\n{command_name}"
            else:
                transition_labels[from_state][to_state] = command_name

        # 绘制热力图
        im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto')

        # 设置坐标轴
        ax.set_xticks(states)
        ax.set_yticks(states)
        ax.set_xticklabels(state_names, fontsize=10)
        ax.set_yticklabels(state_names, fontsize=10)
        ax.set_xlabel('目标状态', fontsize=12)
        ax.set_ylabel('源状态', fontsize=12)
        ax.set_title('状态转移覆盖矩阵', fontsize=14, fontweight='bold')

        # 在每个格子中添加转移次数和指令
        for i in range(4):
            for j in range(4):
                count = int(transition_matrix[i][j])
                if count > 0:
                    text = f"{count}次\n{transition_labels[i][j]}"
                    color = 'white' if count > 1 else 'black'
                    ax.text(j, i, text, ha="center", va="center",
                           color=color, fontsize=9, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('转移次数', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"✅ 测试覆盖矩阵已保存: {output_file}")
        plt.close()

    def generate_html_report(self, output_file='simulink_test_report.html'):
        """生成HTML测试报告"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        pass_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulink测试报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card.success {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }}
        .summary-card.failure {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}
        .summary-card.rate {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .summary-card h3 {{
            margin: 0;
            font-size: 36px;
            font-weight: bold;
        }}
        .summary-card p {{
            margin: 5px 0 0 0;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .pass {{
            color: green;
            font-weight: bold;
        }}
        .fail {{
            color: red;
            font-weight: bold;
        }}
        .test-details {{
            font-size: 12px;
            color: #666;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            text-align: right;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Simulink决策+控制测试报告</h1>

        <div class="summary">
            <div class="summary-card">
                <h3>{total_tests}</h3>
                <p>总测试数</p>
            </div>
            <div class="summary-card success">
                <h3>{passed_tests}</h3>
                <p>通过测试</p>
            </div>
            <div class="summary-card failure">
                <h3>{failed_tests}</h3>
                <p>失败测试</p>
            </div>
            <div class="summary-card rate">
                <h3>{pass_rate:.1f}%</h3>
                <p>通过率</p>
            </div>
        </div>

        <h2>📊 测试结果详情</h2>
        <table>
            <thead>
                <tr>
                    <th>序号</th>
                    <th>测试名称</th>
                    <th>状态</th>
                    <th>详细信息</th>
                </tr>
            </thead>
            <tbody>
"""

        for i, test in enumerate(self.test_results, 1):
            status_class = 'pass' if test['status'] == 'PASS' else 'fail'
            status_symbol = '✅' if test['status'] == 'PASS' else '❌'

            details = ""
            if test['status'] == 'FAIL' and 'errors' in test:
                details = "<br>".join(test['errors'])

            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{test['name']}</td>
                    <td class="{status_class}">{status_symbol} {test['status']}</td>
                    <td class="test-details">{details if details else '无错误'}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <h2>📈 可视化图表</h2>

        <div class="chart-container">
            <h3>状态转移测试序列</h3>
            <img src="simulink_state_transitions.png" alt="状态转移图">
        </div>

        <div class="chart-container">
            <h3>决策分布与控制输出</h3>
            <img src="simulink_decision_distribution.png" alt="决策分布图">
        </div>

        <div class="chart-container">
            <h3>状态转移覆盖矩阵</h3>
            <img src="simulink_test_coverage.png" alt="测试覆盖矩阵">
        </div>

        <p class="timestamp">报告生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
</body>
</html>
"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ HTML测试报告已保存: {output_file}")

    def run_all_tests_with_visualization(self):
        """运行所有测试并生成可视化报告"""
        print(f"\n{'#'*70}")
        print(f"# Simulink 决策+控制完整验证测试 (带可视化)")
        print(f"{'#'*70}")

        start_time = time.time()

        # 运行所有测试组
        self.test_state_transitions()
        self.test_decision_outputs()
        self.test_sppvt_control()

        # 生成可视化报告
        print(f"\n{'='*70}")
        print(f"📊 生成可视化报告...")
        print(f"{'='*70}")

        self.generate_state_transition_diagram()
        self.generate_decision_distribution()
        self.generate_test_coverage_matrix()
        self.generate_html_report()

        # 打印总结
        elapsed_time = time.time() - start_time
        self.print_summary()
        print(f"\n⏱️  总耗时: {elapsed_time:.2f}秒")

        print(f"\n{'='*70}")
        print(f"✨ 测试完成! 请查看以下文件:")
        print(f"   📄 HTML报告: simulink_test_report.html")
        print(f"   📊 状态转移图: simulink_state_transitions.png")
        print(f"   📊 决策分布图: simulink_decision_distribution.png")
        print(f"   📊 覆盖矩阵: simulink_test_coverage.png")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"启动Simulink可视化测试框架")
    print(f"{'='*70}\n")

    tester = VisualizedSimulinkTester(debug=False)
    tester.run_all_tests_with_visualization()