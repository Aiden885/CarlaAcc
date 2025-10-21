#!/usr/bin/env python
"""
横向控制调试工具
用于诊断Stanley控制器和CARLA感知的问题
"""

import math


class LateralControlDebugger:
    """横向控制调试器"""

    def __init__(self):
        self.history = []
        self.max_history = 100

    def log_frame(self, frame_num, ego_vehicle, carla_perception, lateral_controller,
                  steer_output, applied_steer):
        """
        记录每帧的完整状态

        Args:
            frame_num: 帧编号
            ego_vehicle: 自车对象
            carla_perception: 感知模块
            lateral_controller: Stanley控制器
            steer_output: 控制器计算的转向输出
            applied_steer: 实际应用到车辆的转向
        """
        from vehicle_utils import VehicleUtils

        # 获取所有状态
        heading_error = carla_perception.get_heading_error()
        cross_track_error = carla_perception.get_front_axle_offset()
        velocity_kmh = VehicleUtils.get_vehicle_speed(ego_vehicle)
        velocity_ms = velocity_kmh / 3.6

        # 获取控制器内部状态
        controller_state = lateral_controller.get_state()

        # 记录状态
        state = {
            'frame': frame_num,
            'velocity_kmh': velocity_kmh,
            'velocity_ms': velocity_ms,
            'heading_error_rad': heading_error,
            'heading_error_deg': math.degrees(heading_error),
            'cross_track_error_m': cross_track_error,
            'steer_output_rad': steer_output,
            'steer_output_deg': math.degrees(steer_output),
            'applied_steer_rad': applied_steer,
            'applied_steer_deg': math.degrees(applied_steer),
            'k': controller_state['k'],
            'k_soft': controller_state['k_soft'],
            'heading_term_deg': controller_state['last_heading_term_deg'],
            'crosstrack_term_deg': controller_state['last_crosstrack_term_deg']
        }

        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return state

    def print_diagnosis(self, state):
        """
        打印诊断信息

        Args:
            state: 当前帧状态
        """
        print(f"\n{'='*80}")
        print(f"帧 {state['frame']:4d} | 速度: {state['velocity_kmh']:5.1f}km/h ({state['velocity_ms']:4.1f}m/s)")
        print(f"{'='*80}")

        # 航向误差
        print(f"航向误差: {state['heading_error_deg']:+7.2f}° ", end="")
        if abs(state['heading_error_deg']) < 1.0:
            print("✓ 正常")
        elif abs(state['heading_error_deg']) < 5.0:
            print("⚠ 轻微偏离")
        else:
            print("❌ 严重偏离")

        # 横向误差
        print(f"横向误差: {state['cross_track_error_m']:+7.3f}m  ", end="")
        if abs(state['cross_track_error_m']) < 0.1:
            print("✓ 居中")
        elif abs(state['cross_track_error_m']) < 0.5:
            print("⚠ 轻微偏离")
        else:
            print("❌ 严重偏离")

        # Stanley控制器分解
        print(f"\nStanley控制器分解：")
        print(f"  - 航向修正项: {state['heading_term_deg']:+7.2f}°")
        print(f"  - 横向修正项: {state['crosstrack_term_deg']:+7.2f}°")
        print(f"  - 总输出:     {state['steer_output_deg']:+7.2f}°")
        print(f"  - 实际应用:   {state['applied_steer_deg']:+7.2f}°")

        # 检查问题
        issues = []

        # 问题1：输出与应用不一致
        if abs(state['steer_output_deg'] - state['applied_steer_deg']) > 0.5:
            issues.append(f"⚠️ 输出({state['steer_output_deg']:.1f}°)与应用({state['applied_steer_deg']:.1f}°)不一致")

        # 问题2：横向误差大但修正小
        if abs(state['cross_track_error_m']) > 0.5 and abs(state['crosstrack_term_deg']) < 2.0:
            issues.append(f"⚠️ 横向误差大({state['cross_track_error_m']:.2f}m)但修正项小({state['crosstrack_term_deg']:.1f}°)")

        # 问题3：航向误差大但修正小
        if abs(state['heading_error_deg']) > 5.0 and abs(state['heading_term_deg']) < 3.0:
            issues.append(f"⚠️ 航向误差大({state['heading_error_deg']:.1f}°)但修正项小({state['heading_term_deg']:.1f}°)")

        # 问题4：输出饱和
        max_output = math.degrees(state['k'] * 0.4)  # 假设limits是0.4
        if abs(state['steer_output_deg']) > max_output * 0.9:
            issues.append(f"⚠️ 输出接近饱和({state['steer_output_deg']:.1f}° / ±{max_output:.1f}°)")

        # 问题5：符号反向
        if state['cross_track_error_m'] > 0.3 and state['steer_output_deg'] > 0:
            issues.append("❌ 可能符号错误：车偏右(+)但转向也是右(+)")
        elif state['cross_track_error_m'] < -0.3 and state['steer_output_deg'] < 0:
            issues.append("❌ 可能符号错误：车偏左(-)但转向也是左(-)")

        if issues:
            print(f"\n发现问题：")
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"\n✓ 未发现明显问题")

        print(f"{'='*80}\n")

    def analyze_trend(self):
        """
        分析历史趋势

        Returns:
            dict: 趋势分析结果
        """
        if len(self.history) < 10:
            return {"error": "数据不足，至少需要10帧"}

        recent = self.history[-10:]

        # 检查横向误差趋势
        cte_values = [s['cross_track_error_m'] for s in recent]
        cte_trend = cte_values[-1] - cte_values[0]

        # 检查转向输出趋势
        steer_values = [s['steer_output_deg'] for s in recent]
        steer_variance = max(steer_values) - min(steer_values)

        analysis = {
            'cte_start': cte_values[0],
            'cte_end': cte_values[-1],
            'cte_trend': cte_trend,
            'steer_variance': steer_variance,
            'avg_velocity': sum([s['velocity_kmh'] for s in recent]) / len(recent)
        }

        # 诊断
        if abs(cte_trend) > 0.2:
            if cte_trend > 0:
                analysis['diagnosis'] = "❌ 横向误差持续增大（向右偏离）"
            else:
                analysis['diagnosis'] = "❌ 横向误差持续增大（向左偏离）"
        elif steer_variance < 1.0:
            analysis['diagnosis'] = "⚠️ 转向输出几乎不变，控制器可能失效"
        else:
            analysis['diagnosis'] = "✓ 控制器响应正常"

        return analysis

    def print_trend_analysis(self):
        """打印趋势分析"""
        analysis = self.analyze_trend()

        if "error" in analysis:
            print(analysis["error"])
            return

        print(f"\n{'='*80}")
        print(f"趋势分析（最近10帧）")
        print(f"{'='*80}")
        print(f"横向误差变化: {analysis['cte_start']:+.3f}m → {analysis['cte_end']:+.3f}m (Δ{analysis['cte_trend']:+.3f}m)")
        print(f"转向输出方差: {analysis['steer_variance']:.2f}°")
        print(f"平均速度:     {analysis['avg_velocity']:.1f}km/h")
        print(f"\n诊断结果: {analysis['diagnosis']}")
        print(f"{'='*80}\n")


# 全局调试器实例
lateral_debugger = LateralControlDebugger()


def debug_lateral_control(frame_num, ego_vehicle, carla_perception, lateral_controller,
                           steer_output, applied_steer, print_interval=50):
    """
    便捷调试函数

    Args:
        frame_num: 帧编号
        ego_vehicle: 自车
        carla_perception: 感知模块
        lateral_controller: Stanley控制器
        steer_output: 控制器输出
        applied_steer: 应用的转向
        print_interval: 打印间隔（帧数）
    """
    state = lateral_debugger.log_frame(
        frame_num, ego_vehicle, carla_perception,
        lateral_controller, steer_output, applied_steer
    )

    # 定期打印诊断
    if frame_num % print_interval == 0:
        lateral_debugger.print_diagnosis(state)
        lateral_debugger.print_trend_analysis()


if __name__ == '__main__':
    print("横向控制调试工具")
    print("使用方法：")
    print("  from debug_lateral_control import debug_lateral_control")
    print("  debug_lateral_control(frame_num, ego_vehicle, carla_perception,")
    print("                        lateral_controller, steer_output, applied_steer)")