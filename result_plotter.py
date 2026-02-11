"""Real-time result plotter with auto-save functionality."""

import threading
import queue
import time
from collections import deque
from datetime import datetime
import csv
import atexit
import signal
import sys

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Use the Tk backend because it plays nicely with separate threads
matplotlib.use("TkAgg")


class RealtimeResultPlotter:
    """Thread-safe real-time plotter for ACC results with auto-save."""

    def __init__(self, max_points: int = 5000, update_interval: int = 100) -> None:
        """Initialise the plotter.

        Args:
            max_points: Reserved parameter (not currently limiting data storage).
            update_interval: Animation refresh interval in milliseconds.
        """
        self.data_queue = queue.Queue(maxsize=1000)

        self.max_points = max_points
        # Store data with maxlen limit to prevent memory issues
        self.steps = deque(maxlen=max_points)
        self.desired_gaps = deque(maxlen=max_points)
        self.actual_gaps = deque(maxlen=max_points)
        self.errors = deque(maxlen=max_points)
        self.ego_speeds = deque(maxlen=max_points)
        self.target_speeds = deque(maxlen=max_points)
        self.inc_torques = deque(maxlen=max_points)
        self.inc_brake_torques = deque(maxlen=max_points)
        self.vehicle_distances = deque(maxlen=max_points)

        # ACC recording control
        self.acc_started = False  # Only start recording when ACC is enabled
        self.current_step = 0  # Step counter

        self.fig = None
        self.axes = None
        self.animation_obj = None
        self.update_interval = update_interval

        self.plot_thread = None
        self.running = False

        self.total_points = 0
        self.start_time = None

        # Save control - prevent duplicate saves
        self.results_saved = False

        # Register exit handlers to ensure data is saved even on abnormal exit
        atexit.register(self._emergency_save)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def add_data(self, desired_gap: float, actual_gap: float, timestamp: float,
                 ego_speed: float = 0.0, target_speed: float = 0.0,
                 control_enabled: bool = False,
                 request_torque: float = 0.0, request_brake_torque: float = 0.0,
                 incremental_torque: float = 0.0, incremental_brake_torque: float = 0.0,
                 vehicle_distance: float = 0.0) -> None:
        """Push a new sample into the queue from the producer thread.

        Args:
            desired_gap: 期望时距 (秒)
            actual_gap: 实际时距 (秒)
            timestamp: 时间戳 (秒) - not used in step mode
            ego_speed: 自车速度 (km/h)
            target_speed: 前车速度 (km/h)
            control_enabled: ACC控制是否开启
            request_torque: 请求发动机扭矩 (Nm, 正值，加速时，当前不再保存)
            request_brake_torque: 请求制动扭矩 (Nm, 正值，制动时，当前不再保存)
            incremental_torque: 增量控制扭矩 (Nm, 正值，加速时)
            incremental_brake_torque: 增量控制制动扭矩 (Nm, 正值，制动时)
            vehicle_distance: 两车距离 (m)
        """
        # Only start recording after ACC is enabled
        if not self.acc_started:
            if control_enabled:
                self.acc_started = True
                print("[ResultPlotter] ACC开启，开始记录数据...")
            else:
                return  # Skip data before ACC is enabled

        try:
            self.data_queue.put_nowait((desired_gap, actual_gap, ego_speed, target_speed, control_enabled,
                                        request_torque, request_brake_torque,
                                        incremental_torque, incremental_brake_torque, vehicle_distance))
        except queue.Full:
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait((desired_gap, actual_gap, ego_speed, target_speed, control_enabled,
                                            request_torque, request_brake_torque,
                                            incremental_torque, incremental_brake_torque, vehicle_distance))
            except queue.Empty:
                pass

    def start(self) -> None:
        """Start the background plotting thread."""
        if self.running:
            print("[ResultPlotter] Already running.")
            return

        self.running = True
        self.start_time = time.time()
        self.plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
        self.plot_thread.start()
        print("[ResultPlotter] Started (Result Mode - Step based).")

    def stop(self) -> None:
        """Stop the plotting thread, save results, and close the figure window."""
        self.running = False
        if self.plot_thread:
            self.plot_thread.join(timeout=2.0)

        # Auto-save results
        self.save_results()

        if self.fig:
            plt.close(self.fig)
        print("[ResultPlotter] Stopped.")

    def cleanup(self) -> None:
        """Cleanup resources (called by ResourceManager)."""
        self.stop()

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals (Ctrl+C, etc.)"""
        print(f"\n[ResultPlotter] 捕获到退出信号 ({signum})，正在保存结果...")
        self._emergency_save()
        sys.exit(0)

    def _on_window_close(self, event):
        """Handle matplotlib window close event"""
        print("\n[ResultPlotter] 检测到窗口关闭，正在保存结果...")
        self._emergency_save()
        self.running = False

    def _emergency_save(self):
        """Emergency save function called on abnormal exit"""
        if not self.results_saved:
            self.save_results()

    def _save_figure_by_redraw(self, png_filename: str) -> None:
        """重新绘制图形并保存，避免TkAgg后端的保存问题"""
        import matplotlib.pyplot as plt

        # 转换数据为numpy数组
        steps = np.asarray(self.steps, dtype=float)
        desired = np.asarray(self.desired_gaps, dtype=float)
        actual = np.asarray(self.actual_gaps, dtype=float)
        errors = np.asarray(self.errors, dtype=float)
        ego_speeds = np.asarray(self.ego_speeds, dtype=float)
        target_speeds = np.asarray(self.target_speeds, dtype=float)
        inc_torques = np.asarray(self.inc_torques, dtype=float)
        inc_brake_torques = np.asarray(self.inc_brake_torques, dtype=float)
        vehicle_distances = np.asarray(self.vehicle_distances, dtype=float)

        # MATLAB经典配色
        color_desired = "#0072BD"  # MATLAB蓝色
        color_actual = "#D95319"   # MATLAB橙色
        color_error = "#EDB120"    # MATLAB黄色
        color_ego = "#7E2F8E"      # MATLAB紫色
        color_target = "#77AC30"   # MATLAB绿色

        bg_color = "#ffffff"
        grid_color = "#d9d9d9"
        text_color = "#000000"

        # 创建新的figure用于保存（使用Agg后端，不显示）
        fig = plt.figure(figsize=(14, 14), facecolor=bg_color)

        # 使用subplot创建子图，便于自动布局
        ax1 = fig.add_subplot(6, 1, 1, facecolor=bg_color)
        ax2 = fig.add_subplot(6, 1, 2, facecolor=bg_color)
        ax3 = fig.add_subplot(6, 1, 3, facecolor=bg_color)
        ax4 = fig.add_subplot(6, 1, 4, facecolor=bg_color)
        ax5 = fig.add_subplot(6, 1, 5, facecolor=bg_color)
        ax6 = fig.add_subplot(6, 1, 6, facecolor=bg_color)

        # 配置所有子图的样式
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_facecolor(bg_color)
            ax.grid(True, color=grid_color, linestyle="-", linewidth=0.7, alpha=0.8)
            ax.tick_params(colors=text_color, labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#000000")
                spine.set_linewidth(1.2)

        # 子图1：期望 vs 实际时距
        ax1.plot(steps, desired, color=color_desired, linewidth=2.0,
                label="Desired Time Gap", marker='o', markersize=2, markevery=10)
        ax1.plot(steps, actual, color=color_actual, linewidth=2.0,
                label="Actual Time Gap", marker='s', markersize=2, markevery=10)
        ax1.set_ylabel("Time Gap (s)", color=text_color, fontsize=11, fontweight='bold')
        ax1.set_title("Time Gap Tracking", color=text_color, fontsize=13, fontweight="bold")
        ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000",
                  fontsize=10, framealpha=1.0)

        # 子图2：误差
        ax2.plot(steps, errors, color=color_error, linewidth=2.0,
                label="Time Gap Error", marker='d', markersize=2, markevery=10)
        # 添加零位参考线（虚线）
        ax2.axhline(y=0, color='#808080', linestyle='--', linewidth=1.5, alpha=0.8, label="Zero Reference")
        ax2.set_ylabel("Error (s)", color=text_color, fontsize=11, fontweight='bold')
        ax2.set_title("Time Gap Error (Actual - Desired)", color=text_color,
                     fontsize=13, fontweight="bold")
        ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000",
                  fontsize=10, framealpha=1.0)

        # 子图3：速度
        ax3.plot(steps, ego_speeds, color=color_ego, linewidth=2.0,
                label="Ego Speed", marker='v', markersize=2, markevery=10)
        ax3.plot(steps, target_speeds, color=color_target, linewidth=2.0,
                label="Target Speed", marker='^', markersize=2, markevery=10)
        ax3.set_ylabel("Speed (km/h)", color=text_color, fontsize=11, fontweight='bold')
        ax3.set_title("Vehicle Speeds", color=text_color, fontsize=13, fontweight="bold")
        ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000",
                  fontsize=10, framealpha=1.0)

        # 子图4：增量扭矩
        ax4.plot(steps, inc_torques, color="#2ca02c", linewidth=2.0, linestyle="--",
                label="Inc Torque", marker='x', markersize=2, markevery=10)
        ax4.axhline(y=0, color="#000000", linestyle="--", linewidth=1.5, alpha=0.7)
        ax4.set_ylabel("Torque (Nm)", color=text_color, fontsize=11, fontweight='bold')
        ax4.set_title("Incremental Engine Torque", color=text_color, fontsize=13, fontweight="bold")
        ax4.legend(loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000",
                  fontsize=10, framealpha=1.0)

        # 子图5：增量制动扭矩
        ax5.plot(steps, inc_brake_torques, color="#9467bd", linewidth=2.0, linestyle="--",
                label="Inc Brake Torque", marker='x', markersize=2, markevery=10)
        ax5.axhline(y=0, color="#000000", linestyle="--", linewidth=1.5, alpha=0.7)
        ax5.set_ylabel("Brake Torque (Nm)", color=text_color, fontsize=11, fontweight='bold')
        ax5.set_title("Incremental Brake Torque", color=text_color, fontsize=13, fontweight="bold")
        ax5.legend(loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000",
                  fontsize=10, framealpha=1.0)

        # 子图6：两车距离
        ax6.plot(steps, vehicle_distances, color="#17becf", linewidth=2.0,
                label="Vehicle Distance", marker='o', markersize=2, markevery=10)
        ax6.set_xlabel("Step", color=text_color, fontsize=11, fontweight='bold')
        ax6.set_ylabel("Distance (m)", color=text_color, fontsize=11, fontweight='bold')
        ax6.set_title("Vehicle Distance", color=text_color, fontsize=13, fontweight="bold")
        ax6.legend(loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000",
                  fontsize=10, framealpha=1.0)

        # 自动调整布局
        fig.tight_layout()

        # 保存图像
        fig.savefig(png_filename, dpi=150, facecolor='white', bbox_inches='tight')

        # 关闭figure释放内存
        plt.close(fig)

    def save_results(self) -> None:
        """Save all recorded data to CSV and PNG files with timestamp."""
        # Prevent duplicate saves
        if self.results_saved:
            return

        print(f"\n[ResultPlotter] 开始保存结果...")
        print(f"[ResultPlotter] ACC是否已开启: {self.acc_started}")
        print(f"[ResultPlotter] 记录的数据点数: {len(self.steps)}")

        if not self.steps:
            print("[ResultPlotter] ⚠️  没有数据可保存！")
            if not self.acc_started:
                print("[ResultPlotter] 原因: ACC未开启，没有记录任何数据")
                print("[ResultPlotter] 提示: 需要先按空格键开启ACC，再按E/Q键启动ACC控制")
            return

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"acc_results_{timestamp}.csv"
        png_filename = f"acc_results_{timestamp}.png"

        print(f"[ResultPlotter] 准备保存到: {csv_filename} 和 {png_filename}")

        # Save CSV
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['Step', 'Desired Gap (s)', 'Actual Gap (s)',
                                 'Error (s)', 'Ego Speed (km/h)', 'Target Speed (km/h)',
                                 'Incremental Torque (Nm)', 'Incremental Brake Torque (Nm)',
                                 'Vehicle Distance (m)'])
                # Write data
                for i in range(len(self.steps)):
                    writer.writerow([
                        self.steps[i],
                        self.desired_gaps[i],
                        self.actual_gaps[i],
                        self.errors[i],
                        self.ego_speeds[i],
                        self.target_speeds[i],
                        self.inc_torques[i],
                        self.inc_brake_torques[i],
                        self.vehicle_distances[i]
                    ])
            print(f"✅ CSV数据已保存: {csv_filename}")
        except Exception as e:
            print(f"❌ CSV保存失败: {e}")

        # Save PNG - 重新绘制以避免TkAgg后端的保存问题
        print(f"[ResultPlotter] 准备重新绘制图形以保存...")
        try:
            self._save_figure_by_redraw(png_filename)
            print(f"✅ 图像已保存: {png_filename}")
        except Exception as e:
            print(f"❌ 图像保存失败: {e}")
            import traceback
            traceback.print_exc()

        # Mark as saved to prevent duplicate saves
        self.results_saved = True

    def _setup_matlab_style(self) -> None:
        """Configure axes, colours, and legend to mimic MATLAB defaults."""
        plt.style.use("default")

        bg_color = "#ffffff"
        grid_color = "#d9d9d9"
        text_color = "#000000"

        # MATLAB经典配色
        self.color_desired = "#0072BD"  # MATLAB蓝色
        self.color_actual = "#D95319"   # MATLAB橙色
        self.color_error = "#EDB120"    # MATLAB黄色
        self.color_ego = "#7E2F8E"      # MATLAB紫色 - 自车
        self.color_target = "#77AC30"   # MATLAB绿色 - 前车

        self.fig = plt.figure(figsize=(14, 14), facecolor=bg_color)
        self.fig.canvas.manager.set_window_title(
            "ACC Results - Step Mode"
        )

        # Register window close event to save results
        self.fig.canvas.mpl_connect('close_event', self._on_window_close)

        # 6个子图：时距跟踪、误差、速度、扭矩、制动扭矩、两车距离
        h = 0.11    # 每个子图高度
        g = 0.04    # 子图间距
        b = 0.04    # 底部起始
        w = 0.78    # 子图宽度
        l = 0.08    # 左边距
        self.ax1 = plt.axes([l, b + 5*(h+g), w, h], facecolor=bg_color)  # 时距跟踪
        self.ax2 = plt.axes([l, b + 4*(h+g), w, h], facecolor=bg_color)  # 误差
        self.ax3 = plt.axes([l, b + 3*(h+g), w, h], facecolor=bg_color)  # 速度
        self.ax4 = plt.axes([l, b + 2*(h+g), w, h], facecolor=bg_color)  # 加速扭矩
        self.ax5 = plt.axes([l, b + 1*(h+g), w, h], facecolor=bg_color)  # 制动扭矩
        self.ax6 = plt.axes([l, b + 0*(h+g), w, h], facecolor=bg_color)  # 两车距离

        self.axes = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]

        for ax in self.axes:
            ax.set_facecolor(bg_color)
            ax.grid(True, color=grid_color, linestyle="-", linewidth=0.7, alpha=0.8)
            ax.tick_params(colors=text_color, labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#000000")
                spine.set_linewidth(1.2)

        # 子图1：期望 vs 实际时距
        self.line_desired, = self.ax1.plot(
            [], [], color=self.color_desired, linewidth=2.0, label="Desired Time Gap", marker='o', markersize=2, markevery=10
        )
        self.line_actual, = self.ax1.plot(
            [], [], color=self.color_actual, linewidth=2.0, label="Actual Time Gap", marker='s', markersize=2, markevery=10
        )
        self.ax1.set_ylabel("Time Gap (s)", color=text_color, fontsize=11, fontweight='bold')
        self.ax1.set_title(
            "Time Gap Tracking", color=text_color, fontsize=13, fontweight="bold"
        )
        self.ax1.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000", fontsize=10, framealpha=1.0
        )

        # 子图2：误差
        self.line_error, = self.ax2.plot(
            [], [], color=self.color_error, linewidth=2.0, label="Time Gap Error", marker='d', markersize=2, markevery=10
        )
        # 添加零位参考线（虚线）
        self.ax2.axhline(y=0, color='#808080', linestyle='--', linewidth=1.5, alpha=0.8, label="Zero Reference")
        self.ax2.set_ylabel("Error (s)", color=text_color, fontsize=11, fontweight='bold')
        self.ax2.set_title(
            "Time Gap Error (Actual - Desired)", color=text_color, fontsize=13, fontweight="bold"
        )
        self.ax2.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000", fontsize=10, framealpha=1.0
        )

        # 子图3：速度
        self.line_ego_speed, = self.ax3.plot(
            [], [], color=self.color_ego, linewidth=2.0, label="Ego Speed", marker='v', markersize=2, markevery=10
        )
        self.line_target_speed, = self.ax3.plot(
            [], [], color=self.color_target, linewidth=2.0, label="Target Speed", marker='^', markersize=2, markevery=10
        )
        self.ax3.set_ylabel("Speed (km/h)", color=text_color, fontsize=11, fontweight='bold')
        self.ax3.set_title(
            "Vehicle Speeds", color=text_color, fontsize=13, fontweight="bold"
        )
        self.ax3.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000", fontsize=10, framealpha=1.0
        )

        # 子图4：增量扭矩
        self.line_torque_inc, = self.ax4.plot(
            [], [], color="#2ca02c", linewidth=2.0, linestyle="--",
            label="Inc Torque", marker='x', markersize=2, markevery=10
        )
        self.ax4.axhline(y=0, color="#000000", linestyle="--", linewidth=1.5, alpha=0.7)
        self.ax4.set_ylabel("Torque (Nm)", color=text_color, fontsize=11, fontweight='bold')
        self.ax4.set_title(
            "Incremental Engine Torque", color=text_color, fontsize=13, fontweight="bold"
        )
        self.ax4.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000", fontsize=10, framealpha=1.0
        )

        # 子图5：增量制动扭矩
        self.line_brake_torque_inc, = self.ax5.plot(
            [], [], color="#9467bd", linewidth=2.0, linestyle="--",
            label="Inc Brake Torque", marker='x', markersize=2, markevery=10
        )
        self.ax5.axhline(y=0, color="#000000", linestyle="--", linewidth=1.5, alpha=0.7)
        self.ax5.set_ylabel("Brake Torque (Nm)", color=text_color, fontsize=11, fontweight='bold')
        self.ax5.set_title(
            "Incremental Brake Torque", color=text_color, fontsize=13, fontweight="bold"
        )
        self.ax5.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000", fontsize=10, framealpha=1.0
        )

        # 子图6：两车距离
        self.line_distance, = self.ax6.plot(
            [], [], color="#17becf", linewidth=2.0, label="Vehicle Distance", marker='o', markersize=2, markevery=10
        )
        self.ax6.set_xlabel("Step", color=text_color, fontsize=11, fontweight='bold')
        self.ax6.set_ylabel("Distance (m)", color=text_color, fontsize=11, fontweight='bold')
        self.ax6.set_title(
            "Vehicle Distance", color=text_color, fontsize=13, fontweight="bold"
        )
        self.ax6.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1), facecolor=bg_color, edgecolor="#000000", fontsize=10, framealpha=1.0
        )

    def _plot_loop(self) -> None:
        """Own the matplotlib event loop on a dedicated thread."""
        try:
            self._setup_matlab_style()

            self.animation_obj = animation.FuncAnimation(
                self.fig,
                self._update_plot,
                interval=self.update_interval,
                blit=False,
                cache_frame_data=False,
            )

            plt.show()
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[ResultPlotter] Plotting thread error: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False

    def _update_plot(self, _frame: int):
        """Animation callback that refreshes all visible artists."""
        if not self.running:
            return ()

        data_count = 0
        while not self.data_queue.empty() and data_count < 50:
            try:
                item = self.data_queue.get_nowait()
                if len(item) == 10:
                    desired, actual, ego_speed, target_speed, control_enabled, req_torque, req_brake_torque, inc_torque, inc_brake_torque, vehicle_distance = item
                elif len(item) == 9:
                    desired, actual, ego_speed, target_speed, control_enabled, req_torque, req_brake_torque, inc_torque, inc_brake_torque = item
                    vehicle_distance = 0.0
                elif len(item) == 7:
                    desired, actual, ego_speed, target_speed, control_enabled, req_torque, req_brake_torque = item
                    inc_torque, inc_brake_torque = 0.0, 0.0
                else:
                    # 兼容旧格式
                    desired, actual, ego_speed, target_speed, control_enabled = item
                    req_torque, req_brake_torque = 0.0, 0.0
                    inc_torque, inc_brake_torque = 0.0, 0.0
                    vehicle_distance = 0.0
            except queue.Empty:
                break

            # Add data with step counter
            self.steps.append(self.current_step)
            self.desired_gaps.append(desired)
            self.actual_gaps.append(actual)
            self.errors.append(actual - desired)
            self.ego_speeds.append(ego_speed)
            self.target_speeds.append(target_speed)
            self.inc_torques.append(inc_torque)
            self.inc_brake_torques.append(inc_brake_torque)
            self.vehicle_distances.append(vehicle_distance)

            self.current_step += 1
            self.total_points += 1
            data_count += 1

        if not self.steps:
            return ()

        steps = np.asarray(self.steps, dtype=float)
        desired = np.asarray(self.desired_gaps, dtype=float)
        actual = np.asarray(self.actual_gaps, dtype=float)
        errors = np.asarray(self.errors, dtype=float)
        ego_speeds = np.asarray(self.ego_speeds, dtype=float)
        target_speeds = np.asarray(self.target_speeds, dtype=float)

        # 转换为numpy数组
        inc_torques_array = np.asarray(self.inc_torques, dtype=float)
        inc_brake_torques_array = np.asarray(self.inc_brake_torques, dtype=float)
        vehicle_distances_array = np.asarray(self.vehicle_distances, dtype=float)

        # 更新所有曲线数据
        self.line_desired.set_data(steps, desired)
        self.line_actual.set_data(steps, actual)
        self.line_error.set_data(steps, errors)
        self.line_ego_speed.set_data(steps, ego_speeds)
        self.line_target_speed.set_data(steps, target_speeds)
        self.line_torque_inc.set_data(steps, inc_torques_array)
        self.line_brake_torque_inc.set_data(steps, inc_brake_torques_array)
        self.line_distance.set_data(steps, vehicle_distances_array)

        # Set X-axis limits
        x_left, x_right = 0, 1
        if steps.size:
            x_left = float(steps[0])
            x_right = float(steps[-1])

            # Add padding
            if x_right > x_left:
                padding = (x_right - x_left) * 0.02
                x_right += padding

            for axis in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6):
                axis.set_xlim(x_left, x_right)

        # Set Y-axis limits for time gap
        if desired.size and steps.size:
            y_min = float(np.nanmin([desired.min(), actual.min()]))
            y_max = float(np.nanmax([desired.max(), actual.max()]))

            if np.isfinite(y_min) and np.isfinite(y_max):
                if np.isclose(y_min, y_max):
                    span = max(abs(y_min) * 0.1, 0.1)
                else:
                    span = max((y_max - y_min) * 0.1, 0.1)
                lower = y_min - span
                upper = y_max + span
                if np.isclose(lower, upper):
                    upper = lower + 0.1
                self.ax1.set_ylim(lower, upper)
                self.ax1.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

        # Set error Y-axis limits
        if errors.size:
            err_abs_max = float(np.nanmax(np.abs(errors)))

            if not np.isfinite(err_abs_max):
                err_abs_max = 0.1
            err_abs_max = max(err_abs_max, 0.05) * 1.1
            self.ax2.set_ylim(-err_abs_max, err_abs_max)
            self.ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

        # Set speed Y-axis limits
        if ego_speeds.size and target_speeds.size:
            speed_min = float(np.nanmin([ego_speeds.min(), target_speeds.min()]))
            speed_max = float(np.nanmax([ego_speeds.max(), target_speeds.max()]))

            if np.isfinite(speed_min) and np.isfinite(speed_max):
                if np.isclose(speed_min, speed_max):
                    span = max(abs(speed_min) * 0.1, 5.0)
                else:
                    span = max((speed_max - speed_min) * 0.1, 5.0)
                lower = max(0, speed_min - span)  # 速度不能为负
                upper = speed_max + span
                if np.isclose(lower, upper):
                    upper = lower + 10.0
                self.ax3.set_ylim(lower, upper)
                self.ax3.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

        # Set torque Y-axis limits (子图4)
        if inc_torques_array.size and steps.size:
            t_min = float(np.nanmin(inc_torques_array))
            t_max = float(np.nanmax(inc_torques_array))
            if np.isfinite(t_min) and np.isfinite(t_max):
                if np.isclose(t_min, t_max):
                    span = max(abs(t_min) * 0.1, 10.0)
                else:
                    span = max((t_max - t_min) * 0.1, 10.0)
                lower = max(0, t_min - span)  # 扭矩通常>=0
                upper = t_max + span
                if np.isclose(lower, upper):
                    upper = lower + 10.0
                self.ax4.set_ylim(lower, upper)
                self.ax4.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

        # Set brake torque Y-axis limits (子图5)
        if inc_brake_torques_array.size and steps.size:
            bt_min = float(np.nanmin(inc_brake_torques_array))
            bt_max = float(np.nanmax(inc_brake_torques_array))
            if np.isfinite(bt_min) and np.isfinite(bt_max):
                if np.isclose(bt_min, bt_max):
                    span = max(abs(bt_min) * 0.1, 10.0)
                else:
                    span = max((bt_max - bt_min) * 0.1, 10.0)
                lower = max(0, bt_min - span)  # 制动扭矩通常>=0
                upper = bt_max + span
                if np.isclose(lower, upper):
                    upper = lower + 10.0
                self.ax5.set_ylim(lower, upper)
                self.ax5.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

        # Set distance Y-axis limits (子图6)
        if vehicle_distances_array.size and steps.size:
            d_min = float(np.nanmin(vehicle_distances_array))
            d_max = float(np.nanmax(vehicle_distances_array))
            if np.isfinite(d_min) and np.isfinite(d_max):
                if np.isclose(d_min, d_max):
                    span = max(abs(d_min) * 0.1, 5.0)
                else:
                    span = max((d_max - d_min) * 0.1, 5.0)
                lower = max(0, d_min - span)
                upper = d_max + span
                if np.isclose(lower, upper):
                    upper = lower + 10.0
                self.ax6.set_ylim(lower, upper)
                self.ax6.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

        return (self.line_desired, self.line_actual, self.line_error,
                self.line_ego_speed, self.line_target_speed,
                self.line_torque_inc, self.line_brake_torque_inc,
                self.line_distance)
