"""Real-time time gap plotter with a MATLAB-like appearance."""

import threading
import queue
import time
from collections import deque

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider, CheckButtons

# Use the Tk backend because it plays nicely with separate threads
matplotlib.use("TkAgg")


class RealtimeTimeGapPlotter:
    """Thread-safe real-time plotter for desired vs. actual time gaps."""

    def __init__(self, max_points: int = 500, update_interval: int = 100) -> None:
        """Initialise the plotter.

        Args:
            max_points: Reserved parameter (not currently limiting data storage).
            update_interval: Animation refresh interval in milliseconds.
        """
        self.data_queue = queue.Queue(maxsize=1000)

        self.max_points = max_points  # 保留此参数用于其他逻辑，但不限制数据存储
        # 移除maxlen限制，保留所有历史数据以支持滑动条回看
        self.timestamps = deque()
        self.desired_gaps = deque()
        self.actual_gaps = deque()
        self.errors = deque()

        self.fig = None
        self.axes = None
        self.animation_obj = None
        self.update_interval = update_interval

        self.plot_thread = None
        self.running = False

        self.total_points = 0
        self.start_time = None

        # 滑动条控制
        self.slider_start = None
        self.checkbox_auto = None
        self.auto_follow = True  # 默认自动跟随最新数据
        self.manual_start_time = 0.0  # 手动设置的起始时间
        self._updating_slider = False  # 防止递归回调的标志

    def add_data(self, desired_gap: float, actual_gap: float, timestamp: float) -> None:
        """Push a new sample into the queue from the producer thread."""
        try:
            self.data_queue.put_nowait((desired_gap, actual_gap, timestamp))
        except queue.Full:
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait((desired_gap, actual_gap, timestamp))
            except queue.Empty:
                pass

    def start(self) -> None:
        """Start the background plotting thread."""
        if self.running:
            print("[RealtimeTimeGapPlotter] Already running.")
            return

        self.running = True
        self.start_time = time.time()
        self.plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
        self.plot_thread.start()
        print("[RealtimeTimeGapPlotter] Started (MATLAB style).")

    def stop(self) -> None:
        """Stop the plotting thread and close the figure window."""
        self.running = False
        if self.plot_thread:
            self.plot_thread.join(timeout=2.0)
        if self.fig:
            plt.close(self.fig)
        print("[RealtimeTimeGapPlotter] Stopped.")

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

        self.fig = plt.figure(figsize=(14, 8.5), facecolor=bg_color)
        self.fig.canvas.manager.set_window_title(
            "Real-Time Time Gap Tracking - TIME Mode"
        )

        # 调整布局以腾出底部空间放置滑动条
        # 子图占用上方，底部留给控制器（只有一个滑动条，图表可以更大）
        self.ax1 = plt.axes([0.1, 0.56, 0.85, 0.36], facecolor=bg_color)
        self.ax2 = plt.axes([0.1, 0.18, 0.85, 0.28], facecolor=bg_color)

        self.axes = [self.ax1, self.ax2]

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
        self.ax1.set_xlabel("Time (s)", color=text_color, fontsize=11, fontweight='bold')
        self.ax1.set_ylabel("Time Gap (s)", color=text_color, fontsize=11, fontweight='bold')
        self.ax1.set_title(
            "Time Gap Tracking", color=text_color, fontsize=13, fontweight="bold"
        )
        self.ax1.legend(
            loc="upper right", facecolor=bg_color, edgecolor="#000000", fontsize=10, framealpha=1.0
        )

        # 子图2：误差
        self.line_error, = self.ax2.plot(
            [], [], color=self.color_error, linewidth=2.0, label="Time Gap Error", marker='d', markersize=2, markevery=10
        )
        self.ax2.axhline(y=0, color="#000000", linestyle="--", linewidth=1.5, alpha=0.7)
        self.ax2.set_xlabel("Time (s)", color=text_color, fontsize=11, fontweight='bold')
        self.ax2.set_ylabel("Error (s)", color=text_color, fontsize=11, fontweight='bold')
        self.ax2.set_title(
            "Time Gap Error (Actual - Desired)", color=text_color, fontsize=13, fontweight="bold"
        )
        self.ax2.legend(
            loc="upper right", facecolor=bg_color, edgecolor="#000000", fontsize=10, framealpha=1.0
        )

        # 创建滑动条控件
        # 起始时间滑动条（从此时间开始显示到最新数据）
        ax_slider_start = plt.axes([0.1, 0.07, 0.65, 0.03], facecolor='#e0e0e0')
        self.slider_start = Slider(
            ax=ax_slider_start,
            label='Start Time (s)',
            valmin=0.0,
            valmax=100.0,
            valinit=0.0,
            valstep=0.1,
            color=self.color_desired
        )
        self.slider_start.on_changed(self._on_slider_start_change)

        # 自动跟随checkbox
        ax_checkbox = plt.axes([0.80, 0.05, 0.15, 0.08], facecolor=bg_color)
        self.checkbox_auto = CheckButtons(
            ax_checkbox,
            ['Auto Follow'],
            [True]
        )
        self.checkbox_auto.on_clicked(self._on_checkbox_toggle)

    def _on_slider_start_change(self, val):
        """起始时间滑动条变化回调"""
        # 防止递归调用
        if self._updating_slider:
            return

        self.manual_start_time = val
        # 用户手动拖动滑块，切换到手动模式
        if self.auto_follow:
            self.auto_follow = False
            # 更新checkbox状态
            if self.checkbox_auto:
                self._updating_slider = True
                self.checkbox_auto.set_active(0)
                self._updating_slider = False
            print("📊 手动拖动滑块，切换到手动控制模式")

    def _on_checkbox_toggle(self, label):
        """自动跟随checkbox切换回调"""
        self.auto_follow = not self.auto_follow
        if self.auto_follow:
            print("📊 切换到自动跟随模式")
        else:
            print("📊 切换到手动控制模式")

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
            print(f"[RealtimeTimeGapPlotter] Plotting thread error: {exc}")
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
                desired, actual, timestamp = self.data_queue.get_nowait()
            except queue.Empty:
                break

            self.timestamps.append(timestamp)
            self.desired_gaps.append(desired)
            self.actual_gaps.append(actual)
            self.errors.append(actual - desired)

            self.total_points += 1
            data_count += 1

        if not self.timestamps:
            return ()

        ts = np.asarray(self.timestamps, dtype=float)
        desired = np.asarray(self.desired_gaps, dtype=float)
        actual = np.asarray(self.actual_gaps, dtype=float)
        errors = np.asarray(self.errors, dtype=float)

        self.line_desired.set_data(ts, desired)
        self.line_actual.set_data(ts, actual)
        self.line_error.set_data(ts, errors)

        # 动态更新滑动条范围（但不频繁触发重绘）
        if ts.size and self.slider_start and not self._updating_slider:
            data_min = float(ts[0])
            data_max = float(ts[-1])

            # 只在数据范围显著增长时更新滑动条的最大值（减少重绘）
            if data_max > self.slider_start.valmax + 5.0:  # 至少增长5秒才更新
                self._updating_slider = True
                self.slider_start.valmax = data_max
                self.slider_start.ax.set_xlim(0, data_max)
                self._updating_slider = False

            # 在自动跟随模式下，保持起始时间为0（显示所有数据）
            if self.auto_follow:
                self.manual_start_time = 0.0

        # Set X-axis limits and get visible range
        x_left, x_right = 0, 1
        if ts.size:
            data_min = float(ts[0])
            data_max = float(ts[-1])

            if self.auto_follow:
                # 自动跟随模式：从起始显示到最新数据
                x_left = data_min
                x_right = data_max
            else:
                # 手动控制模式：从Start Time显示到最新数据
                x_left = max(data_min, self.manual_start_time)
                x_right = data_max

            # 添加一点padding使图表更美观
            if x_right > x_left:
                padding = (x_right - x_left) * 0.02
                x_right += padding

            for axis in (self.ax1, self.ax2):
                axis.set_xlim(x_left, x_right)

        # Set Y-axis limits using ONLY visible data in current X-axis window
        if desired.size and ts.size:
            # Find indices of data points within visible X-axis range
            visible_mask = (ts >= x_left) & (ts <= x_right)

            if np.any(visible_mask):
                visible_desired = desired[visible_mask]
                visible_actual = actual[visible_mask]

                y_min = float(np.nanmin([visible_desired.min(), visible_actual.min()]))
                y_max = float(np.nanmax([visible_desired.max(), visible_actual.max()]))

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

        # Set error Y-axis limits using ONLY visible data
        if errors.size and ts.size:
            visible_mask = (ts >= x_left) & (ts <= x_right)

            if np.any(visible_mask):
                visible_errors = errors[visible_mask]
                err_abs_max = float(np.nanmax(np.abs(visible_errors)))

                if not np.isfinite(err_abs_max):
                    err_abs_max = 0.1
                err_abs_max = max(err_abs_max, 0.05) * 1.1
                self.ax2.set_ylim(-err_abs_max, err_abs_max)
                self.ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

        return self.line_desired, self.line_actual, self.line_error


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    plotter = RealtimeTimeGapPlotter(max_points=300, update_interval=50)
    plotter.start()

    print("Generating simulated data...")
    import math

    try:
        for i in range(1000):
            t = i * 0.1
            desired_gap = 2.0
            actual_gap = 2.0 + 0.5 * math.sin(t * 0.5) + np.random.normal(0, 0.1)
            plotter.add_data(desired_gap, actual_gap, t)
            time.sleep(0.1)

        print("Data generation complete. Press Ctrl+C to exit.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        plotter.stop()
