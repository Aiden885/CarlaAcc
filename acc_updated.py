"""
ACC自适应巡航控制系统 - 重构版
采用高内聚低耦合设计，职责清晰分离
"""
import csv
import time
from collections import deque

# 新模块导入
from acc_config import ACCConfig
from acc_control_facade import ACCControlFacade
from carla_system_initializer import CarlaSystemInitializer
from control_loop_manager import ControlLoopManager
from display_manager import DisplayManager
from input_manager import InputManager
from output_formatter import OutputFormatter
from realtime_time_gap_plotter import RealtimeTimeGapPlotter
from resource_manager import ResourceManager
from result_plotter import RealtimeResultPlotter
from system_state import SystemState
from two_mode_controller import calculate_two_mode_desired_distance
from vehicle_utils import VehicleUtils


# ========== 常量定义 ==========
DISPLAY_TARGET_FPS = 60  # 显示器目标帧率
CSV_FLUSH_INTERVAL_S = 1.0  # CSV文件flush间隔（秒）
DEFAULT_COMMAND_TYPE = 0  # 默认指令类型

# CSV文件列名
CSV_HEADER = [
    "Time(s)", "Ego_Speed(km/h)", "Target_Speed(km/h)",
    "Actual_Distance(m)", "Desired_Distance(m)", "Control_Mode",
    "Lane_Offset", "ACC_State", "ACC_Active",
    "V_target_Setting", "V_min_Setting", "G2_Setting",
    "Manual_Throttle", "Manual_Brake", "Manual_Steer"
]


class acc:
    """
    - 类名不变
    - 对外接口方法不变
    - main()函数签名不变
    """

    def __init__(self, scenario_mode=None):
        """
        初始化ACC系统

        Args:
            scenario_mode: 场景模式 ("none", "cut-in", "cut-out")
        """
        print("\n" + "=" * 80)
        print("ACC自适应巡航控制系统（重构版 - 高内聚低耦合架构）")
        print("=" * 80)

        # ========== 1. 加载配置 ==========
        self.config = ACCConfig()
        self._apply_scenario_mode(scenario_mode)
        self.use_result_plotter = self.config.use_result_plotter

        # ========== 2. 初始化资源管理器 ==========
        self.resource_manager = ResourceManager()

        # ========== 3. 初始化显示管理器 ==========
        self.display_manager = DisplayManager(
            self.config.display_width,
            self.config.display_height
        )
        self.resource_manager.register_cleanable(self.display_manager)

        # ========== 4. 初始化CARLA系统 ==========
        try:
            carla_initializer = CarlaSystemInitializer(self.config)
            self.carla_resources = carla_initializer.initialize()

            # 注册资源到资源管理器
            self.resource_manager.set_carla_client(self.carla_resources.client)
            self.resource_manager.set_carla_world(self.carla_resources.world)
            self.resource_manager.set_synchronous_mode(self.config.synchronous_mode)
            self.resource_manager.register_carla_actors(self.carla_resources.get_all_actors())

            # 初始化显示管理器的相机
            self.display_manager.init_camera_manager(self.carla_resources.ego_vehicle)

        except Exception as e:
            print(f"[错误] CARLA系统初始化失败: {e}")
            self.resource_manager.cleanup_all()
            raise

        # ========== 5. 初始化ACC控制器 ==========
        self.acc_controller = ACCControlFacade(
            config=self.config,
            debug=self.config.acc_decision_debug
        )
        self.resource_manager.register_cleanable(self.acc_controller)

        # ========== 6. 初始化输入管理器 ==========
        self.input_manager = InputManager(self.config)

        # ========== 7. 初始化控制循环管理器 ==========
        self.control_loop_manager = ControlLoopManager(
            resources=self.carla_resources,
            acc_controller=self.acc_controller,
            config=self.config
        )

        # ========== 8. 初始化CSV记录器 ==========
        self.csv_file = None
        self.csv_writer = None
        self.last_csv_flush_time = 0.0  # 上次flush的时间
        self._init_csv()
        if self.csv_file:
            self.resource_manager.register_file(self.csv_file)

        # ========== 9. 初始化绘图器 ==========
        self._init_plotter()

        # ========== 10. 初始化实时模式（可选）==========
        self.enable_realtime = self.config.enable_realtime
        if self.enable_realtime:
            from enable_realtime_mode import RealtimeRateLimiter
            self.rate_limiter = RealtimeRateLimiter(target_fps=self.config.realtime_target_fps)
            print(f"[信息] 实时模式已启用（1:1速度，{self.config.realtime_target_fps} FPS）")
        else:
            print("[信息] 加速模式（全速运行，约60-80 FPS）")

        # ========== 11. 运行状态 ==========
        self.running = True
        self.start_time = None

        print("\n" + "=" * 80)
        print("[完成] ACC系统初始化完成")
        print("=" * 80)

    def _apply_scenario_mode(self, scenario_mode):
        """
        应用场景模式配置

        Args:
            scenario_mode: 场景模式选择
                - "none": 普通跟车（禁用所有场景）
                - "cut-in": 切入工况（侧向车辆插入）
                - "cut-out": 切出工况（前车离开）
                - None: 使用配置文件中的设定

        Raises:
            ValueError: 如果 scenario_mode 不是合法值
        """
        # 验证参数合法性
        valid_modes = {None, "none", "cut-in", "cut-out"}
        if scenario_mode not in valid_modes:
            raise ValueError(
                f"[错误] 无效的场景模式: {scenario_mode}\n"
                f"   有效值:\n"
                f"   - \"none\": 普通跟车（无特殊场景）\n"
                f"   - \"cut-in\": 切入工况（侧向车辆插入前方）\n"
                f"   - \"cut-out\": 切出工况（前车变道离开）\n"
                f"   - None: 使用配置文件设定（acc_config.py）"
            )

        if scenario_mode == "none":
            self.config.enable_cut_in_scenario = False
            self.config.enable_cut_out_scenario = False
        elif scenario_mode == "cut-in":
            self.config.enable_cut_in_scenario = True
            self.config.enable_cut_out_scenario = False
        elif scenario_mode == "cut-out":
            self.config.enable_cut_in_scenario = False
            self.config.enable_cut_out_scenario = True
        elif scenario_mode is None:
            # 使用配置文件设定，但两个场景不能同时启用
            if self.config.enable_cut_out_scenario and self.config.enable_cut_in_scenario:
                print("[警告] 配置文件中同时启用了切入和切出场景，已自动禁用切入场景")
                self.config.enable_cut_in_scenario = False

    def _init_csv(self):
        """初始化CSV文件"""
        try:
            self.csv_file = open(self.config.csv_output_file, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(CSV_HEADER)
            print("[完成] CSV记录器初始化完成")
        except Exception as e:
            print(f"[警告] CSV文件初始化失败: {e}")

    def _init_plotter(self):
        """初始化绘图器"""
        try:
            if self.use_result_plotter:
                self.realtime_plotter = RealtimeResultPlotter(
                    max_points=self.config.plotter_max_points,
                    update_interval=self.config.plotter_update_interval
                )
                self.realtime_plotter.start()
                print("[完成] 结果保存绘图器已启动 (Step模式，自动保存)")
            else:
                self.realtime_plotter = RealtimeTimeGapPlotter(
                    max_points=self.config.plotter_max_points,
                    update_interval=self.config.plotter_update_interval
                )
                self.realtime_plotter.start()
                print("[完成] 实时测试绘图器已启动 (Time模式)")

            self.resource_manager.register_cleanable(self.realtime_plotter)
        except Exception as e:
            print(f"[警告] 绘图器初始化失败: {e}")
            self.realtime_plotter = None

    def generate_target(self):
        """主循环 - 精简版（使用ControlLoopManager）"""
        try:
            self.start_time = time.time()
            self._print_usage_instructions()

            # 性能分析器（使用固定大小deque防止内存无限增长）
            max_samples = self.config.perf_max_samples
            perf_times = {
                "1_events": deque(maxlen=max_samples),
                "2_manual_input": deque(maxlen=max_samples),
                "3_world_tick": deque(maxlen=max_samples),
                "4_display_tick": deque(maxlen=max_samples),
                "5_control_loop": deque(maxlen=max_samples),
                "6_csv_record": deque(maxlen=max_samples),
                "7_plotter": deque(maxlen=max_samples),
                "8_render": deque(maxlen=max_samples),
                "9_total_cycle": deque(maxlen=max_samples),
            }
            perf_start_time = time.time()
            last_perf_report_time = time.time()

            while self.running:
                cycle_start = time.time()

                # 1. 处理显示事件
                t0 = time.time()
                self._handle_display_events()
                perf_times["1_events"].append(time.time() - t0)

                # 2. 更新手动输入
                t0 = time.time()
                manual_input_state = self.input_manager.update_manual_inputs()
                perf_times["2_manual_input"].append(time.time() - t0)

                # 3. 世界更新
                t0 = time.time()
                if self.carla_resources.world:
                    self.carla_resources.world.tick()
                perf_times["3_world_tick"].append(time.time() - t0)

                # 4. 实时速率限制
                if self.enable_realtime and hasattr(self, "rate_limiter"):
                    self.rate_limiter.wait()

                # 5. 显示更新
                t0 = time.time()
                self.display_manager.tick(DISPLAY_TARGET_FPS)
                perf_times["4_display_tick"].append(time.time() - t0)

                # 6. 执行控制循环
                t0 = time.time()
                step_result = self.control_loop_manager.run_single_step(manual_input_state)
                perf_times["5_control_loop"].append(time.time() - t0)

                # 7. 数据记录
                t0 = time.time()
                self._record_data(step_result)
                perf_times["6_csv_record"].append(time.time() - t0)

                # 8. 绘图更新
                t0 = time.time()
                self._update_plotter(step_result, manual_input_state)
                perf_times["7_plotter"].append(time.time() - t0)

                # 9. 显示渲染
                t0 = time.time()
                system_info = self._get_system_info(step_result, manual_input_state)
                self.display_manager.render_display(system_info)
                perf_times["8_render"].append(time.time() - t0)

                # 10. 输出Simulink I/O信息
                self._print_simulink_io(step_result, manual_input_state)

                # 11. 定期性能报告
                current_time = time.time()
                if current_time - last_perf_report_time >= self.config.performance_report_interval:
                    self._print_performance_report(perf_times, current_time - perf_start_time)
                    last_perf_report_time = current_time

                # 记录完整周期时间
                perf_times["9_total_cycle"].append(time.time() - cycle_start)

        except KeyboardInterrupt:
            print("\n[警告] 用户中断程序")
        except Exception as e:
            print(f"\n[错误] 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 最终性能报告
            if "perf_times" in locals() and "perf_start_time" in locals():
                print("\n" + "=" * 80)
                print("[报告] 最终性能分析报告")
                print("=" * 80)
                self._print_performance_report(perf_times, time.time() - perf_start_time)

            print("\n[信息] 正在清理资源...")
            self.destroy()

    def _handle_display_events(self):
        """处理显示事件"""
        display_events = self.display_manager.handle_display_events()

        for event_type, event_data in display_events:
            if event_type == "quit":
                self.running = False
                return

            # 获取ACC系统状态
            acc_enabled = self.control_loop_manager.system_state.acc.system_enabled

            if event_type == "keydown":
                result = self.input_manager.process_keydown_event(event_data, acc_enabled)
                if result:
                    self._process_input_event(result)

            elif event_type == "keyup":
                result = self.input_manager.process_keyup_event(event_data)
                if result:
                    self._process_input_event(result)

    def _process_input_event(self, event: dict):
        """处理输入事件"""
        event_type = event["type"]

        if event_type == "quit":
            self.running = False

        elif event_type == "acc_toggle":
            current_state = self.control_loop_manager.system_state.acc.system_enabled
            new_state = not current_state
            self.control_loop_manager.set_acc_system_enabled(new_state)
            print(f"[操作] ACC主开关: {'ON' if new_state else 'OFF'}")

        elif event_type == "debounced_space":
            # 去抖动忽略
            pass

        elif event_type == "debug_toggle":
            self.acc_controller.debug = not self.acc_controller.debug
            print(f"[操作] ACC调试模式: {'ON' if self.acc_controller.debug else 'OFF'}")

        elif event_type == "keyboard_command":
            # 设置待处理的指令
            desc = event["desc"]
            self.control_loop_manager.system_state.set_pending_command(
                event["code"],
                desc
            )
            print(f"[操作] 已记录键盘指令: {desc} (将在下一帧处理)")

        elif event_type == "acc_disabled_warning":
            print("[提示] ACC已禁用，按空格键启用")

        elif event_type == "trigger_ramp":
            self.control_loop_manager.trigger_ramp_speed()
            print("[操作] 触发前车斜坡速度")

        elif event_type == "lane_change_left":
            self.control_loop_manager.trigger_lane_change_left()
            print("[操作] 触发前车向左换道")

        elif event_type == "lane_change_right":
            self.control_loop_manager.trigger_lane_change_right()
            print("[操作] 触发前车向右换道")

        elif event_type == "throttle_released":
            print("[操作] 油门松开")

        elif event_type == "brake_released":
            print("[操作] 刹车松开")

    def _record_data(self, step_result):
        """记录数据到CSV"""
        if not self.csv_writer:
            return

        current_time = time.time() - self.start_time
        system_state = step_result.system_state
        env_data = step_result.env_data

        # 计算期望距离
        desired_distance, control_mode = calculate_two_mode_desired_distance(
            system_state.ego.speed_ms
        )

        self.csv_writer.writerow([
            current_time,
            system_state.ego.speed_kmh,
            system_state.target.speed_kmh,
            env_data.get("vehicle_distance", 0.0),
            desired_distance,
            control_mode,
            env_data.get("lane_offset", 0.0),
            "Hybrid Python+Simulink Mode",
            system_state.acc.system_enabled,
            system_state.acc.target_speed_kmh,
            system_state.acc.min_speed_kmh,
            system_state.acc.time_gap_s,
            system_state.ego.throttle,
            system_state.ego.brake,
            system_state.ego.steer
        ])

        # 优化：每秒flush一次，而不是每帧，避免频繁磁盘写入导致性能下降
        now = time.time()
        if now - self.last_csv_flush_time >= CSV_FLUSH_INTERVAL_S:
            self.csv_file.flush()
            self.last_csv_flush_time = now

    def _update_plotter(self, step_result, manual_input_state):
        """更新绘图器"""
        if not self.realtime_plotter:
            return

        env_data = step_result.env_data
        system_state = step_result.system_state
        unified_output = step_result.unified_output

        # 只在TIME模式且有前车时更新
        if env_data.get("control_mode_flag") == 1 and env_data.get("has_target"):
            # 计算req_torque和req_brake_torque（用于绘图）
            control_output_val = unified_output.get("sppvt_control_output", 0.0)
            # TIME模式下SPPVT输出需要取负（符号转换）
            sppvt_torque_demand = -control_output_val

            if sppvt_torque_demand >= 0:
                # 加速：显示加速扭矩
                req_torque = sppvt_torque_demand * self.config.sppvt_accel_scale
                req_brake_torque = 0.0
            else:
                # 制动：显示制动扭矩（绝对值）
                req_torque = 0.0
                req_brake_torque = abs(sppvt_torque_demand * self.config.sppvt_decel_scale)

            # 从enhanced_output获取时距数据
            try:
                from two_mode_controller import enhanced_two_mode_control
                enhanced_output = enhanced_two_mode_control(
                    system_state.ego.speed_ms,
                    env_data.get("vehicle_distance"),
                    system_state.acc.target_speed_kmh / 3.6
                )
                desired_time_gap = enhanced_output.get("reference_value", 0.0)
                actual_time_gap = enhanced_output.get("current_value", 0.0)
            except:
                desired_time_gap = 0.0
                actual_time_gap = 0.0

            self.realtime_plotter.add_data(
                desired_time_gap,
                actual_time_gap,
                system_state.frame_count,
                system_state.ego.speed_kmh,
                system_state.target.speed_kmh,
                unified_output.get("control_enabled", False),
                req_torque,
                req_brake_torque
            )

    def _get_system_info(self, step_result, manual_input_state):
        """获取系统状态信息（用于显示）"""
        system_state = step_result.system_state
        acc_params = self.control_loop_manager.get_acc_params()

        return OutputFormatter.format_system_info(
            ego_vehicle=self.carla_resources.ego_vehicle,
            target_vehicle=self.carla_resources.target_vehicle,
            acc_system_enabled=system_state.acc.system_enabled,
            acc_decision=self.acc_controller,
            acc_params=acc_params,
            throttle=system_state.ego.throttle,
            brake=system_state.ego.brake,
            steer=system_state.ego.steer,
            get_vehicle_speed_func=VehicleUtils.get_vehicle_speed,
            get_vehicle_distance_func=VehicleUtils.get_vehicle_distance
        )

    def _print_simulink_io(self, step_result, manual_input_state):
        """打印Simulink I/O信息"""
        unified_input = {
            "ego_speed_kmh": step_result.system_state.ego.speed_kmh,
            "ego_speed_ms": step_result.system_state.ego.speed_ms,
            "command_type": DEFAULT_COMMAND_TYPE,
            "control_error": step_result.env_data.get("control_error", 0.0),
            "control_mode_flag": step_result.env_data.get("control_mode_flag", 0),
            "V_target_kmh": step_result.system_state.acc.target_speed_kmh,
            "V_min_kmh": step_result.system_state.acc.min_speed_kmh,
            "G2_s": step_result.system_state.acc.time_gap_s,
        }

        OutputFormatter.print_simulink_io(
            frame_num=step_result.system_state.frame_count,
            current_time=time.time() - self.start_time,
            unified_input=unified_input,
            unified_output=step_result.unified_output,
            duration_ms=step_result.simulink_duration_ms,
            manual_throttle_input=manual_input_state.throttle,
            manual_brake_input=manual_input_state.brake,
            w_key_pressed=manual_input_state.w_pressed,
            s_key_pressed=manual_input_state.s_pressed,
            acc_system_enabled=step_result.system_state.acc.system_enabled,
            command_description=step_result.unified_output.get('command_description'),
            final_control=step_result.control_output.to_dict(),
            env_data=step_result.env_data
        )

    def _print_usage_instructions(self):
        """打印使用说明"""
        print("\n=== ACC Integrated Control System ===")

    def _print_performance_report(self, perf_times, elapsed_time):
        """打印性能分析报告"""
        print("\n" + "=" * 60)
        print(f"[报告] 性能分析报告 (运行时间: {elapsed_time:.1f}秒)")
        print("=" * 60)

        if not perf_times:
            print("[警告] 暂无性能数据")
            return

        total_avg = 0
        for key in sorted(perf_times.keys()):
            times = perf_times[key]
            if times:
                avg_ms = (sum(times) / len(times)) * 1000
                max_ms = max(times) * 1000
                min_ms = min(times) * 1000
                print(f"{key:20s}: 平均 {avg_ms:6.2f}ms | 最大 {max_ms:6.2f}ms | "
                      f"最小 {min_ms:6.2f}ms | 次数 {len(times)}")
                total_avg += avg_ms

        if total_avg > 0:
            print("-" * 60)
            print(f"{'总计':20s}: 平均 {total_avg:6.2f}ms/周期")
            fps = 1000.0 / total_avg
            print(f"{'理论帧率':20s}: {fps:6.2f} FPS")
        print("=" * 60)

    def destroy(self):
        """清理资源（使用ResourceManager）"""
        self.resource_manager.cleanup_all()

    # === 向后兼容的接口方法 ===

    def get_current_parameters(self):
        """获取当前ACC参数"""
        return self.control_loop_manager.get_acc_params()

    def get_status_info(self):
        """获取ACC状态信息"""
        return {
            "state_description": "Hybrid Python+Simulink Mode",
            "system_enabled": self.control_loop_manager.system_state.acc.system_enabled
        }

    @property
    def ego_vehicle(self):
        """向后兼容：访问自车"""
        return self.carla_resources.ego_vehicle

    @property
    def target_vehicle(self):
        """向后兼容：访问前车"""
        return self.carla_resources.target_vehicle

    @property
    def world(self):
        """向后兼容：访问世界"""
        return self.carla_resources.world

    @property
    def client(self):
        """向后兼容：访问客户端"""
        return self.carla_resources.client

    @property
    def vehicles(self):
        """向后兼容：访问所有车辆"""
        return self.carla_resources.other_vehicles

    @property
    def acc_decision(self):
        """向后兼容：访问ACC决策器"""
        return self.acc_controller

    @property
    def acc_system_enabled(self):
        """向后兼容：访问ACC开关状态"""
        return self.control_loop_manager.system_state.acc.system_enabled


def main():
    """主函数"""
    # 手动切换工况 "none" / "cut-in" / "cut-out"
    SCENARIO_MODE = "none"  # None=按配置文件，"none"=普通，"cut-in"=切入，"cut-out"=切出

    # 创建ACC实例（画图模式在 acc_config.py 中配置）
    acc_actor = acc(scenario_mode=SCENARIO_MODE)

    try:
        acc_actor.generate_target()
    except KeyboardInterrupt:
        print("[信息] 程序被中断")


if __name__ == "__main__":
    main()
