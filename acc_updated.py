import csv
import time

import carla
import numpy as np
# Pygame相关
import pygame
from pygame.locals import *

from acc_decision_sppvt_interface import ACCDecisionSPPVTInterface
# 导入显示管理器
from display_manager import DisplayManager
# 导入换道控制器
from lane_change_controller import LaneChangeController
# 导入横向PID控制器
from lateral_pid_controller import LateralPIDController
# 导入增强横向控制器（PID + 预瞄）
from enhanced_lateral_controller import EnhancedLateralController
from manual_input_controller import ManualSteeringController
from output_formatter import OutputFormatter
# 导入斜坡速度控制器
from ramp_speed_controller import RampSpeedController
# 导入实时绘图器
from realtime_time_gap_plotter import RealtimeTimeGapPlotter
# 导入扭矩到油门转换器
from torque_to_throttle_converter import TorqueToThrottleConverter
# ACC相关模块
from two_mode_controller import calculate_two_mode_desired_distance, set_two_mode_parameters, enhanced_two_mode_control
# 导入拆分后的工具模块
from vehicle_utils import VehicleUtils


# $env:HTTP_PROXY = "http://127.0.0.1:7890"
# $env:HTTP_PROXY = "http://127.0.0.1:7890"
# $env:ALL_PROXY = "socks5://127.0.0.1:7891"
# 升级条件: (acceleration < 0) && (|velocity| <= delta) && (|error| > eta)


class acc:
    def __init__(self):
        """
        初始化ACC系统
        使用CARLA API直接获取前车距离和车道信息
        """
        # === 显示管理器初始化 ===
        self.display_manager = DisplayManager(1280, 720)

        # === CARLA API感知模块 ===
        self.carla_perception = None  # 将在init_carla()后初始化

        # === 通用变量 ===
        self.max_follow_distance = 50
        self.image_width = 1280
        self.image_height = 720
        self.target_vehicle = None
        self.start_time = None
        self.csv_file = None
        self.csv_writer = None

        # === ACC决策+SPPVT一体化模块 ===
        # 使用完整Simulink模型（包含决策+SPPVT控制），不使用简化的realtime_sppvt_manager
        self.acc_decision_sppvt = ACCDecisionSPPVTInterface(debug=True, use_realtime_sppvt=False)

        # === ACC系统可配置参数 (环境相关，需要传递给Simulink) ===
        self.acc_params = {
            'V_target_kmh': 50.0,  # 默认巡航速度 - 传递给Simulink
            'V_min_kmh': 20.0,  # 最小速度阈值 - 传递给Simulink
            'G2_s': 2.0,  # 时距参数 - 传递给Simulink
            'V_threshold_kmh': 50.0,  # 模式切换阈值 - 用于Two Mode控制器
            'speed_step': 5.0  # 速度调整步长
        }

        # === 保持原有决策接口兼容性 ===
        self.acc_decision = self.acc_decision_sppvt  # 兼容性别名

        # === 控制状态 ===
        self.acc_system_enabled = False  # ACC系统开关（空格键）
        self.manual_control_active = True
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0

        # === 手动输入状态（用于扭矩仲裁）===
        self.manual_steering_controller = ManualSteeringController()
        self.manual_throttle_input = 0.0
        self.manual_brake_input = 0.0
        self.manual_steer_input = 0.0
        self._last_manual_steer_update = time.time()

        # 按键按下状态跟踪
        self.w_key_pressed = False  # W键（油门）是否按下
        self.s_key_pressed = False  # S键（刹车）是否按下
        self.a_key_pressed = False  # A键（左转）是否按下
        self.d_key_pressed = False  # D键（右转）是否按下

        # 键盘指令队列（单帧有效，只记录指令不立即调用Simulink）
        self.pending_keyboard_command = None  # 格式: {'code': int, 'description': str}

        # === 横向控制器 ===
        # 使用增强横向控制器（PID + 预瞄）
        self.lateral_controller = EnhancedLateralController(kp=0.1, ki=0.01, kd=0.02)

        # 🔧 可调整预瞄参数
        # self.lateral_controller.set_weights(weight_current=0.7, weight_lookahead=0.3)
        # self.lateral_controller.set_lookahead_params(base=8.0, gain=0.3)

        # === 运行控制 ===
        self.running = True

        # 初始化CARLA
        self.init_carla()
        self.init_csv()

        # 初始化两模式参数 - 已集成到Simulink，无需外部同步

        # 初始化一体化接口的两模式控制器
        if hasattr(self.acc_decision_sppvt, 'init_two_mode_controller'):
            self.acc_decision_sppvt.init_two_mode_controller()

        # === 初始化实时时距绘图器 ===
        self.realtime_plotter = RealtimeTimeGapPlotter(max_points=500, update_interval=100)
        self.realtime_plotter.start()
        print(" 实时时距绘图器已启动")

    def init_carla(self):
        # 初始化 Carla 客户端
        # self.client = carla.Client('192.168.0.146', 2000)
        # map_name = 'acc_30km'

        self.client = carla.Client('localhost', 2000)
        map_name = 'Town04'
        self.client.set_timeout(60.0)
        try:
            self.world = self.client.get_world()
            self.world = self.client.load_world(map_name, carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load map {map_name}: {e}")

        # 设置同步模式（放宽时间步长以匹配Simulink处理能力）
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS (50ms per frame)
        self.world.apply_settings(settings)

        # 获取蓝图库和地图
        self.blueprint_library = self.world.get_blueprint_library()
        map = self.world.get_map()

        # 获取车辆蓝图
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_vehicle_bp = self.blueprint_library.filter('vehicle.audi.etron')[0]

        # 定义固定生成点x=0.663731, y=-203.651886, z=0.5
        # right x = -352.701508, y = 4627.016113, z=0.5
        # left x=-951.054749, y=4027.188232, z=-0.009344
        # up  x=1951.489014, y=-4947.605469, z=-0.009341
        # down x=2121.978760, y=-3415.833252, z=54.469646
        fixed_point = carla.Location(x=0.663731, y=-203.651886, z=0.5)
        waypoint = map.get_waypoint(fixed_point, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            raise RuntimeError("Failed to find a valid waypoint near the specified location")

        # 生成目标车辆
        spawn_point = waypoint.transform
        spawn_point.location.z += 0.1

        vehicles = []
        target_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if target_vehicle is None:
            raise RuntimeError("Failed to spawn target vehicle at waypoint location")
        vehicles.append(target_vehicle)
        self.target_vehicle = target_vehicle
        self.target_vehicle.set_autopilot(True)

        # === 斜坡速度控制器配置 ===
        # 🔧 可调整参数：修改这些参数来改变斜坡响应特性
        self.ramp_start_speed_kmh = 50.0   # 斜坡起始速度 (km/h)
        self.ramp_target_speed_kmh = 120.0  # 斜坡目标速度 (km/h)
        self.ramp_duration_s = 10.0        # 斜坡持续时间 (秒)

        # 初始化斜坡速度控制器
        self.ramp_controller = RampSpeedController(
            start_speed_kmh=self.ramp_start_speed_kmh,
            target_speed_kmh=self.ramp_target_speed_kmh,
            duration_s=self.ramp_duration_s
        )

        # 前车速度配置（使用constant velocity定速巡航）
        # 初始速度设为斜坡起始速度，按F键后触发斜坡变化
        self.target_speed_kmh = self.ramp_start_speed_kmh  # 初始使用斜坡起始速度
        self.use_constant_velocity = True  # 使用constant velocity模式（不受路口影响）

        # 生成自车：沿车道前进方向偏移一定距离以避免碰撞
        ego_waypoints = waypoint.previous(10.0)
        if not ego_waypoints:
            raise RuntimeError("Failed to find a waypoint 20 meters ahead for ego vehicle spawn")
        ego_spawn_point = ego_waypoints[0].transform
        ego_spawn_point.location.z += 0.1
        self.ego_vehicle = self.world.try_spawn_actor(ego_vehicle_bp, ego_spawn_point)
        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        self.vehicles = vehicles
        self.ego_vehicle.set_autopilot(False)

        # 初始化扭矩到油门转换器
        print("初始化扭矩到油门转换器...")
        self.torque_converter = TorqueToThrottleConverter(self.ego_vehicle)
        self.use_torque_converter = True  # 是否使用物理模型转换器（True）或简单映射（False）

        # 初始化显示管理器的相机
        self.display_manager.init_camera_manager(self.ego_vehicle)

        # 设置交通管理器（必须与CARLA世界同步模式一致）
        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_synchronous_mode(True)
        self.tm_port = tm.get_port()
        tm.auto_lane_change(self.ego_vehicle, False)

        # 保存Traffic Manager引用供后续使用
        self.tm = tm

        # === 初始化换道控制器 ===
        self.lane_change_controller = LaneChangeController(tm, self.target_vehicle)
        print("✅ 前车换道控制器已初始化 (Z键向左, X键向右)")

        if self.use_constant_velocity:
            # === 使用constant velocity模式（定速巡航，不受路口影响）===
            # 注意：需要保持autopilot开启以获得转向控制
            for vehicle in vehicles:
                vehicle.set_autopilot(True, self.tm_port)
                tm.auto_lane_change(vehicle, True)  # 启用换道能力（由换道控制器管理）
                tm.ignore_lights_percentage(vehicle, 100.0)  # 忽略红绿灯

                # 启用恒定速度模式（m/s）
                target_speed_ms = self.target_speed_kmh / 3.6
                vehicle.enable_constant_velocity(carla.Vector3D(target_speed_ms, 0, 0))

            print(f"✅ 前车速度配置完成 (Constant Velocity模式):")
            print(f"   初始速度: {self.target_speed_kmh:.1f} km/h ({target_speed_ms:.2f} m/s)")
            print(f"   模式: 恒定速度（不受路口/限速影响）")
            print(f"   转向控制: Autopilot")
            print(f"\n📊 斜坡速度配置 (按F键触发):")
            print(f"   起始速度: {self.ramp_start_speed_kmh:.1f} km/h")
            print(f"   目标速度: {self.ramp_target_speed_kmh:.1f} km/h")
            print(f"   斜坡时间: {self.ramp_duration_s:.1f} 秒")

        else:
            # === 使用Traffic Manager限速百分比模式 ===
            for vehicle in vehicles:
                vehicle.set_autopilot(True, self.tm_port)
                tm.auto_lane_change(vehicle, False)
                tm.ignore_lights_percentage(vehicle, 100.0)

            # 初始化限速记录（用于主循环中检测限速变化）
            self.last_speed_limit = None

            print(f"✅ 前车速度配置完成 (Traffic Manager模式):")
            print(f"   目标速度: {self.target_speed_kmh:.1f} km/h")
            print(f"   速度控制: 将在主循环中根据实时路段限速动态调整")

        # 设置交通灯
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)

        # === 初始化CARLA API感知模块 ===
        from carla_perception import CarlaPerception
        self.carla_perception = CarlaPerception(
            self.world,
            self.ego_vehicle,
            self.target_vehicle
        )
        print("✅ CARLA API感知模块初始化完成")

    def init_csv(self):
        """初始化CSV文件"""
        self.csv_file = open('speed_data_integrated.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Time(s)',
            'Ego_Speed(km/h)',
            'Target_Speed(km/h)',
            'Actual_Distance(m)',
            'Desired_Distance(m)',
            'Control_Mode',
            'Lane_Offset',
            'ACC_State',
            'ACC_Active',
            'V_target_Setting',
            'V_min_Setting',
            'G2_Setting',
            'Manual_Throttle',
            'Manual_Brake',
            'Manual_Steer'
        ])

    def _sync_two_mode_parameters(self):
        """同步ACC决策参数到两模式控制器"""
        acc_params = self.acc_decision.get_current_parameters()
        set_two_mode_parameters(
            V_threshold_kmh=acc_params['V_target_kmh'],
            G2_s=acc_params['G2_s'],
            target_speed_kmh=acc_params['V_target_kmh']
        )

        # 同时同步到一体化接口
        if hasattr(self.acc_decision_sppvt, 'update_two_mode_parameters'):
            self.acc_decision_sppvt.update_two_mode_parameters(
                V_threshold_kmh=acc_params['V_target_kmh'],
                G2_s=acc_params['G2_s'],
                target_speed_kmh=acc_params['V_target_kmh']
            )

    def get_current_parameters(self):
        """获取当前ACC参数，直接使用Simulink管理的参数"""
        return self.acc_params.copy()

    def get_status_info(self):
        """获取ACC状态信息，基于Simulink状态"""
        return {
            'state_description': 'Pure Simulink Mode',
            'system_enabled': self.acc_system_enabled
        }

    def handle_keyboard_input(self):
        """处理持续按键输入（基础手动驾驶控制）"""
        keys = pygame.key.get_pressed()

        # 注意：W/S键的油门/刹车已经在handle_events()和主循环中处理
        # 这里只处理基础的手动驾驶控制（用于完全手动模式）

        # 基础车辆控制（只在手动模式下有效）
        if self.manual_control_active:
            # 油门（使用UP键）
            if keys[K_UP]:
                self.throttle = min(1.0, self.throttle + 0.02)
            else:
                self.throttle = max(0.0, self.throttle - 0.05)

            # 刹车（使用DOWN键）
            if keys[K_DOWN]:
                self.brake = min(1.0, self.brake + 0.05)
            else:
                self.brake = max(0.0, self.brake - 0.1)

            # 转向
            if keys[K_a] or keys[K_LEFT]:
                self.steer = max(-1.0, self.steer - 0.05)
            elif keys[K_d] or keys[K_RIGHT]:
                self.steer = min(1.0, self.steer + 0.05)
            else:
                self.steer = self.steer * 0.9

            # 应用手动控制
            if self.ego_vehicle:
                control = carla.VehicleControl()
                control.throttle = self.throttle
                control.brake = self.brake
                control.steer = self.steer
                control.hand_brake = False
                self.ego_vehicle.apply_control(control)

    def handle_events(self):
        """Process display events and translate them into ACC actions."""
        display_events = self.display_manager.handle_display_events()

        for event_type, event_data in display_events:
            if event_type == 'quit':
                self.running = False
                return

            if event_type == 'keydown':
                if event_data == K_ESCAPE:
                    self.running = False
                elif event_data == K_p:
                    self.acc_decision.debug = not self.acc_decision.debug
                    print(f"ACC debug: {'ON' if self.acc_decision.debug else 'OFF'}")
                elif event_data == K_SPACE:
                    self.acc_system_enabled = not self.acc_system_enabled
                    status = 'ON' if self.acc_system_enabled else 'OFF'
                    print(f"主开关 ACC master switch: {status}")
                    if not self.acc_system_enabled:
                        self.acc_decision.reset()
                elif event_data in (K_q, K_e, K_r, K_t, K_c):
                    if not self.acc_system_enabled:
                        print('提示 ACC disabled, press SPACE to enable')
                    else:
                        mapping = {
                            K_q: (2, 'Q增速(I1)'),
                            K_e: (1, 'E降速(I0)'),
                            K_r: (4, 'R增距(I3)'),
                            K_t: (3, 'T降距(I2)'),
                            K_c: (7, 'C取消(I6)')
                        }
                        self._send_keyboard_command_to_simulink(*mapping[event_data])
                elif event_data == K_w:
                    self.w_key_pressed = True
                    if self.acc_system_enabled:
                        self._send_keyboard_command_to_simulink(5, 'W油门(I4)')
                elif event_data == K_s:
                    self.s_key_pressed = True
                    if self.acc_system_enabled:
                        self._send_keyboard_command_to_simulink(6, 'S刹车(I5)')
                elif event_data == K_a:
                    self.a_key_pressed = True
                elif event_data == K_d:
                    self.d_key_pressed = True
                elif event_data == K_f:
                    # F键：触发前车斜坡速度
                    self.ramp_controller.trigger()
                elif event_data == K_z:
                    # Z键：前车向左换道
                    # 计算当前目标速度（考虑斜坡）
                    if self.ramp_controller.is_ramp_active():
                        current_speed = self.ramp_controller.get_target_speed()
                    else:
                        current_speed = self.target_speed_kmh
                    self.lane_change_controller.change_lane_left(current_speed)
                elif event_data == K_x:
                    # X键：前车向右换道
                    # 计算当前目标速度（考虑斜坡）
                    if self.ramp_controller.is_ramp_active():
                        current_speed = self.ramp_controller.get_target_speed()
                    else:
                        current_speed = self.target_speed_kmh
                    self.lane_change_controller.change_lane_right(current_speed)

            elif event_type == 'keyup':
                if event_data == K_w:
                    self.w_key_pressed = False
                    self.manual_throttle_input = 0.0
                    self.throttle = 0.0
                    print('释放 油门松开')
                elif event_data == K_s:
                    self.s_key_pressed = False
                    self.manual_brake_input = 0.0
                    self.brake = 0.0
                    print('释放 刹车松开')
                elif event_data == K_a:
                    self.a_key_pressed = False
                    if not self.d_key_pressed:
                        self.manual_steering_controller.reset()
                        self.manual_steer_input = 0.0
                elif event_data == K_d:
                    self.d_key_pressed = False
                    if not self.a_key_pressed:
                        self.manual_steering_controller.reset()
                        self.manual_steer_input = 0.0

    def _send_keyboard_command_to_simulink(self, command_code, description):
        """
        记录键盘指令，将在下一帧的主循环中统一处理

        Args:
            command_code (int): 标准指令码 (根据decision.md)
                0=NONE, 1=I0降速, 2=I1增速, 3=I2降距, 4=I3增距,
                5=I4油门, 6=I5制动, 7=I6取消
            description (str): 指令描述
        """
        # 只记录指令，不立即调用Simulink
        self.pending_keyboard_command = {
            'code': command_code,
            'description': description
        }
        print(f"⌨️ 已记录键盘指令: {description} (将在下一帧处理)")

    def get_system_info(self):
        """获取系统状态信息，用于显示 """
        acc_params = self.get_current_parameters()
        return OutputFormatter.format_system_info(
            ego_vehicle=self.ego_vehicle,
            target_vehicle=self.target_vehicle,
            acc_system_enabled=self.acc_system_enabled,
            acc_decision=self.acc_decision,
            acc_params=acc_params,
            throttle=self.throttle,
            brake=self.brake,
            steer=self.steer,
            get_vehicle_speed_func=VehicleUtils.get_vehicle_speed,
            get_vehicle_distance_func=VehicleUtils.get_vehicle_distance
        )

    def calculate_desired_following_distance(self, ego_speed_kmh, time_gap=2.0, min_distance=5.0):
        """使用两模式控制计算期望跟车距离"""
        ego_speed_ms = ego_speed_kmh / 3.6
        desired_distance, control_mode = calculate_two_mode_desired_distance(ego_speed_ms)
        return desired_distance, control_mode

    def generate_target(self):
        """主循环 - 完整集成ACC决策、控制和显示"""
        try:
            self.start_time = time.time()
            frame_count = 0

            print("\n=== ACC Integrated Control System ===")
            print("系统将在Pygame窗口中显示CARLA画面和ACC控制信息")
            print("\n操作流程:")
            print("  1. 手动驾驶到适速(>30km/h)")
            print("  2. 按空格键开启ACC系统(进入待命状态)")
            print("  3. 按E键启动ACC控制(当速启控)或Q键(继承启控,需有历史)")
            print("\n键盘控制:")
            print("  空格: ACC系统开关(必须先按)")
            print("  E: 降速/当速启控  Q: 增速/继承启控(需有历史)")
            print("  R/T: 增距/降距  C: 取消ACC")
            print("  W/S: 油门/刹车  A/D: 转向")
            print("  F: 触发前车斜坡速度(测试ACC跟随响应)")
            print("  Z/X: 前车向左/向右换道")
            print("  P: 调试模式  ESC: 退出")
            print(f"\n当前状态: ACC系统关闭, 请先按空格键开启")
            print("")

            # 性能分析器
            import collections
            perf_times = collections.defaultdict(list)
            perf_start_time = time.time()
            last_perf_report_time = time.time()

            # 性能报告辅助函数
            def print_performance_report(perf_times, elapsed_time):
                """打印性能统计报告 - 包含首次周期分析"""
                print("\n" + "=" * 60)
                print(f"📊 性能分析报告 (运行时间: {elapsed_time:.1f}秒)")
                print("=" * 60)

                if not perf_times:
                    print("⚠️ 暂无性能数据")
                    return

                total_avg = 0
                for key in sorted(perf_times.keys()):
                    times = perf_times[key]
                    if times:
                        avg_ms = (sum(times) / len(times)) * 1000
                        max_ms = max(times) * 1000
                        min_ms = min(times) * 1000
                        first_ms = times[0] * 1000 if len(times) > 0 else 0

                        # 如果有多次调用,计算排除首次的平均值
                        if len(times) > 1:
                            avg_excluding_first_ms = (sum(times[1:]) / (len(times) - 1)) * 1000
                            print(
                                f"{key:20s}: 平均 {avg_ms:6.2f}ms | 最大 {max_ms:6.2f}ms | 最小 {min_ms:6.2f}ms | 次数 {len(times)}")

                            # 如果首次明显慢于平均值,标记出来
                            if first_ms > avg_excluding_first_ms * 1.5:
                                print(
                                    f"{'':20s}  ⚠️  首次: {first_ms:6.2f}ms (慢于后续平均 {avg_excluding_first_ms:6.2f}ms)")
                        else:
                            print(
                                f"{key:20s}: 平均 {avg_ms:6.2f}ms | 最大 {max_ms:6.2f}ms | 最小 {min_ms:6.2f}ms | 次数 {len(times)}")

                        total_avg += avg_ms

                if total_avg > 0:
                    print("-" * 60)
                    print(f"{'总计':20s}: 平均 {total_avg:6.2f}ms/周期")
                    if total_avg > 0:
                        fps = 1000.0 / total_avg
                        print(f"{'理论帧率':20s}: {fps:6.2f} FPS")
                print("=" * 60 + "")

            while self.running:
                # === 性能分析：记录每个周期开始时间 ===
                cycle_start = time.time()

                # 处理事件
                t0 = time.time()
                self.handle_events()
                perf_times['1_events'].append(time.time() - t0)
                # handle_keyboard_input()已禁用，W/S键通过事件驱动方式处理
                # self.handle_keyboard_input()

                # === 按键持续检测和开度累加逻辑（无论ACC是否开启）===
                # W键（油门）持续按下：每帧累加0.1，松开时立即归零
                if self.w_key_pressed:
                    self.manual_throttle_input = min(1.0, self.manual_throttle_input + 0.1)
                else:
                    # W键未按下时，确保油门归零
                    self.manual_throttle_input = 0.0

                # S键（刹车）持续按下：每帧累加0.2，松开时立即归零
                if self.s_key_pressed:
                    self.manual_brake_input = min(1.0, self.manual_brake_input + 0.2)
                else:
                    # S键未按下时，确保刹车归零
                    self.manual_brake_input = 0.0

                # 手动转向按键：基于真实帧间隔的累加逻辑
                current_time = time.time()
                dt = current_time - self._last_manual_steer_update
                self.manual_steer_input = self.manual_steering_controller.update(
                    steer_left=self.a_key_pressed,
                    steer_right=self.d_key_pressed,
                    dt_seconds=dt
                )
                self._last_manual_steer_update = current_time

                # === 换道控制器状态更新 ===
                lane_change_completed = self.lane_change_controller.update()
                if lane_change_completed:
                    # 换道完成，恢复constant velocity
                    if self.ramp_controller.is_ramp_active():
                        # 如果斜坡激活，使用斜坡速度
                        restore_speed = self.ramp_controller.get_target_speed()
                    else:
                        # 否则使用默认速度
                        restore_speed = self.target_speed_kmh
                    target_speed_ms = restore_speed / 3.6
                    self.target_vehicle.enable_constant_velocity(carla.Vector3D(target_speed_ms, 0, 0))
                    print(f"🔄 恢复constant velocity: {restore_speed:.1f} km/h")

                # === 前车速度控制 ===
                # 如果正在换道，跳过constant velocity控制（由TM接管）
                if not self.lane_change_controller.is_lane_changing():
                    # === 斜坡速度控制（优先级最高）===
                    if self.ramp_controller.is_ramp_active():
                        # 斜坡模式激活时，使用斜坡控制器计算的目标速度
                        ramp_target_speed = self.ramp_controller.get_target_speed()
                        if ramp_target_speed is not None and self.target_vehicle:
                            target_speed_ms = ramp_target_speed / 3.6
                            self.target_vehicle.enable_constant_velocity(carla.Vector3D(target_speed_ms, 0, 0))

                    # === 前车速度控制（根据模式选择）===
                    elif not self.use_constant_velocity:
                        # Traffic Manager模式：每周期更新速度控制（根据实时路段限速）
                        if self.target_vehicle:
                            # 获取前车当前路段的限速
                            current_speed_limit = self.target_vehicle.get_speed_limit()

                            # 诊断信息：前车状态监控
                            target_speed_actual = VehicleUtils.get_vehicle_speed(self.target_vehicle)
                            target_location = self.target_vehicle.get_location()
                            target_waypoint = self.world.get_map().get_waypoint(target_location)
                            is_junction = target_waypoint.is_junction if target_waypoint else False

                            # 简洁输出：限速、实际速度、是否在路口
                            print(
                                f"前车状态 | 限速:{current_speed_limit:.1f} km/h | 实际:{target_speed_actual:.1f} km/h | 路口:{is_junction}")

                            # 防御性处理：处理无效的限速值
                            if current_speed_limit is None:
                                print(f"⚠️ 防御性处理: get_speed_limit()返回None（可能原因：车辆刚生成，尚未通过限速标志）")
                                current_speed_limit = 30.0  # 使用默认限速
                            elif current_speed_limit <= 0.0:
                                print(
                                    f"⚠️ 防御性处理: get_speed_limit()返回无效值{current_speed_limit:.1f}（可能原因：地图数据异常）")
                                current_speed_limit = 30.0  # 使用默认限速
                            elif not np.isfinite(current_speed_limit):
                                print(f"⚠️ 防御性处理: get_speed_limit()返回非有限值（Inf或NaN）")
                                current_speed_limit = 30.0  # 使用默认限速

                            # 检查限速是否变化
                            if self.last_speed_limit != current_speed_limit:
                                # 计算速度百分比偏差
                                # percentage = (speed_limit - target_speed) / speed_limit * 100
                                percentage_diff = ((
                                                               current_speed_limit - self.target_speed_kmh) / current_speed_limit) * 100.0

                                # 更新Traffic Manager设置
                                self.tm.vehicle_percentage_speed_difference(self.target_vehicle, percentage_diff)

                                # 输出限速变化信息
                                if self.last_speed_limit is not None:
                                    print(f"\n🚦 路段限速变化: {self.last_speed_limit:.1f} → {current_speed_limit:.1f} km/h")
                                else:
                                    print(f"\n🚦 初始路段限速: {current_speed_limit:.1f} km/h")

                                print(f"   前车目标速度: {self.target_speed_kmh:.1f} km/h")
                                print(f"   速度百分比偏差: {percentage_diff:.1f}%")

                                # 更新记录
                                self.last_speed_limit = current_speed_limit
                # Constant Velocity模式：无需更新，速度已经固定

                # 世界更新
                t0 = time.time()
                if self.world:
                    self.world.tick()
                perf_times['3_world_tick'].append(time.time() - t0)

                t0 = time.time()
                self.display_manager.tick(60)  # 60 FPS
                perf_times['4_display_tick'].append(time.time() - t0)

                # === 定期输出性能报告 (每10秒) ===
                current_time = time.time()
                if current_time - last_perf_report_time >= 10.0:
                    elapsed = current_time - perf_start_time
                    print_performance_report(perf_times, elapsed)
                    last_perf_report_time = current_time

                t0 = time.time()
                # === 获取车辆状态 ===
                ego_speed = VehicleUtils.get_vehicle_speed(self.ego_vehicle)
                target_speed = VehicleUtils.get_vehicle_speed(self.target_vehicle) if self.target_vehicle else 0.0

                # 使用CARLA API获取距离和车道偏移
                vehicle_distance = self.carla_perception.get_vehicle_distance()
                lane_offset = self.carla_perception.get_lane_offset()

                has_target = vehicle_distance < 200.0  # 检测范围：200米

                # === 使用Simulink一体化接口进行决策和控制 ===
                # 获取当前ACC参数（可能被Simulink或用户修改）
                acc_params = self.get_current_parameters()
                acc_status = self.get_status_info()

                # 使用ACC参数中的目标速度
                target_speed_ms = acc_params['V_target_kmh'] / 3.6

                # 为一体化接口准备输入数据
                ego_speed_ms = ego_speed / 3.6

                # 获取增强两模式控制信息用于Simulink接口
                if has_target:
                    current_distance = vehicle_distance
                    enhanced_two_mode_output = enhanced_two_mode_control(ego_speed_ms, current_distance,
                                                                         target_speed_ms)
                    control_error = enhanced_two_mode_output['control_error']
                    control_mode_flag = enhanced_two_mode_output['control_mode_flag']
                else:
                    enhanced_two_mode_output = enhanced_two_mode_control(ego_speed_ms, None, target_speed_ms)
                    control_error = enhanced_two_mode_output['control_error']
                    control_mode_flag = enhanced_two_mode_output['control_mode_flag']

                # === 车辆控制 ===
                # === 使用一体化接口进行决策和控制计算 ===
                # 传递手动油门状态用于扭矩仲裁管理
                manual_throttle_active = hasattr(self, 'manual_throttle_input') and self.manual_throttle_input > 0

                # === 检查是否有待处理的键盘指令（统一调用点）===
                command_description = None  # 初始化为None，用于格式化输出
                if self.pending_keyboard_command:
                    # 使用键盘指令参数
                    command_type = self.pending_keyboard_command['code']
                    command_description = self.pending_keyboard_command['description']
                    # 清除指令（单帧有效）
                    self.pending_keyboard_command = None
                else:
                    # 使用正常的W/S键状态
                    command_type = 0  # 默认NONE
                    if self.w_key_pressed and self.acc_system_enabled:
                        command_type = 5  # I4油门指令
                    elif self.s_key_pressed and self.acc_system_enabled:
                        command_type = 6  # I5刹车指令

                # 数据清洗：确保所有值都是有限数（非Inf/NaN）
                def sanitize_value(value, default=0.0):
                    """清洗数据：将Inf/NaN替换为默认值"""
                    if value is None or not np.isfinite(value):
                        return default
                    return value

                # 准备一体化接口输入数据
                unified_input = {
                    'ego_speed_kmh': sanitize_value(ego_speed, 0.0),
                    'ego_speed_ms': sanitize_value(ego_speed_ms, 0.0),
                    'command_type': command_type,  # 根据按键状态动态设置
                    'command_active': bool(command_type),
                    'manual_throttle_active': manual_throttle_active,
                    'control_error': sanitize_value(control_error, 0.0),
                    'control_mode_flag': control_mode_flag,
                    'V_target_kmh': sanitize_value(acc_params['V_target_kmh'], 50.0),
                    'V_min_kmh': sanitize_value(acc_params['V_min_kmh'], 30.0),
                    'G2_s': sanitize_value(acc_params['G2_s'], 2.0),
                    'timestamp': time.time(),
                    # 外部状态字段（必需） - 全部标量化（21字段输入总线）
                    'external_stage_offset': 0.0,
                    'external_stage': 1.0,
                    'external_error_sign': 0.0,
                    'external_upgrade_count': 0.0,
                    'external_control_error': 0.0,
                    'external_error_derivative': 0.0,
                    'external_error_second_derivative': 0.0
                }

                # 调用一体化接口获取决策+SPPVT输出
                try:
                    t0 = time.time()
                    unified_output = self.acc_decision_sppvt.process_decision_and_control(unified_input)
                    simulink_duration_ms = (time.time() - t0) * 1000  # 保存耗时(毫秒)
                    perf_times['0_simulink'].append((time.time() - t0))

                    # === 同步Simulink输出的参数回到系统 ===
                    # V_target_kmh和G2_s可能被Simulink修改（键盘指令）
                    params_changed = False
                    if 'updated_V_target_kmh' in unified_output:
                        old_V_target = self.acc_params['V_target_kmh']
                        new_V_target = unified_output['updated_V_target_kmh']
                        if abs(new_V_target - old_V_target) > 0.1:
                            self.acc_params['V_target_kmh'] = new_V_target
                            params_changed = True

                    if 'updated_G2_s' in unified_output:
                        old_G2 = self.acc_params['G2_s']
                        new_G2 = unified_output['updated_G2_s']
                        if abs(new_G2 - old_G2) > 0.01:
                            self.acc_params['G2_s'] = new_G2
                            params_changed = True

                    # 同步到 two_mode_controller（关键修复：确保误差计算使用最新G2）
                    if params_changed:
                        set_two_mode_parameters(
                            V_threshold_kmh=self.acc_params['V_target_kmh'],
                            G2_s=self.acc_params['G2_s'],
                            target_speed_kmh=self.acc_params['V_target_kmh']
                        )

                    # 键盘调整将通过Simulink输入输出处理，不再使用备用决策

                    # 为了向后兼容，从统一输出中提取传统的决策输出格式
                    decision_output = {
                        'control_enabled': unified_output['control_enabled'],
                        'current_control_mode': f"SPPVT_{unified_output.get('sppvt_stage', 1)}",
                        'current_decision': unified_output['current_decision'],
                        'torque_arbitration_active': unified_output['torque_arbitration_active']
                    }

                    # === 同步扭矩仲裁状态到acc_decision对象（用于pygame显示）===
                    self.acc_decision_sppvt.torque_arbitration_active = unified_output.get('torque_arbitration_active',
                                                                                           False)

                    # === 实时绘图：添加TIME模式数据 ===
                    if control_mode_flag == 1 and has_target:  # TIME模式且有前车
                        # 提取时距数据
                        desired_time_gap = enhanced_two_mode_output.get('reference_value', 0.0)  # 期望时距
                        actual_time_gap = enhanced_two_mode_output.get('current_value', 0.0)  # 实际时距
                        current_time = time.time() - self.start_time

                        # 添加到实时绘图器（包含速度数据和控制状态）
                        self.realtime_plotter.add_data(
                            desired_time_gap,
                            actual_time_gap,
                            current_time,
                            ego_speed,  # 自车速度 (km/h)
                            target_speed,  # 前车速度 (km/h)
                            decision_output['control_enabled']  # ACC控制状态
                        )

                    # 获取Simulink的原始控制输出（与control_error同符号）
                    control_output = unified_output.get('target_accel', 0.0)

                    # 根据控制模式进行符号转换
                    if control_mode_flag == 1:  # TIME模式
                        # TIME模式下，control_error < 0 表示距离太远需要加速
                        # 但control_output与control_error同符号，所以需要反转
                        sppvt_target_accel = -control_output
                    elif control_mode_flag == 2:  # SPEED模式
                        # SPEED模式下，control_error > 0 表示需要加速
                        # control_output与control_error同符号，直接使用
                        sppvt_target_accel = control_output
                    else:
                        # 未知模式，保守使用原值
                        sppvt_target_accel = control_output

                except Exception as e:
                    print(f"❌ Simulink一体化接口调用失败: {e}")
                    print("错误详情:")
                    import traceback
                    traceback.print_exc()
                    # 不再回退，让问题充分暴露
                    raise e

                # === 扭矩仲裁处理（油门指令时） ===
                torque_arbitration = decision_output.get('torque_arbitration_active', False)

                # === ACC控制执行条件判断 ===
                acc_should_control = (self.acc_system_enabled and
                                      decision_output['control_enabled'])

                # 初始化最终控制信息字典
                final_control = {
                    'throttle': 0.0,
                    'brake': 0.0,
                    'steer': 0.0,
                    'mode': 'MANUAL',
                    'torque_arbitration': False,
                    'sppvt_throttle': 0.0,
                    'driver_throttle': 0.0
                }

                if acc_should_control:
                    # ACC控制模式 - 只有在主动控制模式下才执行
                    try:
                        # === 横向控制：使用增强控制器（PID + 预瞄）===
                        # 1. 获取当前横向偏移
                        current_offset = -lane_offset  # 符号反转：左负右正

                        # 2. 计算预瞄距离（根据车速动态调整）
                        lookahead_distance = self.lateral_controller.calculate_lookahead_distance(ego_speed_ms)

                        # 3. 获取前瞻偏移
                        lookahead_offset = -self.carla_perception.get_lookahead_offset(lookahead_distance)

                        # 4. 使用增强控制器计算转向输出
                        steer_output = self.lateral_controller.update(
                            current_offset=current_offset,
                            lookahead_offset=lookahead_offset,
                            speed_ms=ego_speed_ms,
                            dt=0.05
                        )

                        # Decide control action based on enable flag
                        if decision_output.get('control_enabled', False):
                            # ACC control active; build control command
                            control = carla.VehicleControl()
                            control.manual_gear_shift = False
                            control.gear = 1

                            if sppvt_target_accel is not None:
                                # 使用扭矩到油门转换器（完整RPM模型）
                                # 注意：变量名sppvt_target_accel是历史遗留，实际上SPPVT输出的是发动机扭矩(N·m)
                                sppvt_engine_torque = sppvt_target_accel * 400  # 重命名以明确含义
                                print(f"SPPVT输出扭矩: {sppvt_engine_torque}")
                                if self.use_torque_converter:
                                    # 使用完整RPM模型：发动机扭矩 → 油门/刹车
                                    control.throttle, control.brake = self.torque_converter.engine_torque_to_throttle(
                                        sppvt_engine_torque, ego_speed
                                    )
                                    # 打印油门和刹车
                                    print(f"油门: {control.throttle}, 刹车: {control.brake}")

                                else:
                                    # 简化映射（备用方案，不推荐）
                                    # 假设最大扭矩749 N·m
                                    print("使用了简化方案")
                                    if sppvt_engine_torque > 0:
                                        control.throttle = min(sppvt_engine_torque / 749.0, 1.0)
                                        control.brake = 0.0
                                    else:
                                        control.throttle = 0.0
                                        # 假设总制动扭矩 = 1000 × 4 × 传动比 ≈ 36816 N·m
                                        control.brake = min(abs(sppvt_engine_torque) / 100.0, 1.0)
                            else:
                                # No valid SPPVT output; keep longitudinal command zero
                                control.throttle = 0.0
                                control.brake = 0.0

                            # Apply PID-based lateral steering
                            control.steer = steer_output

                            # === Handle W/S override ===
                            if self.manual_throttle_input > 0:
                                if torque_arbitration:
                                    final_throttle = max(control.throttle, self.manual_throttle_input)
                                    control.throttle = final_throttle
                                else:
                                    control.throttle = self.manual_throttle_input
                                    control.brake = 0.0

                            if self.manual_brake_input > 0:
                                control.throttle = 0.0
                                control.brake = self.manual_brake_input

                        else:
                            # ACC未激活：不执行控制或使用巡航模式
                            control = carla.VehicleControl()

                        if control.brake < 0.01:
                            control.brake = 0.0
                        self.ego_vehicle.apply_control(control)
                        self.throttle = control.throttle
                        self.brake = control.brake
                        self.steer = control.steer

                        # 收集最终控制信息
                        if sppvt_target_accel is not None:
                            # 使用正确的键名：sppvt_stage_output（与Simulink总线定义一致）
                            sppvt_stage = unified_output.get('sppvt_stage_output', 0)
                            # SPPVT阶段：0=未知, 1=Stage1, 2=Stage2, 3=Stage3
                            if sppvt_stage >= 1:
                                current_mode = f"UNIFIED_SPPVT_Stage{int(sppvt_stage)}"
                            else:
                                current_mode = "UNIFIED_SPPVT_Unknown"
                        else:
                            current_mode = decision_output.get('current_control_mode', 'Unknown')

                        final_control.update({
                            'throttle': control.throttle,
                            'brake': control.brake,
                            'steer': control.steer,
                            'mode': current_mode,
                            'torque_arbitration': torque_arbitration,
                            'sppvt_throttle': control.throttle if sppvt_target_accel is not None else 0.0,
                            'driver_throttle': self.manual_throttle_input
                        })

                    except Exception as e:
                        print(f"❌ ACC control error详细信息:")
                        print(f"   错误类型: {type(e).__name__}")
                        print(f"   错误消息: {str(e)}")
                        print(f"   lane_offset: {lane_offset}")
                        print(
                            f"   sppvt_target_accel: {sppvt_target_accel if 'sppvt_target_accel' in locals() else 'Not available'}")
                        print(f"   unified_input: {unified_input if 'unified_input' in locals() else 'Not available'}")
                        import traceback
                        print(f"   完整错误堆栈:")
                        traceback.print_exc()

                else:
                    # ACC不控制时，但仍需处理手动油门/刹车输入
                    # === 处理手动油门/刹车输入（独立于ACC控制状态）===
                    if self.manual_throttle_input > 0 or self.manual_brake_input > 0:
                        control = carla.VehicleControl()
                        manual_steer_active = self.manual_steer_input

                        # 刹车优先级高于油门
                        if self.manual_brake_input > 0:
                            control.throttle = 0.0
                            control.brake = self.manual_brake_input
                            control.steer = manual_steer_active
                        elif self.manual_throttle_input > 0:
                            control.throttle = self.manual_throttle_input
                            control.brake = 0.0
                            control.steer = manual_steer_active

                        if self.ego_vehicle:
                            self.ego_vehicle.apply_control(control)
                        self.throttle = control.throttle
                        self.brake = control.brake
                        self.steer = control.steer

                        # 收集手动控制信息
                        final_control.update({
                            'throttle': control.throttle,
                            'brake': control.brake,
                            'steer': control.steer,
                            'mode': 'MANUAL'
                        })
                    else:
                        manual_steer_active = self.manual_steer_input
                        if manual_steer_active != 0.0:
                            control = carla.VehicleControl()
                            control.throttle = 0.0
                            control.brake = 0.0
                            control.steer = manual_steer_active
                            if self.ego_vehicle:
                                self.ego_vehicle.apply_control(control)
                            self.throttle = control.throttle
                            self.brake = control.brake
                            self.steer = control.steer
                            final_control.update({
                                'steer': manual_steer_active,
                                'mode': 'MANUAL'
                            })
                        else:
                            # 无任何输入时，应用全零控制指令以停止车辆
                            control = carla.VehicleControl()
                            control.throttle = 0.0
                            control.brake = 0.0
                            control.steer = 0.0
                            if self.ego_vehicle:
                                self.ego_vehicle.apply_control(control)
                            self.throttle = 0.0
                            self.brake = 0.0
                            self.steer = 0.0

                # === 收集环境数据用于输出 ===
                # 计算期望距离和距离误差
                desired_distance = enhanced_two_mode_output.get('desired_distance', 0.0) if has_target else 0.0
                distance_error = vehicle_distance - desired_distance if has_target else 0.0

                # 控制模式名称
                control_mode_names = {1: "TIME模式", 2: "SPEED模式"}
                control_mode_name = control_mode_names.get(control_mode_flag, "Unknown")

                env_data = {
                    'ego_speed_kmh': ego_speed,
                    'ego_speed_ms': ego_speed_ms,
                    'target_speed_kmh': target_speed,
                    'target_speed_ms': target_speed / 3.6,
                    'vehicle_distance': vehicle_distance,
                    'desired_distance': desired_distance,
                    'distance_error': distance_error,
                    'has_target': has_target,
                    'lane_offset': lane_offset,
                    'control_error': control_error,
                    'control_mode_flag': control_mode_flag,
                    'control_mode_name': control_mode_name,
                    'control_output': control_output,  # Simulink原始输出
                    'sppvt_target_accel': sppvt_target_accel  # 符号转换后的加速度
                }

                # === 格式化输出Simulink I/O信息 ===
                OutputFormatter.print_simulink_io(
                    frame_num=frame_count,
                    current_time=time.time() - self.start_time,
                    unified_input=unified_input,
                    unified_output=unified_output,
                    duration_ms=simulink_duration_ms,
                    manual_throttle_input=self.manual_throttle_input,
                    manual_brake_input=self.manual_brake_input,
                    w_key_pressed=self.w_key_pressed,
                    s_key_pressed=self.s_key_pressed,
                    acc_system_enabled=self.acc_system_enabled,
                    command_description=command_description,
                    final_control=final_control,
                    env_data=env_data
                )

                # === 数据记录 ===
                current_time = time.time() - self.start_time
                desired_distance, control_mode = self.calculate_desired_following_distance(ego_speed)

                self.csv_writer.writerow([
                    current_time,
                    ego_speed,
                    target_speed,
                    vehicle_distance,
                    desired_distance,
                    control_mode,
                    VehicleUtils.get_lane_offset(self.ego_vehicle, self.world),
                    acc_status['state_description'],
                    self.acc_system_enabled,
                    acc_params['V_target_kmh'],
                    acc_params['V_min_kmh'],
                    acc_params['G2_s'],
                    self.throttle,
                    self.brake,
                    self.steer
                ])
                self.csv_file.flush()

                # === Pygame渲染 ===
                system_info = self.get_system_info()
                self.display_manager.render_display(system_info)

                frame_count += 1

        except KeyboardInterrupt:
            print("\nStopped by user.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # === 输出最终性能报告 ===
            if 'perf_times' in locals() and 'perf_start_time' in locals():
                final_elapsed = time.time() - perf_start_time
                print("\n" + "=" * 60)
                print("🏁 最终性能分析报告")
                print("=" * 60)
                print_performance_report(perf_times, final_elapsed)

            print("Cleaning up...")
            if self.csv_file:
                self.csv_file.close()
            self.display_manager.destroy()
            self.destroy()

    def destroy(self):
        # 停止实时绘图器
        if hasattr(self, 'realtime_plotter'):
            self.realtime_plotter.stop()

        # 禁用前车的constant velocity（如果启用）
        if hasattr(self, 'use_constant_velocity') and self.use_constant_velocity:
            if self.target_vehicle:
                self.target_vehicle.disable_constant_velocity()
                print("✅ 已禁用前车的constant velocity模式")

        # 销毁车辆
        for vehicle in self.vehicles:
            vehicle.destroy()
        self.ego_vehicle.destroy()

        # 恢复异步模式
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        print(f"Destroyed {len(self.vehicles)} vehicles, ego vehicle, and restored settings.")


def main():
    """
    主函数 - 使用CARLA API模式

    Usage:
        python acc_updated.py
    """
    # 创建ACC实例
    acc_actor = acc()

    try:
        acc_actor.generate_target()
    except KeyboardInterrupt:
        print("Program interrupted.")


if __name__ == '__main__':
    main()
