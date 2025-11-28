import csv
import time
import argparse

import carla
import numpy as np
# Pygame相关
import pygame
from pygame.locals import *

from acc_config import ACCConfig
from acc_control_facade import ACCControlFacade
# 导入显示管理器
from display_manager import DisplayManager
# 导入增强横向控制器（PID + 预瞄）
from enhanced_lateral_controller import EnhancedLateralController
# 导入换道控制器
from lane_change_controller import LaneChangeController
from manual_input_controller import ManualSteeringController
from output_formatter import OutputFormatter
from cut_in_scenario import CutInScenarioManager
from cut_out_scenario import CutOutScenarioManager
# 导入斜坡速度控制器
from ramp_speed_controller import RampSpeedController
# 导入实时绘图器
from realtime_time_gap_plotter import RealtimeTimeGapPlotter
# 导入结果保存绘图器（可选，用于保存结果）
from result_plotter import RealtimeResultPlotter
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
    def __init__(self, use_result_plotter=None, scenario_mode=None):
        """
        初始化ACC系统
        使用CARLA API直接获取前车距离和车道信息

        Args:
            use_result_plotter: True=使用结果保存画图器, False=使用实时测试画图器
                               None=使用配置文件默认值
        """
        # === 加载配置 ===
        self.config = ACCConfig()

        # 根据入参选择工况模式（互斥：none / cut-in / cut-out）
        if scenario_mode == 'none':
            self.config.enable_cut_in_scenario = False
            self.config.enable_cut_out_scenario = False
        elif scenario_mode == 'cut-in':
            self.config.enable_cut_in_scenario = True
            self.config.enable_cut_out_scenario = False
        elif scenario_mode == 'cut-out':
            self.config.enable_cut_in_scenario = False
            self.config.enable_cut_out_scenario = True
        elif scenario_mode is None:
            # 按配置文件：若两者都开，默认优先切出
            if self.config.enable_cut_out_scenario and self.config.enable_cut_in_scenario:
                self.config.enable_cut_in_scenario = False

        # 允许通过参数覆盖配置
        if use_result_plotter is not None:
            self.config.use_result_plotter = use_result_plotter
        self.use_result_plotter = self.config.use_result_plotter

        # === 显示管理器初始化 ===
        self.display_manager = DisplayManager(self.config.display_width, self.config.display_height)

        # === CARLA API感知模块 ===
        self.carla_perception = None  # 将在init_carla()后初始化

        # === 通用变量 ===
        self.max_follow_distance = self.config.max_follow_distance
        self.image_width = self.config.display_width
        self.image_height = self.config.display_height
        self.target_vehicle = None
        self.start_time = None
        self.csv_file = None
        self.csv_writer = None

        # === ACC混合控制器模块 ===
        # Python决策 + Simulink SPPVT控制的混合架构
        self.acc_decision_sppvt = ACCControlFacade(
            config=self.config,
            debug=self.config.acc_decision_debug
        )

        # === ACC系统可配置参数 (环境相关，需要传递给Simulink) ===
        self.acc_params = self.config.get_acc_params()
        # 初始化两模式控制参数与配置一致
        set_two_mode_parameters(
            V_threshold_kmh=self.acc_params['V_target_kmh'],
            G2_s=self.acc_params['G2_s'],
            target_speed_kmh=self.acc_params['V_target_kmh']
        )

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
        self.cut_in_manager = None
        self.cut_in_vehicle = None
        self.cut_out_manager = None
        self.cut_out_front_vehicle = None
        self.cut_out_follow_vehicle = None
        self.sim_elapsed_s = 0.0

        # 按键按下状态跟踪
        self.w_key_pressed = False  # W键（油门）是否按下
        self.s_key_pressed = False  # S键（刹车）是否按下
        self.a_key_pressed = False  # A键（左转）是否按下
        self.d_key_pressed = False  # D键（右转）是否按下

        # 键盘指令队列（单帧有效，只记录指令不立即调用Simulink）
        self.pending_keyboard_command = None  # 格式: {'code': int, 'description': str}

        # === 横向控制器 ===
        # 使用增强横向控制器（PID + 预瞄）
        lateral_params = self.config.get_lateral_controller_params()
        self.lateral_controller = EnhancedLateralController(
            kp=lateral_params['kp'],
            ki=lateral_params['ki'],
            kd=lateral_params['kd']
        )

        # 可选预瞄参数配置（如果配置文件中定义了）
        if hasattr(self.config, 'lateral_lookahead_params'):
            lookahead = self.config.lateral_lookahead_params
            self.lateral_controller.set_weights(
                weight_current=lookahead['weight_current'],
                weight_lookahead=lookahead['weight_lookahead']
            )
            self.lateral_controller.set_lookahead_params(
                base=lookahead['base_distance'],
                gain=lookahead['gain']
            )

        # === 运行控制 ===
        self.running = True

        # 初始化CARLA
        self.init_carla()
        self.init_csv()

        # 初始化两模式参数 - 已集成到Simulink，无需外部同步

        # 初始化一体化接口的两模式控制器
        if hasattr(self.acc_decision_sppvt, 'init_two_mode_controller'):
            self.acc_decision_sppvt.init_two_mode_controller()

        # === 初始化绘图器 ===
        if self.use_result_plotter:
            # 使用结果保存画图器（Step模式，ACC开启后记录，自动保存CSV和PNG）
            self.realtime_plotter = RealtimeResultPlotter(
                max_points=self.config.plotter_max_points,
                update_interval=self.config.plotter_update_interval
            )
            self.realtime_plotter.start()
            print("✅ 结果保存绘图器已启动 (Step模式，自动保存)")
        else:
            # 使用实时测试画图器（Time模式，有滑动条控件）
            self.realtime_plotter = RealtimeTimeGapPlotter(
                max_points=self.config.plotter_max_points,
                update_interval=self.config.plotter_update_interval
            )
            self.realtime_plotter.start()
            print("✅ 实时测试绘图器已启动 (Time模式)")

    def init_carla(self):
        # 初始化 Carla 客户端（使用配置）
        self.client = carla.Client(self.config.carla_host, self.config.carla_port)

        self.client.set_timeout(self.config.carla_timeout)
        try:
            self.world = self.client.get_world()
            self.world = self.client.load_world(
                self.config.map_name,
                carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles
            )
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load map {self.config.map_name}: {e}")

        # 设置同步模式（放宽时间步长以匹配Simulink处理能力）
        settings = self.world.get_settings()
        settings.synchronous_mode = self.config.synchronous_mode
        settings.fixed_delta_seconds = self.config.fixed_delta_seconds
        self.world.apply_settings(settings)

        # 获取蓝图库和地图
        self.blueprint_library = self.world.get_blueprint_library()
        map = self.world.get_map()

        # 获取车辆蓝图（使用配置）
        vehicle_bp = self.blueprint_library.filter(self.config.target_vehicle_blueprint)[0]
        ego_vehicle_bp = self.blueprint_library.filter(self.config.ego_vehicle_blueprint)[0]

        # 定义固定生成点（使用配置；切入/切出工况下改用预设前车位置）
        if self.config.enable_cut_out_scenario and self.config.cut_out_front_spawn_location:
            spawn_location = self.config.cut_out_front_spawn_location
        elif self.config.enable_cut_in_scenario and self.config.cut_in_target_spawn_location:
            spawn_location = self.config.cut_in_target_spawn_location
        else:
            spawn_location = self.config.spawn_location
        # 其他可选位置（注释保留供参考）:
        # right x=1654.013672, y=6322.584473, z=-0.009410
        # left x=897.991394, y=5805.217773, z=-0.009344
        # up  x=1964.847778, y=-4824.012207, z=-0.009341
        # down x=2127.969482, y=-3362.775146, z=54.469646
        waypoint = map.get_waypoint(
            spawn_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            raise RuntimeError("Failed to find a valid waypoint near the specified location")

        # 生成目标车辆
        spawn_point = waypoint.transform
        spawn_point.location.z += self.config.spawn_z_offset

        vehicles = []
        target_vehicle = None

        # 优先处理切出工况：生成两个前车（lead在最前，cut在中间将切出），初始跟车对象为cut
        if self.config.enable_cut_out_scenario:
            lead_wp = map.get_waypoint(self.config.cut_out_front_spawn_location, project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            if lead_wp is None:
                raise RuntimeError("Cut-out spawn locations are not on drivable lanes")

            lead_tf = lead_wp.transform
            lead_tf.location = self.config.cut_out_front_spawn_location

            # cut车辆：沿lead_wp反向后移指定距离，保持同车道
            cut_prev = lead_wp.previous(self.config.cut_out_cut_back_distance_m)
            if not cut_prev:
                raise RuntimeError("Failed to find waypoint for cut-out cut vehicle")
            cut_tf = cut_prev[0].transform
            cut_tf.location.z += self.config.spawn_z_offset

            lead_vehicle = self.world.try_spawn_actor(vehicle_bp, lead_tf)
            cut_vehicle = self.world.try_spawn_actor(vehicle_bp, cut_tf)
            if lead_vehicle is None or cut_vehicle is None:
                raise RuntimeError("Failed to spawn cut-out scenario vehicles")
            vehicles.extend([lead_vehicle, cut_vehicle])
            target_vehicle = cut_vehicle  # 初始跟车目标=将要切出的车辆（中间车）
            self.cut_out_lead_vehicle = lead_vehicle
            self.cut_out_cut_vehicle = cut_vehicle
            lead_vehicle.set_autopilot(True)
            cut_vehicle.set_autopilot(True)

        else:
            # 目标车：若启用切入工况，使用预设位置；否则使用waypoint位置
            if self.config.enable_cut_in_scenario and self.config.cut_in_target_spawn_location is not None:
                tgt_wp = map.get_waypoint(self.config.cut_in_target_spawn_location, project_to_road=True,
                                          lane_type=carla.LaneType.Driving)
                if tgt_wp is not None:
                    tgt_transform = tgt_wp.transform
                    tgt_transform.location = carla.Location(self.config.cut_in_target_spawn_location.x,
                                                            self.config.cut_in_target_spawn_location.y,
                                                            self.config.cut_in_target_spawn_location.z + 0.0)
                    target_vehicle = self.world.try_spawn_actor(vehicle_bp, tgt_transform)
                else:
                    print("⚠️ 切入工况预设前车坐标不在可行驶车道，回退到默认spawn点")
            if target_vehicle is None:
                target_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if target_vehicle is None:
                raise RuntimeError("Failed to spawn target vehicle at waypoint location")
            vehicles.append(target_vehicle)
            target_vehicle.set_autopilot(True)

        self.target_vehicle = target_vehicle

        # === Traffic Manager速度控制配置 ===
        # 假设的道路限速（根据实际CARLA地图调整）
        # 如果前车速度不对，需要调整这个值来匹配实际道路限速
        self.assumed_road_speed_limit_kmh = self.config.assumed_road_speed_limit_kmh

        # === 斜坡速度控制器配置 ===
        # 初始化斜坡速度控制器（使用配置）
        ramp_params = self.config.get_ramp_controller_params()
        self.ramp_start_speed_kmh = ramp_params['start_speed_kmh']
        self.ramp_target_speed_kmh = ramp_params['target_speed_kmh']
        self.ramp_duration_s = ramp_params['duration_s']

        self.ramp_controller = RampSpeedController(
            start_speed_kmh=self.ramp_start_speed_kmh,
            target_speed_kmh=self.ramp_target_speed_kmh,
            duration_s=self.ramp_duration_s
        )

        # 前车速度配置（使用constant velocity定速巡航）
        # 初始速度设为斜坡起始速度，按F键后触发斜坡变化
        self.target_speed_kmh = self.config.target_speed_kmh  # 使用配置的初始速度
        self.use_constant_velocity = self.config.use_constant_velocity

        # 生成自车：沿车道前进方向偏移一定距离以避免碰撞（相对于目标车所在车道）
        # 自车生成：切出工况时以切出车位置为基准后移，其余沿目标车基准点后移
        ego_base_location = (self.config.cut_out_front_spawn_location
                             if self.config.enable_cut_out_scenario and self.config.cut_out_front_spawn_location
                             else spawn_location)
        ego_base_wp = map.get_waypoint(ego_base_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_base_wp is None:
            raise RuntimeError("Failed to find ego base waypoint for spawn")
        ego_waypoints = ego_base_wp.previous(self.config.ego_spawn_distance)
        if not ego_waypoints:
            raise RuntimeError("Failed to find a waypoint for ego vehicle spawn")
        ego_spawn_point = ego_waypoints[0].transform
        ego_spawn_point.location.z += self.config.spawn_z_offset
        self.ego_vehicle = self.world.try_spawn_actor(ego_vehicle_bp, ego_spawn_point)
        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        self.vehicles = vehicles
        self.ego_vehicle.set_autopilot(False)

        # 初始化扭矩到油门转换器（使用配置）
        print("初始化扭矩到油门转换器...")
        self.torque_converter = TorqueToThrottleConverter(self.ego_vehicle)
        self.use_torque_converter = self.config.use_torque_converter

        # 初始化显示管理器的相机
        self.display_manager.init_camera_manager(self.ego_vehicle)

        # 设置交通管理器（必须与CARLA世界同步模式一致）
        tm = self.client.get_trafficmanager(self.config.tm_port)
        tm.set_global_distance_to_leading_vehicle(self.config.tm_global_distance)
        tm.set_synchronous_mode(self.config.synchronous_mode)
        self.tm_port = tm.get_port()
        tm.auto_lane_change(self.ego_vehicle, False)

        # 保存Traffic Manager引用供后续使用
        self.tm = tm

        # 可选：初始化切入工况，生成侧向待切入车辆
        if self.config.enable_cut_in_scenario:
            self.cut_in_manager = CutInScenarioManager(
                world=self.world,
                traffic_manager=tm,
                ego_vehicle=self.ego_vehicle,
                lead_vehicle=self.target_vehicle,
                blueprint_library=self.blueprint_library,
                enable_from_left=self.config.cut_in_from_left,
                trigger_time_s=self.config.cut_in_trigger_time_s,
                lateral_threshold_m=self.config.cut_in_lateral_threshold_m,
                vehicle_blueprint=self.config.cut_in_vehicle_blueprint,
                side_spawn_location=self.config.cut_in_side_spawn_location,
                side_back_offset=self.config.cut_in_side_spawn_back_offset_m,
                side_spawn_z_lift=self.config.cut_in_side_spawn_z_lift_m,
                lane_entry_margin=self.config.cut_in_lane_entry_margin_m
            )
            if self.cut_in_manager and self.cut_in_manager.cut_in_vehicle:
                self.cut_in_vehicle = self.cut_in_manager.cut_in_vehicle
                vehicles.append(self.cut_in_vehicle)
                print("✅ 切入工况：侧向车辆已加入控制列表")
        elif self.config.enable_cut_out_scenario:
            # 初始化切出工况管理器（cut_vehicle为初始跟车目标，lead_vehicle为最前车）
            self.cut_out_manager = CutOutScenarioManager(
                world=self.world,
                traffic_manager=tm,
                ego_vehicle=self.ego_vehicle,
                cut_vehicle=self.cut_out_cut_vehicle,
                lead_vehicle=self.cut_out_lead_vehicle,
                blueprint_library=self.blueprint_library,
                trigger_time_s=self.config.cut_out_trigger_time_s,
                change_to_right=self.config.cut_out_change_to_right
            )
            print("✅ 切出工况管理器已初始化（初始跟车=切出车，切出后切到前车）")

        # === 初始化换道控制器 ===
        self.lane_change_controller = LaneChangeController(
            tm,
            self.target_vehicle,
            assumed_road_speed_limit_kmh=self.config.assumed_road_speed_limit_kmh
        )
        print("✅ 前车换道控制器已初始化 (Z键向左, X键向右)")

        if self.use_constant_velocity:
            # === 使用Traffic Manager速度控制（替代constant_velocity）===
            # constant_velocity与autopilot冲突，改用TM速度控制
            for vehicle in vehicles:
                vehicle.set_autopilot(True, self.tm_port)
                # 切入/切出场景下禁用自动换道，避免TM提前变道
                allow_lane_change = not (self.config.enable_cut_in_scenario or self.config.enable_cut_out_scenario)
                tm.auto_lane_change(vehicle, allow_lane_change)
                tm.ignore_lights_percentage(vehicle, 100.0)  # 忽略红绿灯

                # 设置目标速度（使用speed_limit百分比机制）
                # percentage_diff正值=减速，负值=超速
                percentage_diff = ((
                                               self.assumed_road_speed_limit_kmh - self.target_speed_kmh) / self.assumed_road_speed_limit_kmh) * 100.0
                tm.vehicle_percentage_speed_difference(vehicle, percentage_diff)

                # 设置其他TM参数，使速度更稳定
                tm.distance_to_leading_vehicle(vehicle, self.config.tm_target_vehicle_distance)

            print(f"✅ 前车速度配置完成 (Traffic Manager模式):")
            print(f"   假设道路限速: {self.assumed_road_speed_limit_kmh:.1f} km/h")
            print(f"   目标速度: {self.target_speed_kmh:.1f} km/h")
            print(f"   速度控制: Traffic Manager (百分比差值{percentage_diff:.1f}%)")
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
        """初始化CSV文件（使用配置）"""
        self.csv_file = open(self.config.csv_output_file, 'w', newline='')
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

    def get_current_parameters(self):
        """获取当前ACC参数，直接使用Simulink管理的参数"""
        return self.acc_params.copy()

    def get_status_info(self):
        """获取ACC状态信息，基于混合架构"""
        return {
            'state_description': 'Hybrid Python+Simulink Mode',
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

    def _set_target_vehicle(self, vehicle):
        """统一更新前车引用（感知/换道控制器同步）"""
        self.target_vehicle = vehicle
        if hasattr(self, 'carla_perception') and self.carla_perception:
            self.carla_perception.target_vehicle = vehicle
        if hasattr(self, 'lane_change_controller') and self.lane_change_controller:
            self.lane_change_controller.vehicle = vehicle

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
            self.sim_elapsed_s = 0.0
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
                # W键（油门）持续按下：每帧累加配置步长，松开时立即归零
                if self.w_key_pressed:
                    self.manual_throttle_input = min(1.0, self.manual_throttle_input + self.config.manual_throttle_step)
                else:
                    # W键未按下时，确保油门归零
                    self.manual_throttle_input = 0.0

                # S键（刹车）持续按下：每帧累加配置步长，松开时立即归零
                if self.s_key_pressed:
                    self.manual_brake_input = min(1.0, self.manual_brake_input + self.config.manual_brake_step)
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
                    # 换道完成，恢复TM速度控制
                    if self.ramp_controller.is_ramp_active():
                        # 如果斜坡激活，使用斜坡速度
                        restore_speed = self.ramp_controller.get_target_speed()
                    else:
                        # 否则使用默认速度
                        restore_speed = self.target_speed_kmh

                    # 使用TM速度控制
                    percentage_diff = ((
                                                   self.assumed_road_speed_limit_kmh - restore_speed) / self.assumed_road_speed_limit_kmh) * 100.0
                    self.tm.vehicle_percentage_speed_difference(self.target_vehicle, percentage_diff)
                    print(f"🔄 恢复TM速度控制: {restore_speed:.1f} km/h (百分比{percentage_diff:.1f}%)")

                # === 前车速度控制 ===
                # 如果正在换道，跳过constant velocity控制（由TM接管）
                if not self.lane_change_controller.is_lane_changing():
                    # === 斜坡速度控制（优先级最高）===
                    if self.ramp_controller.is_ramp_active():
                        # 斜坡模式激活时，使用斜坡控制器计算的目标速度
                        ramp_target_speed = self.ramp_controller.get_target_speed()
                        if ramp_target_speed is not None and self.target_vehicle:
                            # 使用TM速度控制（每帧更新）
                            percentage_diff = ((
                                                           self.assumed_road_speed_limit_kmh - ramp_target_speed) / self.assumed_road_speed_limit_kmh) * 100.0
                            self.tm.vehicle_percentage_speed_difference(self.target_vehicle, percentage_diff)

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
                                    print(
                                        f"\n🚦 路段限速变化: {self.last_speed_limit:.1f} → {current_speed_limit:.1f} km/h")
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

                # === 定期输出性能报告（使用配置的间隔）===
                current_time = time.time()
                if current_time - last_perf_report_time >= self.config.performance_report_interval:
                    elapsed = current_time - perf_start_time
                    print_performance_report(perf_times, elapsed)
                    last_perf_report_time = current_time
                # 由于同步模式固定步长，使用固定delta积累更符合触发时间控制
                self.sim_elapsed_s += self.config.fixed_delta_seconds

                t0 = time.time()
                # === 获取车辆状态 ===
                # 切入工况更新（触发变道、尝试切换跟车目标）
                self._update_cut_in_scenario(self.sim_elapsed_s)
                # 切出工况更新（触发变道、尝试切换跟车目标）
                self._update_cut_out_scenario(self.sim_elapsed_s)

                ego_speed = VehicleUtils.get_vehicle_speed(self.ego_vehicle)
                target_speed = VehicleUtils.get_vehicle_speed(self.target_vehicle) if self.target_vehicle else 0.0

                # 使用CARLA API获取距离和车道偏移
                vehicle_distance = self.carla_perception.get_vehicle_distance()
                lane_offset = self.carla_perception.get_lane_offset()

                has_target = vehicle_distance < self.config.detection_range  # 检测范围（使用配置）

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
                        'current_control_mode': f"SPPVT_{unified_output.get('sppvt_stage_output', 1)}",
                        'current_decision': unified_output['current_decision'],
                        'torque_arbitration_active': unified_output['torque_arbitration_active']
                    }

                    # === 同步扭矩仲裁状态到acc_decision对象（用于pygame显示）===
                    # 保存决策层的原始标志，显示层可做平滑处理
                    self.acc_decision_sppvt.torque_arbitration_active_raw = unified_output.get(
                        'torque_arbitration_active', False)
                    self.acc_decision_sppvt.torque_arbitration_active = unified_output.get(
                        'torque_arbitration_active', False)

                    # === 实时绘图：添加TIME模式数据 ===
                    if control_mode_flag == 1 and has_target:  # TIME模式且有前车
                        # 提取时距数据
                        desired_time_gap = enhanced_two_mode_output.get('reference_value', 0.0)  # 期望时距
                        actual_time_gap = enhanced_two_mode_output.get('current_value', 0.0)  # 实际时距
                        # 使用步数作为横轴（避免将步长误解为秒）
                        self.realtime_plotter.add_data(
                            desired_time_gap,
                            actual_time_gap,
                            frame_count,  # step index
                            ego_speed,  # 自车速度 (km/h)
                            target_speed,  # 前车速度 (km/h)
                            decision_output['control_enabled']  # ACC控制状态
                        )

                    # 获取SPPVT的原始控制输出
                    control_output = unified_output.get('sppvt_control_output', 0.0)

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

                # 显示用的仲裁标志：仅在ACC在控且有人为油门输入时显示，避免ACC未开启时误亮
                display_torque_arbitration = bool(acc_should_control and (
                        self.w_key_pressed or self.manual_throttle_input > 0))
                self.torque_arbitration_display = display_torque_arbitration
                self.acc_decision_sppvt.torque_arbitration_active = display_torque_arbitration

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
                                # === SPPVT输出缩放转换 ===
                                # 根据正负值使用不同的缩放系数
                                if sppvt_target_accel >= 0:
                                    # 加速模式：SPPVT输出 → 发动机扭矩 (N·m)
                                    sppvt_engine_torque = sppvt_target_accel * self.config.sppvt_accel_scale
                                    print(f"[加速] SPPVT输出: {sppvt_target_accel:.3f} × {self.config.sppvt_accel_scale} = 扭矩: {sppvt_engine_torque:.2f} N·m")
                                else:
                                    # 减速模式：SPPVT输出 → 减速度 (m/s²)
                                    sppvt_engine_torque = sppvt_target_accel * self.config.sppvt_decel_scale
                                    print(f"[减速] SPPVT输出: {sppvt_target_accel:.3f} × {self.config.sppvt_decel_scale} = 减速度: {sppvt_engine_torque:.2f} m/s²")
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

                            # Apply PID-based lateral steering，手动转向输入优先
                            if abs(self.manual_steer_input) > 1e-3:
                                control.steer = self.manual_steer_input
                            else:
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

                # 显示输出使用防误亮/防闪烁的仲裁标志
                display_unified_output = dict(unified_output)
                display_unified_output['torque_arbitration_active'] = display_torque_arbitration

                # === 格式化输出Simulink I/O信息 ===
                OutputFormatter.print_simulink_io(
                    frame_num=frame_count,
                    current_time=time.time() - self.start_time,
                    unified_input=unified_input,
                    unified_output=display_unified_output,
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

        # 清理混合控制器资源
        if hasattr(self, 'acc_decision_sppvt'):
            self.acc_decision_sppvt.cleanup()

        # 前车速度控制已由TM管理，无需手动禁用

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

    def _update_cut_in_scenario(self, elapsed_time_s: float):
        """切入工况的周期更新：触发变道并在满足条件时切换跟车目标。"""
        if not self.cut_in_manager:
            return

        self.cut_in_manager.maybe_trigger_cut_in(elapsed_time_s)

        if (self.cut_in_manager.cut_in_vehicle and
                self.target_vehicle != self.cut_in_manager.cut_in_vehicle and
                self.cut_in_manager.should_switch_to_cut_in()):
            print("🚗 切入车辆已进入本车道，切换跟车目标")
            self._set_target_vehicle(self.cut_in_manager.cut_in_vehicle)

    def _update_cut_out_scenario(self, elapsed_time_s: float):
        """切出工况的周期更新：触发前车切出并在离开本车道后切换跟车目标。"""
        if not self.cut_out_manager:
            return

        self.cut_out_manager.maybe_trigger_cut_out(elapsed_time_s)

        # 切出车离开本车道后，切换跟车目标到留在本车道的车辆
        if self.cut_out_manager._lane_change_completed:
            if self.cut_out_lead_vehicle and self.target_vehicle != self.cut_out_lead_vehicle:
                print("🚗 切出车已离开车道，切换跟车目标到最前车")
                self._set_target_vehicle(self.cut_out_lead_vehicle)
            # 重置状态，停止重试
            self.cut_out_manager._lane_change_triggered = False
            self.cut_out_manager._lane_change_completed = False
            self.cut_out_manager._scenario_completed = True


def main():
    # 手动切换工况 "none" / "cut-in" / "cut-out"
    SCENARIO_MODE = "cut-in"  # None=按配置文件，"none"=普通，"cut-in"=切入，"cut-out"=切出

    # 启用结果保存画图器
    USE_RESULT_PLOTTER = False

    # 创建ACC实例
    acc_actor = acc(use_result_plotter=USE_RESULT_PLOTTER, scenario_mode=SCENARIO_MODE)

    try:
        acc_actor.generate_target()
    except KeyboardInterrupt:
        print("Program interrupted.")


if __name__ == '__main__':
    main()
