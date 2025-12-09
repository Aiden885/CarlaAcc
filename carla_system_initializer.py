"""
CARLA系统初始化器
负责CARLA世界、车辆、传感器的初始化
从acc_updated.py的init_carla方法中提取
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import carla

from acc_config import ACCConfig
from carla_perception import CarlaPerception
from cut_in_scenario import CutInScenarioManager
from cut_out_scenario import CutOutScenarioManager
from lane_change_controller import LaneChangeController
from ramp_speed_controller import RampSpeedController
from torque_to_throttle_converter import TorqueToThrottleConverter


@dataclass
class CarlaSystemResources:
    """CARLA系统资源封装"""
    client: carla.Client
    world: carla.World
    blueprint_library: carla.BlueprintLibrary
    traffic_manager: carla.TrafficManager

    # 车辆
    ego_vehicle: carla.Vehicle
    target_vehicle: carla.Vehicle
    other_vehicles: List[carla.Vehicle]

    # 传感器和感知
    carla_perception: CarlaPerception

    # 控制器
    torque_converter: TorqueToThrottleConverter
    ramp_controller: RampSpeedController
    lane_change_controller: LaneChangeController

    # 场景管理器（可选）
    cut_in_manager: Optional[CutInScenarioManager] = None
    cut_out_manager: Optional[CutOutScenarioManager] = None
    cut_out_lead_vehicle: Optional[carla.Vehicle] = None
    cut_out_cut_vehicle: Optional[carla.Vehicle] = None

    def get_all_actors(self) -> List[carla.Actor]:
        """获取所有需要管理的actors"""
        actors = [self.ego_vehicle, self.target_vehicle] + self.other_vehicles

        # 添加场景车辆
        if self.cut_out_lead_vehicle:
            actors.append(self.cut_out_lead_vehicle)
        if self.cut_out_cut_vehicle and self.cut_out_cut_vehicle not in actors:
            actors.append(self.cut_out_cut_vehicle)
        if self.cut_in_manager and self.cut_in_manager.cut_in_vehicle:
            if self.cut_in_manager.cut_in_vehicle not in actors:
                actors.append(self.cut_in_manager.cut_in_vehicle)

        return [a for a in actors if a is not None]


class CarlaSystemInitializer:
    """
    CARLA系统初始化器
    职责：
    1. 连接CARLA服务器
    2. 加载地图
    3. 设置同步模式
    4. 生成车辆
    5. 配置交通管理器
    6. 初始化传感器和感知模块
    """

    def __init__(self, config: ACCConfig):
        self.config = config

    def initialize(self) -> CarlaSystemResources:
        """
        初始化CARLA系统

        Returns:
            CarlaSystemResources: 初始化后的系统资源
        """
        print("\n" + "=" * 60)
        print("🚀 初始化CARLA系统...")
        print("=" * 60)

        # 1. 连接客户端
        client, world = self._init_client_and_world()

        # 2. 设置同步模式
        self._setup_synchronous_mode(world)

        # 3. 获取蓝图库和地图
        blueprint_library = world.get_blueprint_library()
        carla_map = world.get_map()

        # 4. 生成车辆
        vehicles_data = self._spawn_vehicles(world, blueprint_library, carla_map)

        # 5. 配置交通管理器
        traffic_manager = self._setup_traffic_manager(
            client,
            vehicles_data['all_vehicles'],
            vehicles_data['target_vehicle']
        )

        # 6. 初始化场景管理器
        cut_in_manager, cut_out_manager = self._init_scenario_managers(
            world,
            traffic_manager,
            blueprint_library,
            vehicles_data
        )

        # 7. 初始化控制器
        torque_converter = TorqueToThrottleConverter(vehicles_data['ego_vehicle'])
        ramp_controller = self._init_ramp_controller()
        lane_change_controller = LaneChangeController(
            traffic_manager,
            vehicles_data['target_vehicle'],
            assumed_road_speed_limit_kmh=self.config.assumed_road_speed_limit_kmh
        )

        # 8. 初始化感知模块
        carla_perception = CarlaPerception(
            world,
            vehicles_data['ego_vehicle'],
            vehicles_data['target_vehicle']
        )

        # 9. 设置交通灯
        self._setup_traffic_lights(world)

        print("=" * 60)
        print("✅ CARLA系统初始化完成")
        print("=" * 60)

        # 10. 封装资源
        return CarlaSystemResources(
            client=client,
            world=world,
            blueprint_library=blueprint_library,
            traffic_manager=traffic_manager,
            ego_vehicle=vehicles_data['ego_vehicle'],
            target_vehicle=vehicles_data['target_vehicle'],
            other_vehicles=vehicles_data['other_vehicles'],
            carla_perception=carla_perception,
            torque_converter=torque_converter,
            ramp_controller=ramp_controller,
            lane_change_controller=lane_change_controller,
            cut_in_manager=cut_in_manager,
            cut_out_manager=cut_out_manager,
            cut_out_lead_vehicle=vehicles_data.get('cut_out_lead_vehicle'),
            cut_out_cut_vehicle=vehicles_data.get('cut_out_cut_vehicle')
        )

    def _init_client_and_world(self) -> Tuple[carla.Client, carla.World]:
        """初始化CARLA客户端和世界"""
        print(f"\n1️⃣ 连接CARLA服务器 ({self.config.carla_host}:{self.config.carla_port})...")
        client = carla.Client(self.config.carla_host, self.config.carla_port)
        client.set_timeout(self.config.carla_timeout)

        try:
            world = client.get_world()
            print(f"   当前地图: {world.get_map().name}")

            if world.get_map().name != self.config.map_name:
                print(f"   加载地图: {self.config.map_name}...")
                world = client.load_world(
                    self.config.map_name,
                    carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles
                )
                print(f"   ✅ 地图加载成功")
            else:
                print(f"   ✅ 地图已是目标地图，跳过加载")

        except RuntimeError as e:
            raise RuntimeError(f"Failed to load map {self.config.map_name}: {e}")

        return client, world

    def _setup_synchronous_mode(self, world: carla.World):
        """设置同步模式"""
        print(f"\n2️⃣ 配置同步模式...")
        settings = world.get_settings()
        settings.synchronous_mode = self.config.synchronous_mode
        settings.fixed_delta_seconds = self.config.fixed_delta_seconds
        world.apply_settings(settings)
        print(f"   同步模式: {self.config.synchronous_mode}")
        print(f"   固定步长: {self.config.fixed_delta_seconds}s")

    def _spawn_vehicles(self, world, blueprint_library, carla_map) -> dict:
        """
        生成车辆

        Returns:
            dict: 包含所有生成的车辆信息
        """
        print(f"\n3️⃣ 生成车辆...")

        # 获取蓝图
        vehicle_bp = blueprint_library.filter(self.config.target_vehicle_blueprint)[0]
        ego_vehicle_bp = blueprint_library.filter(self.config.ego_vehicle_blueprint)[0]

        # 确定生成位置
        spawn_location = self._determine_spawn_location()
        waypoint = carla_map.get_waypoint(
            spawn_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            raise RuntimeError("Failed to find a valid waypoint near the specified location")

        spawn_point = waypoint.transform
        spawn_point.location.z += self.config.spawn_z_offset

        vehicles = []
        target_vehicle = None
        cut_out_lead_vehicle = None
        cut_out_cut_vehicle = None

        # 处理切出工况：生成两个前车
        if self.config.enable_cut_out_scenario:
            print("   🚗 切出工况：生成前车（lead + cut）...")
            result = self._spawn_cut_out_vehicles(world, carla_map, vehicle_bp)
            cut_out_lead_vehicle = result['lead_vehicle']
            cut_out_cut_vehicle = result['cut_vehicle']
            target_vehicle = cut_out_cut_vehicle  # 初始跟车目标
            vehicles.extend([cut_out_lead_vehicle, cut_out_cut_vehicle])
            print(f"   ✅ 切出工况车辆生成完成")
        else:
            # 普通工况或切入工况：生成单个前车
            target_vehicle = self._spawn_target_vehicle(world, carla_map, vehicle_bp, spawn_point)
            vehicles.append(target_vehicle)
            target_vehicle.set_autopilot(True)
            print(f"   ✅ 前车生成完成")

        # 生成自车
        ego_vehicle = self._spawn_ego_vehicle(world, carla_map, ego_vehicle_bp, spawn_location)
        print(f"   ✅ 自车生成完成")

        return {
            'ego_vehicle': ego_vehicle,
            'target_vehicle': target_vehicle,
            'other_vehicles': vehicles,
            'all_vehicles': vehicles + [ego_vehicle],
            'cut_out_lead_vehicle': cut_out_lead_vehicle,
            'cut_out_cut_vehicle': cut_out_cut_vehicle
        }

    def _determine_spawn_location(self) -> carla.Location:
        """确定生成位置（根据场景模式）"""
        if self.config.enable_cut_out_scenario and self.config.cut_out_front_spawn_location:
            return self.config.cut_out_front_spawn_location
        elif self.config.enable_cut_in_scenario and self.config.cut_in_target_spawn_location:
            return self.config.cut_in_target_spawn_location
        else:
            return self.config.spawn_location

    def _spawn_cut_out_vehicles(self, world, carla_map, vehicle_bp) -> dict:
        """生成切出工况的两个前车"""
        lead_wp = carla_map.get_waypoint(
            self.config.cut_out_front_spawn_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if lead_wp is None:
            raise RuntimeError("Cut-out spawn locations are not on drivable lanes")

        # Lead车辆
        lead_tf = lead_wp.transform
        lead_tf.location = self.config.cut_out_front_spawn_location
        lead_tf.location.z += self.config.spawn_z_offset

        # Cut车辆：沿lead_wp反向后移
        cut_prev = lead_wp.previous(self.config.cut_out_cut_back_distance_m)
        if not cut_prev:
            raise RuntimeError("Failed to find waypoint for cut-out cut vehicle")

        cut_tf = cut_prev[0].transform
        cut_tf.location.z += self.config.spawn_z_offset

        # 生成
        lead_vehicle = world.try_spawn_actor(vehicle_bp, lead_tf)
        cut_vehicle = world.try_spawn_actor(vehicle_bp, cut_tf)

        if lead_vehicle is None or cut_vehicle is None:
            raise RuntimeError("Failed to spawn cut-out scenario vehicles")

        lead_vehicle.set_autopilot(True)
        cut_vehicle.set_autopilot(True)

        return {
            'lead_vehicle': lead_vehicle,
            'cut_vehicle': cut_vehicle
        }

    def _spawn_target_vehicle(self, world, carla_map, vehicle_bp, spawn_point) -> carla.Vehicle:
        """生成目标车辆（前车）"""
        target_vehicle = None

        # 切入工况使用预设位置
        if self.config.enable_cut_in_scenario and self.config.cut_in_target_spawn_location:
            tgt_wp = carla_map.get_waypoint(
                self.config.cut_in_target_spawn_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if tgt_wp is not None:
                tgt_transform = tgt_wp.transform
                tgt_transform.location = carla.Location(
                    self.config.cut_in_target_spawn_location.x,
                    self.config.cut_in_target_spawn_location.y,
                    self.config.cut_in_target_spawn_location.z + self.config.spawn_z_offset
                )
                target_vehicle = world.try_spawn_actor(vehicle_bp, tgt_transform)
            else:
                print("   ⚠️ 切入工况预设前车坐标不在可行驶车道，回退到默认spawn点")

        # 使用默认spawn点
        if target_vehicle is None:
            target_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if target_vehicle is None:
            raise RuntimeError("Failed to spawn target vehicle at waypoint location")

        return target_vehicle

    def _spawn_ego_vehicle(self, world, carla_map, ego_vehicle_bp, base_location) -> carla.Vehicle:
        """生成自车"""
        # 确定基准位置
        if self.config.enable_cut_out_scenario and self.config.cut_out_front_spawn_location:
            ego_base_location = self.config.cut_out_front_spawn_location
        else:
            ego_base_location = base_location

        ego_base_wp = carla_map.get_waypoint(
            ego_base_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if ego_base_wp is None:
            raise RuntimeError("Failed to find ego base waypoint for spawn")

        # 沿车道后移
        ego_waypoints = ego_base_wp.previous(self.config.ego_spawn_distance)
        if not ego_waypoints:
            raise RuntimeError("Failed to find a waypoint for ego vehicle spawn")

        ego_spawn_point = ego_waypoints[0].transform
        ego_spawn_point.location.z += self.config.spawn_z_offset

        ego_vehicle = world.try_spawn_actor(ego_vehicle_bp, ego_spawn_point)
        if ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")

        ego_vehicle.set_autopilot(False)
        return ego_vehicle

    def _setup_traffic_manager(self, client, all_vehicles, target_vehicle) -> carla.TrafficManager:
        """配置交通管理器"""
        print(f"\n4️⃣ 配置交通管理器...")
        tm = client.get_trafficmanager(self.config.tm_port)
        tm.set_global_distance_to_leading_vehicle(self.config.tm_global_distance)
        tm.set_synchronous_mode(self.config.synchronous_mode)

        # 配置前车速度
        self._configure_vehicle_speed(tm, all_vehicles, target_vehicle)

        print(f"   ✅ 交通管理器配置完成")
        return tm

    def _configure_vehicle_speed(self, tm, all_vehicles, target_vehicle):
        """配置车辆速度"""
        if self.config.use_constant_velocity:
            print(f"   速度模式: Traffic Manager")
            for vehicle in all_vehicles:
                if vehicle == target_vehicle:  # 只配置前车，不包括自车
                    vehicle.set_autopilot(True, tm.get_port())

                    # 切入/切出场景禁用自动换道
                    allow_lane_change = not (
                        self.config.enable_cut_in_scenario or
                        self.config.enable_cut_out_scenario
                    )
                    tm.auto_lane_change(vehicle, allow_lane_change)
                    tm.ignore_lights_percentage(vehicle, 100.0)

                    # 设置目标速度
                    percentage_diff = (
                        (self.config.assumed_road_speed_limit_kmh - self.config.target_speed_kmh) /
                        self.config.assumed_road_speed_limit_kmh
                    ) * 100.0
                    tm.vehicle_percentage_speed_difference(vehicle, percentage_diff)
                    tm.distance_to_leading_vehicle(vehicle, self.config.tm_target_vehicle_distance)

            print(f"   假设道路限速: {self.config.assumed_road_speed_limit_kmh:.1f} km/h")
            print(f"   目标速度: {self.config.target_speed_kmh:.1f} km/h")

    def _init_ramp_controller(self) -> RampSpeedController:
        """初始化斜坡速度控制器"""
        ramp_params = self.config.get_ramp_controller_params()
        return RampSpeedController(
            start_speed_kmh=ramp_params['start_speed_kmh'],
            target_speed_kmh=ramp_params['target_speed_kmh'],
            duration_s=ramp_params['duration_s']
        )

    def _init_scenario_managers(self, world, tm, blueprint_library, vehicles_data):
        """初始化场景管理器"""
        cut_in_manager = None
        cut_out_manager = None

        if self.config.enable_cut_in_scenario:
            print(f"\n5️⃣ 初始化切入场景管理器...")
            cut_in_manager = CutInScenarioManager(
                world=world,
                traffic_manager=tm,
                ego_vehicle=vehicles_data['ego_vehicle'],
                lead_vehicle=vehicles_data['target_vehicle'],
                blueprint_library=blueprint_library,
                enable_from_left=self.config.cut_in_from_left,
                trigger_time_s=self.config.cut_in_trigger_time_s,
                lateral_threshold_m=self.config.cut_in_lateral_threshold_m,
                vehicle_blueprint=self.config.cut_in_vehicle_blueprint,
                side_spawn_location=self.config.cut_in_side_spawn_location,
                side_back_offset=self.config.cut_in_side_spawn_back_offset_m,
                side_spawn_z_lift=self.config.cut_in_side_spawn_z_lift_m,
                lane_entry_margin=self.config.cut_in_lane_entry_margin_m
            )
            if cut_in_manager and cut_in_manager.cut_in_vehicle:
                vehicles_data['other_vehicles'].append(cut_in_manager.cut_in_vehicle)
                print("   ✅ 切入场景初始化完成")

        elif self.config.enable_cut_out_scenario:
            print(f"\n5️⃣ 初始化切出场景管理器...")
            cut_out_manager = CutOutScenarioManager(
                world=world,
                traffic_manager=tm,
                ego_vehicle=vehicles_data['ego_vehicle'],
                cut_vehicle=vehicles_data['cut_out_cut_vehicle'],
                lead_vehicle=vehicles_data['cut_out_lead_vehicle'],
                blueprint_library=blueprint_library,
                trigger_time_s=self.config.cut_out_trigger_time_s,
                change_to_right=self.config.cut_out_change_to_right
            )
            print("   ✅ 切出场景初始化完成")

        return cut_in_manager, cut_out_manager

    def _setup_traffic_lights(self, world):
        """设置交通灯为绿灯并冻结"""
        print(f"\n6️⃣ 配置交通灯...")
        traffic_lights = world.get_actors().filter('traffic.traffic_light')
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
        print(f"   ✅ {len(traffic_lights)} 个交通灯已设置为绿灯")
