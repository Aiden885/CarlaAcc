"""
控制循环管理器
负责执行主控制循环的核心逻辑
从acc_updated.py的generate_target方法中提取
"""
import time
from typing import Tuple

import numpy as np
import carla

from acc_config import ACCConfig
from acc_control_facade import ACCControlFacade
from carla_system_initializer import CarlaSystemResources
from enhanced_lateral_controller import EnhancedLateralController
from system_state import (
    SystemState, VehicleState, PerceptionData, ControlOutput, StepResult
)
from two_mode_controller import (
    calculate_two_mode_desired_distance,
    enhanced_two_mode_control,
    set_two_mode_parameters
)
from vehicle_utils import VehicleUtils


class ControlLoopManager:
    """
    控制循环管理器
    职责：
    1. 执行单个控制周期
    2. 协调感知、决策、控制模块
    3. 管理场景切换
    4. 更新车辆控制
    """

    def __init__(
        self,
        resources: CarlaSystemResources,
        acc_controller: ACCControlFacade,
        config: ACCConfig
    ):
        self.resources = resources
        self.acc_controller = acc_controller
        self.config = config

        # 系统状态
        self.system_state = SystemState()

        # 横向控制器（从资源中提取配置）
        lateral_params = config.get_lateral_controller_params()
        self.lateral_controller = EnhancedLateralController(
            kp=lateral_params['kp'],
            ki=lateral_params['ki'],
            kd=lateral_params['kd']
        )

        # 可选预瞄参数
        if hasattr(config, 'lateral_lookahead_params'):
            lookahead = config.lateral_lookahead_params
            self.lateral_controller.set_weights(
                weight_current=lookahead['weight_current'],
                weight_lookahead=lookahead['weight_lookahead']
            )
            self.lateral_controller.set_lookahead_params(
                base=lookahead['base_distance'],
                gain=lookahead['gain']
            )

        # ACC参数（从配置加载）
        self.acc_params = config.get_acc_params()

        # 初始化两模式控制参数
        set_two_mode_parameters(
            V_threshold_kmh=self.acc_params['V_target_kmh'],
            G2_s=self.acc_params['G2_s'],
            target_speed_kmh=self.acc_params['V_target_kmh']
        )

        # 前车速度控制状态
        self.last_speed_limit = None

    def run_single_step(self, manual_input_state) -> StepResult:
        """
        执行单个控制周期

        Args:
            manual_input_state: 手动输入状态

        Returns:
            StepResult: 包含控制输出和系统状态的结果
        """
        # 更新仿真时间
        self.system_state.sim_elapsed_s += self.config.fixed_delta_seconds
        self.system_state.frame_count += 1

        # 1. 更新场景（切入/切出）
        self._update_scenarios(self.system_state.sim_elapsed_s)

        # 2. 更新前车速度控制
        self._update_target_vehicle_speed()

        # 3. 感知环境
        perception_data = self._perceive_environment()
        self.system_state.perception = perception_data

        # 4. 更新车辆状态
        self._update_vehicle_states()

        # 5. ACC决策和控制
        unified_output, simulink_duration_ms = self._acc_decision_and_control(
            perception_data,
            manual_input_state
        )

        # 6. 计算最终控制输出
        control_output = self._compute_vehicle_control(
            unified_output,
            perception_data,
            manual_input_state
        )

        # 7. 应用控制
        self._apply_vehicle_control(control_output, manual_input_state)

        # 8. 准备环境数据
        env_data = self._prepare_env_data(perception_data)

        # 9. 返回结果
        return StepResult(
            control_output=control_output,
            system_state=self.system_state,
            unified_output=unified_output,
            env_data=env_data,
            simulink_duration_ms=simulink_duration_ms
        )

    def _update_scenarios(self, elapsed_time_s: float):
        """更新场景（切入/切出）"""
        # 切入场景
        if self.resources.cut_in_manager:
            self.resources.cut_in_manager.maybe_trigger_cut_in(elapsed_time_s)
            if (self.resources.cut_in_manager.cut_in_vehicle and
                self.resources.target_vehicle != self.resources.cut_in_manager.cut_in_vehicle and
                self.resources.cut_in_manager.should_switch_to_cut_in()):
                print("🚗 切入车辆已进入本车道，切换跟车目标")
                self._set_target_vehicle(self.resources.cut_in_manager.cut_in_vehicle)

        # 切出场景
        if self.resources.cut_out_manager:
            self.resources.cut_out_manager.maybe_trigger_cut_out(elapsed_time_s)
            if self.resources.cut_out_manager._lane_change_completed:
                if (self.resources.cut_out_lead_vehicle and
                    self.resources.target_vehicle != self.resources.cut_out_lead_vehicle):
                    print("🚗 切出车已离开车道，切换跟车目标到最前车")
                    self._set_target_vehicle(self.resources.cut_out_lead_vehicle)
                # 重置状态
                self.resources.cut_out_manager._lane_change_triggered = False
                self.resources.cut_out_manager._lane_change_completed = False
                self.resources.cut_out_manager._scenario_completed = True

    def _set_target_vehicle(self, vehicle):
        """统一更新前车引用"""
        self.resources.target_vehicle = vehicle
        self.resources.carla_perception.target_vehicle = vehicle
        self.resources.lane_change_controller.vehicle = vehicle

    def _update_target_vehicle_speed(self):
        """更新前车速度控制"""
        # 换道控制器更新
        lane_change_completed = self.resources.lane_change_controller.update()
        if lane_change_completed:
            # 恢复速度控制
            if self.resources.ramp_controller.is_ramp_active():
                restore_speed = self.resources.ramp_controller.get_target_speed()
            else:
                restore_speed = self.config.target_speed_kmh

            percentage_diff = (
                (self.config.assumed_road_speed_limit_kmh - restore_speed) /
                self.config.assumed_road_speed_limit_kmh
            ) * 100.0
            self.resources.traffic_manager.vehicle_percentage_speed_difference(
                self.resources.target_vehicle,
                percentage_diff
            )

        # 检查是否正在换道
        if self.resources.lane_change_controller.is_lane_changing():
            return

        # 斜坡速度控制
        if self.resources.ramp_controller.is_ramp_active():
            ramp_target_speed = self.resources.ramp_controller.get_target_speed()
            if ramp_target_speed is not None:
                percentage_diff = (
                    (self.config.assumed_road_speed_limit_kmh - ramp_target_speed) /
                    self.config.assumed_road_speed_limit_kmh
                ) * 100.0
                self.resources.traffic_manager.vehicle_percentage_speed_difference(
                    self.resources.target_vehicle,
                    percentage_diff
                )

        # Traffic Manager模式：动态更新速度
        elif not self.config.use_constant_velocity:
            if self.resources.target_vehicle:
                current_speed_limit = self.resources.target_vehicle.get_speed_limit()

                # 防御性处理
                if current_speed_limit is None or current_speed_limit <= 0.0 or not np.isfinite(current_speed_limit):
                    current_speed_limit = 30.0

                # 检查限速变化
                if self.last_speed_limit != current_speed_limit:
                    percentage_diff = (
                        (current_speed_limit - self.config.target_speed_kmh) /
                        current_speed_limit
                    ) * 100.0
                    self.resources.traffic_manager.vehicle_percentage_speed_difference(
                        self.resources.target_vehicle,
                        percentage_diff
                    )
                    self.last_speed_limit = current_speed_limit

    def _perceive_environment(self) -> PerceptionData:
        """感知环境"""
        # 获取距离和车道偏移
        vehicle_distance = self.resources.carla_perception.get_vehicle_distance()
        lane_offset = self.resources.carla_perception.get_lane_offset()

        # 判断是否有目标
        has_target = vehicle_distance < self.config.detection_range

        # 获取速度
        ego_speed_kmh = VehicleUtils.get_vehicle_speed(self.resources.ego_vehicle)
        target_speed_kmh = VehicleUtils.get_vehicle_speed(self.resources.target_vehicle) if self.resources.target_vehicle else 0.0

        ego_speed_ms = ego_speed_kmh / 3.6
        target_speed_ms = self.acc_params['V_target_kmh'] / 3.6

        # 计算两模式控制信息
        if has_target:
            enhanced_output = enhanced_two_mode_control(ego_speed_ms, vehicle_distance, target_speed_ms)
        else:
            enhanced_output = enhanced_two_mode_control(ego_speed_ms, None, target_speed_ms)

        # 控制模式名称
        control_mode_names = {1: "TIME模式", 2: "SPEED模式"}
        control_mode_name = control_mode_names.get(enhanced_output['control_mode_flag'], "Unknown")

        # 计算期望距离和误差
        desired_distance = enhanced_output.get('desired_distance', 0.0) if has_target else 0.0
        distance_error = vehicle_distance - desired_distance if has_target else 0.0

        return PerceptionData(
            vehicle_distance=vehicle_distance,
            lane_offset=lane_offset,
            has_target=has_target,
            target_speed_kmh=target_speed_kmh,
            desired_distance=desired_distance,
            distance_error=distance_error,
            control_error=enhanced_output['control_error'],
            control_mode_flag=enhanced_output['control_mode_flag'],
            control_mode_name=control_mode_name
        )

    def _update_vehicle_states(self):
        """更新车辆状态"""
        # 自车状态
        self.system_state.ego.speed_kmh = VehicleUtils.get_vehicle_speed(self.resources.ego_vehicle)
        self.system_state.ego.speed_ms = self.system_state.ego.speed_kmh / 3.6

        # 前车状态
        if self.resources.target_vehicle:
            self.system_state.target.speed_kmh = VehicleUtils.get_vehicle_speed(self.resources.target_vehicle)
            self.system_state.target.speed_ms = self.system_state.target.speed_kmh / 3.6

        # ACC状态
        self.system_state.acc.target_speed_kmh = self.acc_params['V_target_kmh']
        self.system_state.acc.time_gap_s = self.acc_params['G2_s']
        self.system_state.acc.min_speed_kmh = self.acc_params['V_min_kmh']

    def _acc_decision_and_control(self, perception_data: PerceptionData, manual_input_state) -> tuple:
        """ACC决策和控制"""
        # 准备输入数据
        command_type = 0  # 默认NONE
        command_description = None

        # 检查待处理的键盘指令
        if self.system_state.pending_command and self.system_state.pending_command.is_valid():
            command_type = self.system_state.pending_command.code
            command_description = self.system_state.pending_command.description
            self.system_state.reset_pending_command()
        else:
            # 使用W/S键状态
            if manual_input_state.w_pressed and self.system_state.acc.system_enabled:
                command_type = 5
            elif manual_input_state.s_pressed and self.system_state.acc.system_enabled:
                command_type = 6

        # 数据清洗
        def sanitize(value, default=0.0):
            if value is None or not np.isfinite(value):
                return default
            return value

        # 构造输入
        unified_input = {
            'ego_speed_kmh': sanitize(self.system_state.ego.speed_kmh),
            'ego_speed_ms': sanitize(self.system_state.ego.speed_ms),
            'command_type': command_type,
            'command_active': bool(command_type),
            'manual_throttle_active': manual_input_state.has_throttle_input(),
            'control_error': sanitize(perception_data.control_error),
            'control_mode_flag': perception_data.control_mode_flag,
            'V_target_kmh': sanitize(self.acc_params['V_target_kmh'], 50.0),
            'V_min_kmh': sanitize(self.acc_params['V_min_kmh'], 30.0),
            'G2_s': sanitize(self.acc_params['G2_s'], 2.0),
            'timestamp': time.time(),
        }

        # 调用Simulink
        t0 = time.time()
        unified_output = self.acc_controller.process_decision_and_control(unified_input)
        simulink_duration_ms = (time.time() - t0) * 1000

        # 同步参数变化
        self._sync_parameters_from_simulink(unified_output)

        # 保存Simulink输出中的指令描述（用于显示）
        unified_output['command_description'] = command_description

        return unified_output, simulink_duration_ms

    def _sync_parameters_from_simulink(self, unified_output: dict):
        """同步Simulink输出的参数"""
        params_changed = False

        if 'updated_V_target_kmh' in unified_output:
            new_V_target = unified_output['updated_V_target_kmh']
            if abs(new_V_target - self.acc_params['V_target_kmh']) > 0.1:
                self.acc_params['V_target_kmh'] = new_V_target
                params_changed = True

        if 'updated_G2_s' in unified_output:
            new_G2 = unified_output['updated_G2_s']
            if abs(new_G2 - self.acc_params['G2_s']) > 0.01:
                self.acc_params['G2_s'] = new_G2
                params_changed = True

        # 同步到两模式控制器
        if params_changed:
            set_two_mode_parameters(
                V_threshold_kmh=self.acc_params['V_target_kmh'],
                G2_s=self.acc_params['G2_s'],
                target_speed_kmh=self.acc_params['V_target_kmh']
            )

    def _compute_vehicle_control(
        self,
        unified_output: dict,
        perception_data: PerceptionData,
        manual_input_state
    ) -> ControlOutput:
        """计算车辆控制输出"""
        control_output = ControlOutput()

        # 检查ACC是否应该控制
        acc_should_control = (
            self.system_state.acc.system_enabled and
            unified_output.get('control_enabled', False)
        )

        if not acc_should_control:
            # ACC不控制，使用手动输入
            control_output.throttle = manual_input_state.throttle
            control_output.brake = manual_input_state.brake
            control_output.steer = manual_input_state.steer
            control_output.mode = "MANUAL"
            return control_output

        # ACC控制模式
        # 横向控制
        steer = self._compute_lateral_control(perception_data, manual_input_state)

        # 纵向控制
        throttle, brake = self._compute_longitudinal_control(unified_output, perception_data)

        # 扭矩仲裁
        torque_arbitration = unified_output.get('torque_arbitration_active', False)
        if manual_input_state.has_throttle_input():
            if torque_arbitration:
                throttle = max(throttle, manual_input_state.throttle)
                brake = 0.0
            else:
                throttle = manual_input_state.throttle
                brake = 0.0

        if manual_input_state.has_brake_input():
            throttle = 0.0
            brake = manual_input_state.brake

        # 获取控制模式
        sppvt_stage = unified_output.get('sppvt_stage_output', 0)
        if sppvt_stage >= 1:
            mode = f"UNIFIED_SPPVT_Stage{int(sppvt_stage)}"
        else:
            mode = "UNIFIED_SPPVT_Unknown"

        control_output.throttle = throttle
        control_output.brake = brake
        control_output.steer = steer
        control_output.mode = mode
        control_output.torque_arbitration = torque_arbitration
        control_output.sppvt_throttle = throttle
        control_output.driver_throttle = manual_input_state.throttle

        return control_output

    def _compute_lateral_control(self, perception_data: PerceptionData, manual_input_state) -> float:
        """计算横向控制"""
        # 手动转向优先
        if abs(manual_input_state.steer) > 1e-3:
            return manual_input_state.steer

        # 使用增强横向控制器
        current_offset = -perception_data.lane_offset
        lookahead_distance = self.lateral_controller.calculate_lookahead_distance(self.system_state.ego.speed_ms)
        lookahead_offset = -self.resources.carla_perception.get_lookahead_offset(lookahead_distance)

        steer_output = self.lateral_controller.update(
            current_offset=current_offset,
            lookahead_offset=lookahead_offset,
            speed_ms=self.system_state.ego.speed_ms,
            dt=0.05
        )

        return steer_output

    def _compute_longitudinal_control(self, unified_output: dict, perception_data: PerceptionData) -> Tuple[float, float]:
        """计算纵向控制（油门/刹车）"""
        # SPPVT输出：扭矩（无量纲），需要缩放到实际发动机扭矩
        control_output = unified_output.get('sppvt_control_output', 0.0)
        control_mode_flag = perception_data.control_mode_flag

        # 符号转换（根据控制模式调整符号）
        if control_mode_flag == 1:  # TIME模式
            sppvt_torque_demand = -control_output
        elif control_mode_flag == 2:  # SPEED模式
            sppvt_torque_demand = control_output
        else:
            sppvt_torque_demand = control_output

        # SPPVT输出缩放（调试用的KP增益）
        # 注意：sppvt_accel_scale/sppvt_decel_scale 是缩放增益，不是单位转换
        if sppvt_torque_demand >= 0:
            # 加速：缩放到发动机扭矩
            sppvt_engine_torque = sppvt_torque_demand * self.config.sppvt_accel_scale
        else:
            # 减速：缩放到发动机扭矩
            sppvt_engine_torque = sppvt_torque_demand * self.config.sppvt_decel_scale

        # 扭矩转油门/刹车
        if self.config.use_torque_converter:
            throttle, brake = self.resources.torque_converter.engine_torque_to_throttle(
                sppvt_engine_torque,
                self.system_state.ego.speed_kmh
            )
            # 防止同时有油门和刹车
            if sppvt_torque_demand >= 0:
                brake = 0.0
        else:
            # 简化映射
            if sppvt_engine_torque > 0:
                throttle = min(sppvt_engine_torque / 749.0, 1.0)
                brake = 0.0
            else:
                throttle = 0.0
                brake = min(abs(sppvt_engine_torque) / 100.0, 1.0)

        return throttle, brake

    def _apply_vehicle_control(self, control_output: ControlOutput, manual_input_state):
        """应用车辆控制"""
        control = carla.VehicleControl()
        control.throttle = control_output.throttle
        control.brake = max(0.0, control_output.brake) if control_output.brake >= 0.01 else 0.0
        control.steer = control_output.steer
        control.manual_gear_shift = False
        control.gear = 1

        self.resources.ego_vehicle.apply_control(control)

        # 更新系统状态
        self.system_state.ego.throttle = control.throttle
        self.system_state.ego.brake = control.brake
        self.system_state.ego.steer = control.steer

    def _prepare_env_data(self, perception_data: PerceptionData) -> dict:
        """准备环境数据（用于显示和记录）"""
        return {
            'ego_speed_kmh': self.system_state.ego.speed_kmh,
            'ego_speed_ms': self.system_state.ego.speed_ms,
            'target_speed_kmh': self.system_state.target.speed_kmh,
            'target_speed_ms': self.system_state.target.speed_ms,
            'vehicle_distance': perception_data.vehicle_distance,
            'desired_distance': perception_data.desired_distance,
            'distance_error': perception_data.distance_error,
            'has_target': perception_data.has_target,
            'lane_offset': perception_data.lane_offset,
            'control_error': perception_data.control_error,
            'control_mode_flag': perception_data.control_mode_flag,
            'control_mode_name': perception_data.control_mode_name
        }

    def get_acc_params(self) -> dict:
        """获取当前ACC参数"""
        return self.acc_params.copy()

    def set_acc_system_enabled(self, enabled: bool):
        """设置ACC系统开关"""
        self.system_state.acc.system_enabled = enabled
        if not enabled:
            self.acc_controller.reset()

    def trigger_ramp_speed(self):
        """触发前车斜坡速度"""
        self.resources.ramp_controller.trigger()

    def trigger_lane_change_left(self):
        """触发前车向左换道"""
        if self.resources.ramp_controller.is_ramp_active():
            current_speed = self.resources.ramp_controller.get_target_speed()
        else:
            current_speed = self.config.target_speed_kmh
        self.resources.lane_change_controller.change_lane_left(current_speed)

    def trigger_lane_change_right(self):
        """触发前车向右换道"""
        if self.resources.ramp_controller.is_ramp_active():
            current_speed = self.resources.ramp_controller.get_target_speed()
        else:
            current_speed = self.config.target_speed_kmh
        self.resources.lane_change_controller.change_lane_right(current_speed)
