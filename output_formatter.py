"""
Output formatting utilities for ACC system
输出格式化模块 - 提供Simulink I/O和系统信息的格式化输出
"""
from acc_controller import ACCState


class OutputFormatter:
    """输出格式化工具类"""

    @staticmethod
    def print_simulink_io(frame_num, current_time, unified_input, unified_output, duration_ms,
                         manual_throttle_input, manual_brake_input, w_key_pressed, s_key_pressed,
                         acc_system_enabled, command_description=None, final_control=None, env_data=None):
        """
        格式化输出Simulink输入输出信息

        Args:
            frame_num: 帧编号
            current_time: 当前时间(秒)
            unified_input: Simulink输入字典
            unified_output: Simulink输出字典
            duration_ms: Simulink调用耗时(毫秒)
            manual_throttle_input: 手动油门输入值
            manual_brake_input: 手动刹车输入值
            w_key_pressed: W键是否按下
            s_key_pressed: S键是否按下
            acc_system_enabled: ACC系统是否开启
            command_description: 键盘指令描述(可选)
            final_control: 最终控制输出(油门/刹车/转向, 可选)
            env_data: 环境感知数据字典(可选)
        """
        # 指令名称映射
        cmd_names = {
            0: "NONE",
            1: "I0(降速)",
            2: "I1(增速)",
            3: "I2(降距)",
            4: "I3(增距)",
            5: "I4(油门)",
            6: "I5(刹车)",
            7: "I6(取消)"
        }

        # 控制模式名称
        mode_names = {
            1: "TIME模式",
            2: "SPEED模式"
        }

        print(f"\n{'='*80}")
        print(f"帧#{frame_num} | 时间:{current_time:.2f}s | Simulink:{duration_ms:.1f}ms")
        print(f"{'='*80}")

        # [环境感知] - 表格式
        if env_data:
            print(f"[环境感知]")
            has_target_str = "✓" if env_data.get('has_target', False) else "✗"
            target_speed_str = f"{env_data.get('target_speed_kmh', 0.0):.1f}km/h" if env_data.get('has_target', False) else "N/A"

            # 距离误差符号
            distance_error = env_data.get('distance_error', 0.0)
            distance_error_str = f"{distance_error:+.2f}m" if env_data.get('has_target', False) else "N/A"

            # 车道偏移符号
            lane_offset = env_data.get('lane_offset', 0.0)
            lane_offset_str = f"{lane_offset:+.2f}m"

            # 控制模式
            control_mode = env_data.get('control_mode_name', 'Unknown')

            # SPPVT扭矩输出（从unified_output读取）
            sppvt_torque_output = unified_output.get('sppvt_control_output', 0.0)

            # 根据控制模式确定 control_error 的单位
            control_mode_flag = env_data.get('control_mode_flag', 1)
            control_error_unit = "s" if control_mode_flag == 1 else "m/s"

            print(f"  自车     前车      实际距离  期望距离  距离误差  模式      control_error  SPPVT扭矩输出")
            print(f"  {env_data.get('ego_speed_kmh', 0.0):.1f}km/h {target_speed_str:8s}  "
                  f"{env_data.get('vehicle_distance', 0.0):.2f}m   {env_data.get('desired_distance', 0.0):.2f}m   "
                  f"{distance_error_str:8s} {control_mode:8s}  "
                  f"{env_data.get('control_error', 0.0):+.3f}{control_error_unit:4s}  "
                  f"{sppvt_torque_output:+.3f}")
            print(f"  ")

            # Two-Mode计算说明
            if env_data.get('has_target', False):
                if env_data.get('control_mode_flag', 1) == 1:
                    print(f"[Two-Mode计算] control_error = 距离误差({distance_error:.2f}m) ÷ 自车速度({env_data.get('ego_speed_ms', 0.0):.2f}m/s) = {env_data.get('control_error', 0.0):.3f}s")
                else:
                    print(f"[Two-Mode计算] control_error = 前车速度({env_data.get('target_speed_ms', 0.0):.2f}m/s) - 自车速度({env_data.get('ego_speed_ms', 0.0):.2f}m/s) = {env_data.get('control_error', 0.0):.3f}m/s")
            else:
                print(f"[Two-Mode计算] 无前车 | control_error = 目标速度({env_data.get('target_speed_ms', 0.0):.2f}m/s) - 自车速度({env_data.get('ego_speed_ms', 0.0):.2f}m/s) = {env_data.get('control_error', 0.0):.3f}m/s")

            print(f"[ACC参数] V_target:{unified_input['V_target_kmh']:.1f}km/h | G2:{unified_input['G2_s']:.1f}s | V_min:{unified_input['V_min_kmh']:.1f}km/h | 手动: 油门{manual_throttle_input:.2f} 刹车{manual_brake_input:.2f}")
            print(f"")

        # [键盘] - 紧凑单行
        if command_description:
            cmd_str = command_description
        else:
            cmd_type = unified_input['command_type']
            cmd_str = cmd_names.get(cmd_type, f'Unknown({cmd_type})')

        acc_status = "开启" if acc_system_enabled else "关闭"
        w_status = "按下" if w_key_pressed else "松开"
        s_status = "按下" if s_key_pressed else "松开"
        print(f"[键盘] {cmd_str} | ACC:{acc_status} | W:{w_status} S:{s_status}")

        # [Simulink→] - 紧凑单行
        # 根据控制模式确定 control_error 的单位
        input_control_mode_flag = unified_input.get('control_mode_flag', 1)
        input_error_unit = "s" if input_control_mode_flag == 1 else "m/s"
        print(f"[Simulink→] 速度{unified_input['ego_speed_kmh']:.1f}km/h cmd:{unified_input['command_type']} "
              f"err:{unified_input['control_error']:.3f}{input_error_unit} mode:{unified_input['control_mode_flag']} "
              f"V_target:{unified_input['V_target_kmh']:.1f} G2:{unified_input['G2_s']:.1f}")

        # [Simulink←] - 紧凑单行，使用S0/S1/S2/S3格式显示状态
        old_V = unified_input['V_target_kmh']
        new_V = unified_output.get('updated_V_target_kmh', old_V)
        old_G2 = unified_input['G2_s']
        new_G2 = unified_output.get('updated_G2_s', old_G2)

        # 状态映射：S0=在控, S1=有史待命, S2=无史待命, S3=低速
        state_num = unified_output.get('current_state', 2)
        state_names = {0: 'S0在控', 1: 'S1有史待命', 2: 'S2无史待命', 3: 'S3低速'}
        state_str = state_names.get(state_num, f'S{state_num}未知')

        # 决策映射
        decision_num = unified_output.get('current_decision', 0)
        decision_names = {
            1: 'R1系统待命', 2: 'R2保持', 3: 'R3跟随', 4: 'R4迫近',
            5: 'R5降速', 6: 'R6加速', 7: 'R7扭矩仲裁', 8: 'R8取消'
        }
        decision_str = decision_names.get(decision_num, f'R{decision_num}')

        control_enabled_str = '✓' if unified_output.get('control_enabled', False) else '✗'

        upgrade_flag = unified_output.get('sppvt_status_output', 0)
        print(f"[Simulink←] 状态:{state_str} 决策:{decision_str} 控制:{control_enabled_str} "
              f"accel:{unified_output.get('target_accel', 0.0):.3f} "
              f"stage:{unified_output.get('sppvt_stage_output', 0)} "
              f"升级:{upgrade_flag:.0f} "
              f"仲裁:{unified_output.get('torque_arbitration_active', False)}")

        # 分类显示输出字段，便于逐帧检查
        # ===== Simulink集成模型输出（10个UDP返回值）=====

        print("\n[Simulink集成模型输出 - Decision子系统]")
        simulink_decision_fields = [
            ('next_state', 'current_state'),  # (输出字段名, 显示名)
            ('current_decision', 'current_decision'),
            ('control_enabled', 'control_enabled'),
            ('next_has_history', 'next_has_history'),
            ('next_last_active_decision', 'next_last_active_decision'),
        ]
        for field_name, display_name in simulink_decision_fields:
            present = field_name in unified_output
            value = unified_output.get(field_name, None)
            if not present:
                print(f"  {display_name:28s}: <缺失> [MISSING]")
                continue
            if value is None:
                print(f"  {display_name:28s}: None")
                continue
            if isinstance(value, float):
                print(f"  {display_name:28s}: {value:.6f}")
            else:
                print(f"  {display_name:28s}: {value}")

        print("\n[Simulink集成模型输出 - SPPVT子系统]")
        simulink_sppvt_fields = [
            'sppvt_control_output',       # 控制输出（SPPVT计算的目标加速度）
            'sppvt_velocity_output',      # 误差导数（速度）
            'sppvt_acceleration_output',  # 误差二阶导数（加速度）
            'sppvt_jerk_output',          # 加加速度（Jerk）
            'sppvt_status_output',        # should_upgrade 标志
        ]
        for name in simulink_sppvt_fields:
            present = name in unified_output
            value = unified_output.get(name, None)
            if not present:
                print(f"  {name:28s}: <缺失> [MISSING]")
                continue
            if value is None:
                print(f"  {name:28s}: None")
                continue
            if isinstance(value, float):
                print(f"  {name:28s}: {value:.6f}")
            else:
                print(f"  {name:28s}: {value}")

        # ===== Python端输出（非Simulink计算）=====

        print("\n[Python端输出 - 参数调整]")
        python_param_fields = [
            'updated_V_target_kmh',  # 键盘调整后的目标速度
            'updated_G2_s',          # 键盘调整后的时距
        ]
        for name in python_param_fields:
            present = name in unified_output
            value = unified_output.get(name, None)
            if not present:
                print(f"  {name:28s}: <缺失> [MISSING]")
                continue
            if value is None:
                print(f"  {name:28s}: None")
                continue
            if isinstance(value, float):
                print(f"  {name:28s}: {value:.6f}")
            else:
                print(f"  {name:28s}: {value}")

        print("\n[Python端输出 - 决策辅助]")
        python_decision_aux_fields = [
            'torque_arbitration_active',  # 扭矩仲裁标志（从decision判断）
            'debug_message',              # 调试消息
        ]
        for name in python_decision_aux_fields:
            present = name in unified_output
            value = unified_output.get(name, None)
            if not present:
                print(f"  {name:28s}: <缺失> [MISSING]")
                continue
            if value is None:
                print(f"  {name:28s}: None")
                continue
            print(f"  {name:28s}: {value}")

        print("\n[Python端输出 - SPPVT状态维护]")
        python_sppvt_state_fields = [
            'sppvt_stage_output',     # 当前阶段（简化为固定值1）
            'sppvt_status_output',    # 状态指示
            'new_stage_offset',       # 级差值（Python维护）
            'new_stage',              # 新阶段（Python维护）
            'new_error_sign',         # 误差符号
            'new_upgrade_count',      # 升级计数
            'new_control_error',      # 控制误差（复制自输入）
            'new_error_derivative',   # 误差导数（复制自Simulink输出）
            'new_error_second_derivative',  # 误差二阶导数（复制自Simulink输出）
        ]
        for name in python_sppvt_state_fields:
            present = name in unified_output
            value = unified_output.get(name, None)
            if not present:
                print(f"  {name:28s}: <缺失> [MISSING]")
                continue
            if value is None:
                print(f"  {name:28s}: None")
                continue
            if isinstance(value, float):
                print(f"  {name:28s}: {value:.6f}")
            else:
                print(f"  {name:28s}: {value}")

        # # SPPVT内部状态显示（无条件输出）
        # adapter_states = unified_output.get('new_adapter_states', None)
        # if adapter_states is not None:
        #     if len(adapter_states) >= 3:
        #         print(f"            SPPVT状态: prev_error={adapter_states[0]:.3f} "
        #               f"prev_velocity={adapter_states[1]:.3f} "
        #               f"prev_accel={adapter_states[2]:.3f}")
        #     elif len(adapter_states) == 2:
        #         print(f"            SPPVT状态: prev_error={adapter_states[0]:.3f} "
        #               f"prev_velocity={adapter_states[1]:.3f} "
        #               f"prev_accel=❌缺失(Simulink只输出了2个元素)")
        #     else:
        #         print(f"            SPPVT状态: [数组长度错误] 期望3个元素，实际{len(adapter_states)}个: {adapter_states}")
        # else:
        #     print(f"            SPPVT状态: [数据不可用] adapter_states=None")

        # 参数变化检查
        if abs(new_V - old_V) > 0.1 or abs(new_G2 - old_G2) > 0.1:
            print(f"            参数变化: V_target:{old_V:.1f}→{new_V:.1f} G2:{old_G2:.1f}→{new_G2:.1f}")

        # [控制执行] - 紧凑单行
        if final_control:
            mode_str = final_control.get('mode', 'Unknown')
            print(f"[控制执行] {mode_str} | 油门:{final_control.get('throttle', 0.0):.2f} "
                  f"刹车:{final_control.get('brake', 0.0):.2f} 转向:{final_control.get('steer', 0.0):.3f}")

            # 扭矩仲裁信息
            if final_control.get('torque_arbitration'):
                print(f"            ⚖️ 扭矩仲裁: SPPVT={final_control.get('sppvt_throttle', 0.0):.2f} + Driver={final_control.get('driver_throttle', 0.0):.2f}")

        print(f"{'='*80}\n")

    @staticmethod
    def format_system_info(ego_vehicle, target_vehicle, acc_system_enabled, acc_decision,
                          acc_params, throttle, brake, steer,
                          get_vehicle_speed_func, get_vehicle_distance_func):
        """
        获取系统状态信息，用于显示

        Args:
            ego_vehicle: 自车对象
            target_vehicle: 目标车辆对象
            acc_system_enabled: ACC系统是否开启
            acc_decision: ACC决策对象
            acc_params: ACC参数字典
            throttle: 当前油门值
            brake: 当前刹车值
            steer: 当前转向值
            get_vehicle_speed_func: 获取车速的函数
            get_vehicle_distance_func: 获取车距的函数

        Returns:
            dict: 系统信息字典
        """
        ego_speed = get_vehicle_speed_func(ego_vehicle)
        target_distance = get_vehicle_distance_func(ego_vehicle, target_vehicle)
        has_target = target_distance < 50.0

        from vehicle_utils import VehicleUtils

        lane_offset = 0.0
        try:
            lane_offset = VehicleUtils.get_lane_offset(ego_vehicle, ego_vehicle.get_world())
        except Exception:
            lane_offset = 0.0

        # 兼容性处理：acc_decision.current_state 可能是整数或枚举
        current_state = acc_decision.current_state
        if isinstance(current_state, ACCState):
            is_in_control = (current_state == ACCState.ACTIVE_CONTROL)
        else:
            # 如果是整数，比较值
            is_in_control = (current_state == ACCState.ACTIVE_CONTROL.value)

        return {
            'ego_speed': ego_speed,
            'target_distance': target_distance,
            'has_target': has_target,
            'acc_system_enabled': acc_system_enabled,
            'acc_control_active': is_in_control,
            'acc_state': 'Hybrid Python+Simulink Mode',  # 状态描述
            'torque_arbitration_active': (hasattr(acc_decision, 'torque_arbitration_active') and
                                          acc_decision.torque_arbitration_active),
            'V_target_kmh': acc_params['V_target_kmh'],
            'V_min_kmh': acc_params['V_min_kmh'],
            'G2_s': acc_params['G2_s'],
            'throttle': throttle,
            'brake': brake,
            'steer': steer,
            'lane_offset': lane_offset
        }
