"""
ACC系统配置类
统一管理所有可配置参数，便于调整和维护
"""
import carla


class ACCConfig:
    """ACC系统配置参数集中管理"""

    def __init__(self):
        # ========== 显示配置 ==========
        self.display_width = 1280
        self.display_height = 720

        # ========== CARLA连接配置 ==========
        self.carla_host = 'localhost'
        self.carla_port = 2000
        self.carla_timeout = 60.0
        self.map_name = 'acc_30km_new'

        # 同步模式配置
        self.synchronous_mode = True
        self.fixed_delta_seconds = 0.05  # 20 FPS (恢复为0.05s，提供更流畅的仿真)

        # 实时模式（速率限制）
        self.enable_realtime = True  # True=1:1实时速度, False=全速运行（默认，3-4倍快）
        self.realtime_target_fps = 20  # 实时模式目标FPS

        # ========== 车辆生成配置 ==========
        # 自车蓝图
        self.ego_vehicle_blueprint = 'vehicle.audi.etron'
        # 目标车蓝图
        self.target_vehicle_blueprint = 'vehicle.tesla.model3'

        # 侧向切入工况配置
        self.enable_cut_in_scenario = True          # 是否启用切入测试工况
        self.cut_in_vehicle_blueprint = 'vehicle.tesla.model3'
        self.cut_in_from_left = True                 # True 从左侧切入，False 从右侧
        self.cut_in_trigger_time_s = 5.0             # 多少秒后触发切入
        self.cut_in_lateral_threshold_m = 0.5        # 进入本车道的横向阈值（米）
        self.cut_in_target_spawn_location = carla.Location(x=2393.492188, y=-340.496368, z=35.952347)
        self.cut_in_side_spawn_location = carla.Location(x=2394.350342, y=-367.901031, z=37.677494)


        self.cut_in_side_spawn_back_offset_m = 10.0  # 侧车沿车道后移距离（米），0=不后移  切入10.0
        self.cut_in_side_spawn_z_lift_m = 0.5        # 侧车生成抬升高度（米），避免贴地失败
        self.cut_in_lane_entry_margin_m = 0.1        # 车身中心跨过车道线内侧距离阈值（米）

        # 侧向切出工况配置
        self.enable_cut_out_scenario = False          # 是否启用切出测试工况
        self.cut_out_vehicle_blueprint = 'vehicle.tesla.model3'
        self.cut_out_trigger_time_s = 15.0            # 多少秒后触发切出前车变道
        self.cut_out_change_to_right = True           # True 向右切出，False 向左
        self.cut_out_front_spawn_location = carla.Location(x=2566.59, y=2065.70, z=12)
        self.cut_out_cut_back_distance_m = 50.0       # 切出车相对前车的后移距离（米）

        # 固定生成点 (Town04默认位置)
        self.spawn_location = carla.Location(x=2511.432617, y=1281.097046, z=0.036094)
        self.spawn_z_offset = 0.1  # 生成高度偏移，避免掉落
        self.ego_spawn_distance = 50.0  # 自车生成距离（米，目标车后方） 切入50.0

        # ========== Traffic Manager配置 ==========
        self.tm_port = 8000
        self.tm_global_distance = 2.0  # 全局跟车距离
        self.tm_target_vehicle_distance = 10.0  # 目标车跟车距离

        # 前车速度控制
        self.assumed_road_speed_limit_kmh = 30.0  # 假设道路限速
        self.target_speed_kmh = 80.0  # 前车初始目标速度 (改为40 km/h，合理的测试速度)
        self.use_constant_velocity = True  # 使用恒速模式（不受路口影响）

        # ========== 集成UDP配置 ==========
        self.integrated_udp_send_port = 27000       # Python → Simulink UDP Receive
        self.integrated_udp_recv_port = 27001       # Python ← Simulink UDP Send
        self.integrated_udp_local_send_port = 9090  # Python 源端口
        self.integrated_udp_timeout = 2.0           # UDP超时（秒）

        # ========== ACC系统参数 ==========
        self.acc_params = {
            'V_target_kmh': 50.0,      # 默认巡航速度
            'V_min_kmh': 20.0,         # 最小速度阈值
            'G2_s': 4.0,               # 时距参数
            'V_threshold_kmh': 50.0,   # 模式切换阈值
            'speed_step': 5.0          # 速度调整步长
        }
        self.max_target_speed_kmh = 150.0    # 巡航速度上限（键盘/ACC控制使用）

        self.sppvt_params = {
            'dt': 0.05,
            'kp': 1.0,
            'max_accel': 2.0,
            'max_decel': -4.56,    # 基于 CARLA Audi e-tron 真实测量的理论最大减速度
            'delta': 0.05,
            'eta': 0.2,
            'sppvt_rho': 0.1,
        }
        self.integrated_sppvt_rho = self.sppvt_params['sppvt_rho']

        # ========== 感知配置 ==========
        self.max_follow_distance = 50.0     # 最大跟车距离（用于初始化）
        self.detection_range = 200.0        # 前车检测范围（米）

        # ========== 横向控制器配置 ==========
        self.lateral_controller_params = {
            'kp': 0.1,
            'ki': 0.01,
            'kd': 0.02
        }

        # 预瞄参数（注释掉表示使用默认值）
        # self.lateral_lookahead_params = {
        #     'weight_current': 0.7,
        #     'weight_lookahead': 0.3,
        #     'base_distance': 8.0,
        #     'gain': 0.3
        # }

        # ========== 扭矩转换器配置 ==========
        self.use_torque_converter = True  # 是否使用物理模型转换器

        # SPPVT 输出到物理量的缩放系数
        # 基于 SPPVT 输出范围 ±2.5 和 Audi e-tron 真实参数设计
        self.sppvt_accel_scale = 210.0   # 加速：SPPVT输出 → 发动机扭矩 (N·m)

        self.sppvt_decel_scale = 2.0     # 减速：SPPVT输出 → 减速度 (m/s²)
                                          # 理论最大减速度：4.56 m/s² (从CARLA测得)

        # ========== 斜坡速度控制器配置 ==========
        self.ramp_controller_params = {
            'start_speed_kmh': 80.0,    # 斜坡起始速度
            'target_speed_kmh': 120.0,  # 斜坡目标速度
            'duration_s': 10.0          # 斜坡持续时间（秒）
        }

        # ========== 绘图器配置 ==========
        self.plotter_max_points = 5000        # 绘图器最大数据点数
        self.plotter_update_interval = 100    # 绘图器更新间隔（ms）
        self.use_result_plotter = True        # True=结果保存模式, False=实时测试模式（默认保存模式）

        # ========== 性能分析配置 ==========
        self.perf_max_samples = 1000          # 性能统计最大样本数（防止内存无限增长）
        self.performance_report_interval = 10.0  # 性能报告输出间隔（秒）

        # ========== 手动控制配置 ==========
        self.manual_throttle_step = 0.1   # 油门累加步长（每帧）
        self.manual_brake_step = 0.2      # 刹车累加步长（每帧）

        # ========== CSV记录配置 ==========
        self.csv_output_file = 'speed_data_integrated.csv'

        # ========== ACC决策+SPPVT配置 ==========
        self.acc_decision_debug = True        # 是否开启调试模式
        self.use_realtime_sppvt = False       # 是否使用实时SPPVT（False=使用完整Simulink模型）

    def get_acc_params(self):
        """获取ACC参数副本（避免外部直接修改）"""
        return self.acc_params.copy()

    def get_lateral_controller_params(self):
        """获取横向控制器参数"""
        return self.lateral_controller_params.copy()

    def get_ramp_controller_params(self):
        """获取斜坡速度控制器参数"""
        return self.ramp_controller_params.copy()

    def get_sppvt_params(self):
        return self.sppvt_params.copy()

    def __str__(self):
        """打印配置摘要"""
        return f"""
=== ACC系统配置摘要 ===
CARLA: {self.carla_host}:{self.carla_port} | 地图: {self.map_name}
同步模式: {self.synchronous_mode} | 时间步长: {self.fixed_delta_seconds}s
ACC巡航速度: {self.acc_params['V_target_kmh']} km/h | 时距: {self.acc_params['G2_s']}s
前车目标速度: {self.target_speed_kmh} km/h | 恒速模式: {self.use_constant_velocity}
扭矩转换器: {'启用' if self.use_torque_converter else '禁用'}
绘图器: {'结果保存模式' if self.use_result_plotter else '实时测试模式'}
========================
"""
