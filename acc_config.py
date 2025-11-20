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
        self.fixed_delta_seconds = 0.05  # 20 FPS (50ms per frame)

        # ========== 车辆生成配置 ==========
        # 自车蓝图
        self.ego_vehicle_blueprint = 'vehicle.audi.etron'
        # 目标车蓝图
        self.target_vehicle_blueprint = 'vehicle.tesla.model3'

        # 固定生成点 (Town04默认位置)
        self.spawn_location = carla.Location(x=2511.432617, y=1281.097046, z=0.5)
        self.spawn_z_offset = 0.1  # 生成高度偏移，避免掉落
        self.ego_spawn_distance = 10.0  # 自车生成距离（米，目标车后方）

        # ========== Traffic Manager配置 ==========
        self.tm_port = 8000
        self.tm_global_distance = 2.0  # 全局跟车距离
        self.tm_target_vehicle_distance = 10.0  # 目标车跟车距离

        # 前车速度控制
        self.assumed_road_speed_limit_kmh = 30.0  # 假设道路限速
        self.target_speed_kmh = 90.0  # 前车初始目标速度
        self.use_constant_velocity = True  # 使用恒速模式（不受路口影响）

        # ========== ACC系统参数 ==========
        self.acc_params = {
            'V_target_kmh': 50.0,      # 默认巡航速度
            'V_min_kmh': 20.0,         # 最小速度阈值
            'G2_s': 2.0,               # 时距参数
            'V_threshold_kmh': 50.0,   # 模式切换阈值
            'speed_step': 5.0          # 速度调整步长
        }
        self.max_target_speed_kmh = 150.0    # 巡航速度上限（键盘/ACC控制使用）

        self.sppvt_params = {
            'dt': 0.05,
            'kp': 1.0,
            'max_accel': 2.0,
            'max_decel': -3.0,
            'delta': 0.05,
            'eta': 0.2,
            'sppvt_rho': 0.1,
        }

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

        # ========== 斜坡速度控制器配置 ==========
        self.ramp_controller_params = {
            'start_speed_kmh': 90.0,    # 斜坡起始速度
            'target_speed_kmh': 120.0,  # 斜坡目标速度
            'duration_s': 10.0          # 斜坡持续时间（秒）
        }

        # ========== 绘图器配置 ==========
        self.plotter_max_points = 5000        # 绘图器最大数据点数
        self.plotter_update_interval = 100    # 绘图器更新间隔（ms）
        self.use_result_plotter = True        # True=结果保存模式, False=实时测试模式（默认保存模式）

        # ========== 手动控制配置 ==========
        self.manual_throttle_step = 0.1   # 油门累加步长（每帧）
        self.manual_brake_step = 0.2      # 刹车累加步长（每帧）

        # ========== 性能分析配置 ==========
        self.performance_report_interval = 10.0  # 性能报告输出间隔（秒）

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
