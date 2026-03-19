"""
配置迁移验证脚本
对比重构前后的配置值，确保完全一致
"""
from acc_config import ACCConfig

def verify_config():
    """验证配置值与原代码中的硬编码值一致"""
    config = ACCConfig()

    print("=" * 60)
    print("配置迁移验证报告")
    print("=" * 60)

    # 验证计数器
    passed = 0
    failed = 0

    # 定义预期值（从原代码中提取的硬编码值）
    expected_values = {
        # 显示配置
        'display_width': 1280,
        'display_height': 720,

        # CARLA连接配置
        'carla_host': 'localhost',
        'carla_port': 2000,
        'carla_timeout': 60.0,
        'map_name': 'Town04',
        'fixed_delta_seconds': 0.05,

        # 车辆蓝图
        'target_vehicle_blueprint': 'vehicle.tesla.model3',
        'ego_vehicle_blueprint': 'vehicle.audi.etron',

        # 生成点配置
        'spawn_z_offset': 0.1,
        'ego_spawn_distance': 10.0,

        # Traffic Manager
        'tm_port': 8000,
        'tm_global_distance': 2.0,
        'tm_target_vehicle_distance': 10.0,
        'assumed_road_speed_limit_kmh': 30.0,
        'target_speed_kmh': 90.0,
        'use_constant_velocity': True,

        # ACC参数
        'acc_params.V_target_kmh': 50.0,
        'acc_params.V_min_kmh': 20.0,
        'acc_params.G2_s': 2.0,
        'acc_params.V_threshold_kmh': 50.0,
        'acc_params.speed_step': 5.0,

        # 感知配置
        'max_follow_distance': 50.0,
        'detection_range': 200.0,

        # 横向控制器
        'lateral_controller_params.kp': 0.1,
        'lateral_controller_params.ki': 0.01,
        'lateral_controller_params.kd': 0.02,

        # 扭矩转换器
        'use_torque_converter': True,

        # 斜坡速度控制
        'ramp_controller_params.start_speed_kmh': 90.0,
        'ramp_controller_params.target_speed_kmh': 120.0,
        'ramp_controller_params.duration_s': 10.0,

        # 绘图器
        'plotter_max_points': 5000,
        'plotter_update_interval': 100,
        'use_result_plotter': True,

        # 手动控制
        'manual_throttle_step': 0.1,
        'manual_brake_step': 0.2,

        # 性能分析
        'performance_report_interval': 10.0,

        # CSV文件
        'csv_output_file': 'speed_data_integrated.csv',

        # ACC决策
        'acc_decision_debug': True,
        'use_realtime_sppvt': False,
    }

    # 验证每个值
    for key, expected in expected_values.items():
        if '.' in key:
            # 处理嵌套属性（如 acc_params.V_target_kmh）
            parts = key.split('.')
            actual = getattr(config, parts[0])[parts[1]]
        else:
            actual = getattr(config, key)

        if actual == expected:
            print(f"[PASS] {key:40s} = {actual}")
            passed += 1
        else:
            print(f"[FAIL] {key:40s} = {actual} (expected: {expected})")
            failed += 1

    print("=" * 60)
    print(f"Result: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("SUCCESS: Config migration verified! All values match original code.")
        return True
    else:
        print("WARNING: Config migration verification failed! Check inconsistent items.")
        return False

if __name__ == '__main__':
    success = verify_config()
    exit(0 if success else 1)
