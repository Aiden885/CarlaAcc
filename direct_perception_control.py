import carla
import math
import numpy as np
import threading
import time
import sys
import os

# 确保可以导入acc模块
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)


class DirectPerceptionControl:
    def __init__(self):
        # 连接CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()

        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.01
        self.world.apply_settings(settings)

        # 加载地图 - 保持与acc.py相同的地图设置
        self.world = self.client.load_world('Town04')

        # 导入并初始化acc类 - 这将创建车辆和传感器
        from acc import acc
        self.perception = acc()

        # 使用感知模块的车辆
        self.ego_vehicle = self.perception.ego_vehicle

        # 控制参数 - 可以根据需要调整
        self.desired_distance = 10.0
        self.kp = 0.03
        self.ki = 0
        self.kd = 0
        self.prev_error = 0
        self.integral = 0

        # 设置观察者
        self.spectator = self.world.get_spectator()

        # 添加目标跟踪变量
        self.selected_target_id = None
        self.selected_target = None

        print("直接感知控制初始化完成")

        # 启动感知线程 - 确保不要重复启动
        self.start_perception_threads()

    def start_perception_threads(self):
        """启动感知模块的传感器线程"""
        try:
            # 确保之前没有启动过这些监听器
            self.radar_thread = threading.Thread(
                target=lambda: self.perception.radar.listen(self.perception.radar_callback))
            self.camera_thread = threading.Thread(
                target=lambda: self.perception.camera.listen(self.perception.camera_callback))
            self.lidar_thread = threading.Thread(
                target=lambda: self.perception.lidar.listen(self.perception.lidar_callback))

            self.radar_thread.daemon = True
            self.camera_thread.daemon = True
            self.lidar_thread.daemon = True

            self.radar_thread.start()
            self.camera_thread.start()
            self.lidar_thread.start()
            print("感知线程启动成功")
        except Exception as e:
            print(f"启动感知线程失败: {e}")
            # 如果线程已经启动，忽略错误
            pass

    def normalize_angle(self, angle):
        """将角度标准化到[-pi, pi]范围内"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def pid_control(self, error, dt):
        """PID控制器实现"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def update_spectator(self):
        """更新观察者位置"""
        if self.ego_vehicle:
            transform = self.ego_vehicle.get_transform()
            spectator_transform = carla.Transform(
                transform.location + carla.Location(x=-8.0, z=6.0),
                carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw)
            )
            self.spectator.set_transform(spectator_transform)

    def adaptive_k_lateral_error(self, lateral_error, k_base=1, error_threshold=4.0):
        """基于横向偏差的自适应K值"""
        error_factor = 1

        # 添加类型检查，防止错误
        if not isinstance(lateral_error, (int, float)):
            print(f"错误: lateral_error 类型不正确: {type(lateral_error)}")
            return k_base  # 返回默认值

        adjustment = error_factor * min(abs(lateral_error) / error_threshold, 1.0)
        k = k_base * (1.0 + adjustment)
        return k

    def calculate_deviations_from_perception(self, perception_target):
        """
        从感知目标计算控制偏差 - 适应不同类型的目标数据结构
        """
        if perception_target is None:
            return None, None, None

        try:
            # 获取自车朝向(在任何情况下都需要)
            ego_transform = self.ego_vehicle.get_transform()
            ego_yaw = math.radians(ego_transform.rotation.yaw)

            # 检查目标类型 - 修改为同时支持list和tuple
            if (isinstance(perception_target, list) or isinstance(perception_target, tuple)) and len(
                    perception_target) >= 10:
                # 目标是雷达检测数组/元组 [x, y, z, w, l, h, vx, vy, vz, id]
                target_x, target_y, target_z, target_w, target_l, target_h, target_vx, target_vy, target_vz, target_id = perception_target

                # 打印雷达原始数据，便于分析
                print(
                    f"雷达目标原始数据: x={target_x:.2f}m, y={target_y:.2f}m, vx={target_vx:.2f}m/s, vy={target_vy:.2f}m/s")

                # 假设雷达数据已经在自车坐标系中
                longitudinal_deviation = target_x - self.desired_distance
                lateral_deviation = target_y

                # 计算航向偏差(修复：考虑自车朝向)
                if abs(target_vx) > 0.01 or abs(target_vy) > 0.01:
                    # 使用速度向量推断目标朝向
                    # target_heading = math.atan2(target_vy, target_vx)
                    # # 计算相对于自车的航向偏差
                    # heading_deviation = self.normalize_angle(target_heading - ego_yaw)

                    target_heading = math.atan2(target_y, target_x)
                    heading_deviation = self.normalize_angle(target_heading - ego_yaw)
                else:
                    # 使用位置向量推断朝向
                    # 注意：这可能不准确，因为位置向量不一定代表朝向
                    target_heading = math.atan2(target_y, target_x)
                    heading_deviation = self.normalize_angle(target_heading - ego_yaw)

                # 输出详细计算过程
                print(
                    f"雷达目标计算过程: 目标朝向={math.degrees(target_heading):.2f}°, 自车朝向={math.degrees(ego_yaw):.2f}°")

            else:
                print(f"未知的目标数据类型: {type(perception_target)}")
                return None, None, None

            # 最终输出偏差
            print(
                f"计算偏差: 纵向={longitudinal_deviation:.2f}m, 横向={lateral_deviation:.2f}m, 航向={math.degrees(heading_deviation):.2f}°")

            # 检查偏差的合理性
            if abs(lateral_deviation) > 10:
                print(f"警告: 横向偏差异常大! 可能存在坐标系或符号问题")
            if abs(heading_deviation) > math.pi / 2:
                print(f"警告: 航向偏差异常大! 可能存在朝向计算问题")

            return longitudinal_deviation, lateral_deviation, heading_deviation

        except Exception as e:
            print(f"计算偏差时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def select_target_with_minimum_lateral_deviation(self):
        """选择并固定跟踪前排中间的目标车辆"""
        print("\n-- 目标选择过程 --")

        # 优先使用perception中指定的target_vehicle
        if hasattr(self.perception, 'target_vehicle') and self.perception.target_vehicle:
            target_vehicle = self.perception.target_vehicle
            target_id = getattr(target_vehicle, 'id', None)

            # 如果是初次选择，记录目标ID
            if self.selected_target_id is None and target_id is not None:
                self.selected_target_id = target_id
                print(f"初次选择目标: 中间车辆 (ID: {self.selected_target_id})")

            # 在雷达检测到的目标中查找目标车辆
            if hasattr(self.perception, 'track_id') and self.perception.track_id:
                for track in self.perception.track_id:
                    # 根据ID找到指定目标
                    if self.selected_target_id is not None and track[9] == self.selected_target_id:
                        self.selected_target = track
                        print(f"固定跟踪目标ID: {self.selected_target_id}")
                        return track


            # 如果雷达未检测到目标但存在target_vehicle，使用其位置信息
            if self.selected_target_id is not None and target_id == self.selected_target_id:
                # 将目标车辆位置转换为雷达坐标系
                target_location = target_vehicle.get_location()
                ego_transform = self.ego_vehicle.get_transform()

                # 计算目标车辆相对于自车的位置
                dx = target_location.x - ego_transform.location.x
                dy = target_location.y - ego_transform.location.y
                dz = target_location.z - ego_transform.location.z

                # 创建一个类似雷达检测结果的数组 [x, y, z, w, l, h, vx, vy, vz, id]
                # 假设宽度、长度、高度和速度
                target_info = [dx, dy, dz, 2.0, 4.5, 1.8, 0.0, 0.0, 0.0, self.selected_target_id]
                self.selected_target = target_info
                print(f"使用直接位置计算跟踪目标 ID: {self.selected_target_id}")
                return target_info

        print("未找到指定的目标车辆")
        return None

    def direct_control_from_perception(self):
        """基于感知直接计算控制"""
        try:
            print("\n==== 开始新一帧控制计算 ====")

            # === 第1步：打印所有可用的雷达目标信息 ===
            if hasattr(self.perception, 'track_id') and self.perception.track_id:
                if isinstance(self.perception.track_id, list) and len(self.perception.track_id) > 0:
                    print(f"可用雷达目标数量: {len(self.perception.track_id)}")
                    for i, track in enumerate(self.perception.track_id):
                        # 打印每个目标的关键信息
                        print(f"目标[{i}]: 位置(x={track[0]:.2f}m, y={track[1]:.2f}m, z={track[2]:.2f}m), "
                              f"速度(vx={track[6]:.2f}m/s, vy={track[7]:.2f}m/s), ID={track[9]}")
                else:
                    print("雷达目标列表为空或格式不正确")
            else:
                print("未找到雷达目标数据")

            # === 第2步：目标选择 - 使用新的选择方法 ===
            radar_target = self.select_target_with_minimum_lateral_deviation()

            # 如果没有雷达目标，尝试使用target_vehicle
            perception_target = radar_target
            if perception_target is None:
                perception_target = getattr(self.perception, 'target_vehicle', None)
                if hasattr(perception_target, 'get_location'):
                    location = perception_target.get_location()
                    print(
                        f"使用CARLA Actor作为目标 - ID: {perception_target.id}, 位置: {location.x:.2f}, {location.y:.2f}, {location.z:.2f}")
                else:
                    print(f"未能获取有效目标 - perception_target类型: {type(perception_target)}")

            # === 第3步：无目标处理 ===
            if perception_target is None:
                control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.3)
                self.ego_vehicle.apply_control(control)
                print("未检测到目标，减速行驶")
                return

            # === 第4步：计算偏差 ===
            print("\n-- 偏差计算过程 --")

            # 获取自车位置和朝向
            ego_transform = self.ego_vehicle.get_transform()
            ego_location = ego_transform.location
            ego_rotation = ego_transform.rotation
            ego_yaw = math.radians(ego_rotation.yaw)
            print(
                f"自车位置: x={ego_location.x:.2f}, y={ego_location.y:.2f}, z={ego_location.z:.2f}, 朝向={ego_rotation.yaw:.2f}°")

            # 计算各种偏差
            longitudinal_deviation, lateral_deviation, heading_deviation = self.calculate_deviations_from_perception(
                perception_target)

            # 打印计算结果
            if isinstance(perception_target, (list, tuple)):
                print(
                    f"原始目标数据 - 位置: ({perception_target[0]:.2f}, {perception_target[1]:.2f}, {perception_target[2]:.2f})")
                if abs(perception_target[0]) > 30 or abs(perception_target[1]) > 20:
                    print(f"警告: 目标位置可能不合理! x={perception_target[0]:.2f}, y={perception_target[1]:.2f}")

            if longitudinal_deviation is None:
                control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.3)
                self.ego_vehicle.apply_control(control)
                print("偏差计算失败，减速行驶")
                return

            # 检查偏差值的合理性
            if abs(longitudinal_deviation) > 30:
                print(f"警告: 纵向偏差异常大! {longitudinal_deviation:.2f}m")
            if abs(lateral_deviation) > 10:
                print(f"警告: 横向偏差异常大! {lateral_deviation:.2f}m")
            if abs(heading_deviation) > math.radians(60):
                print(f"警告: 航向偏差异常大! {math.degrees(heading_deviation):.2f}°")

            # === 第5步：控制计算 ===
            print("\n-- 控制计算过程 --")

            # 获取自车速度
            velocity = self.ego_vehicle.get_velocity()
            speed = max(0.01, math.sqrt(velocity.x ** 2 + velocity.y ** 2))
            print(f"自车速度: {speed:.2f} m/s ({speed * 3.6:.2f} km/h)")

            # 纵向控制（PID控制器）
            dt = 0.05  # 控制周期
            throttle_control = self.pid_control(longitudinal_deviation, dt)
            print(f"PID计算: 纵向偏差={longitudinal_deviation:.2f}m, PID输出={throttle_control:.2f}")

            # 转换为油门和刹车
            raw_throttle = max(0.0, min(1.0, throttle_control))
            raw_brake = max(0.0, min(1.0, -throttle_control))

            # 横向控制（Stanley控制器公式）
            k = 0.1
            crosstrack_term = math.atan2(k * lateral_deviation, speed)
            raw_steer = heading_deviation + crosstrack_term

            # 打印详细计算过程
            print(f"Stanley计算: 横向偏差={lateral_deviation:.2f}m, k={k}")
            print(
                f"横向项: k*lateral_deviation={k * lateral_deviation:.4f}, speed={speed:.2f} → atan2()={math.degrees(crosstrack_term):.2f}°")
            print(f"航向偏差: {math.degrees(heading_deviation):.2f}°")
            print(f"原始转向值: {raw_steer:.4f} rad ({math.degrees(raw_steer):.2f}°)")

            # 限制在[-1, 1]范围内
            steer = max(-1.0, min(1.0, raw_steer))

            # 如果限幅生效，输出警告
            if steer != raw_steer:
                print(f"警告: 转向值被限制! 原始值={raw_steer:.2f}, 限制后={steer:.2f}")

            # 最终控制值
            throttle = raw_throttle
            brake = raw_brake


            # === 第6步：应用控制 ===
            control = carla.VehicleControl(throttle=throttle, steer= steer, brake=brake)
            self.ego_vehicle.apply_control(control)

            # 最终控制输出日志
            print(f"\n最终控制: 油门={throttle:.2f}, 转向={steer:.2f}, 刹车={brake:.2f}")
            print("==== 本帧控制计算完成 ====\n")

        except Exception as e:
            print(f"控制计算出错: {e}")
            import traceback
            traceback.print_exc()
            # 出错时安全停车
            control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5)
            self.ego_vehicle.apply_control(control)

    def run(self):
        """主循环"""
        try:
            print("启动直接感知控制. 按Ctrl+C停止.")

            # 等待感知模块初始化完成
            time.sleep(2.0)

            while True:
                # 更新模拟
                self.world.tick()

                # 更新观察者位置
                self.update_spectator()

                # 执行直接控制
                self.direct_control_from_perception()

        except KeyboardInterrupt:
            print("用户停止")
        finally:
            # 清理资源
            try:
                # 如果感知模块有destroy方法
                if hasattr(self.perception, 'destroy'):
                    self.perception.destroy()
                else:
                    # 手动清理
                    if hasattr(self.perception, 'radar') and self.perception.radar:
                        self.perception.radar.stop()
                        self.perception.radar.destroy()
                    if hasattr(self.perception, 'camera') and self.perception.camera:
                        self.perception.camera.stop()
                        self.perception.camera.destroy()
                    if hasattr(self.perception, 'lidar') and self.perception.lidar:
                        self.perception.lidar.stop()
                        self.perception.lidar.destroy()
                    if hasattr(self.perception, 'ego_vehicle') and self.perception.ego_vehicle:
                        self.perception.ego_vehicle.destroy()
            except Exception as e:
                print(f"清理资源时出错: {e}")
            print("清理完成")


# 启动控制器
if __name__ == "__main__":
    try:
        controller = DirectPerceptionControl()
        controller.run()
    except Exception as e:
        print(f"运行控制器时出错: {e}")