import carla
import math
import time


class SimpleACC:
    def __init__(self):
        # 连接CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()

        # 启用世界同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.01  # 每帧 0.01 秒（20 FPS）
        self.world.apply_settings(settings)

        # 加载地图(保留Town04，确保一致性)
        self.world = self.client.load_world('Town04')

        # 车辆对象
        self.ego_vehicle = None
        self.target_vehicle = None
        self.vehicles = []

        # PID控制参数
        self.desired_distance = 10.0  # 期望跟车距离(米)
        self.kp = 0.05  # 比例增益
        self.ki = 0  # 积分增益
        self.kd = 0  # 微分增益


        # PID控制器状态
        self.prev_error = 0
        self.integral = 0

        # 初始化
        self.setup_vehicles()
        self.setup_spectator()



    def setup_vehicles(self):
        """设置自车和目标车辆"""
        # 获取车辆蓝图
        blueprint_library = self.world.get_blueprint_library()
        ego_vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]
        target_vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        # 获取地图和所有可能的生成点
        map = self.world.get_map()
        spawn_points = map.get_spawn_points()

        # 尝试多个生成点直到成功
        for spawn_index in range(len(spawn_points)):
            try:
                start_point = spawn_points[spawn_index]

                # 生成目标车辆
                self.target_vehicle = self.world.spawn_actor(target_vehicle_bp, start_point)
                self.vehicles.append(self.target_vehicle)
                print(f"目标车辆已生成在位置: {start_point.location}")

                # 在目标车辆后方找一个适合的位置生成自车
                spawn_transform = carla.Transform(
                    start_point.location - carla.Location(x=20.0),  # 在目标车后方20米
                    start_point.rotation
                )

                self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, spawn_transform)
                self.vehicles.append(self.ego_vehicle)
                print(f"自车已生成在位置: {spawn_transform.location}")

                # 找到有效的生成点，退出循环
                break

            except RuntimeError as e:
                print(f"在生成点 {spawn_index} 尝试失败: {e}")
                # 清理任何可能已创建的车辆
                for vehicle in self.vehicles:
                    if vehicle.is_alive:
                        vehicle.destroy()
                self.vehicles = []
                self.target_vehicle = None
                self.ego_vehicle = None

                # 继续尝试下一个生成点
                continue

        # 检查是否成功生成了车辆
        if not self.ego_vehicle or not self.target_vehicle:
            print("警告: 无法生成车辆，尝试使用随机生成点")
            try:
                # 随机选择两个相距足够远的生成点
                import random
                available_points = spawn_points.copy()
                random.shuffle(available_points)

                # 生成目标车辆
                target_point = available_points[0]
                self.target_vehicle = self.world.spawn_actor(target_vehicle_bp, target_point)
                self.vehicles.append(self.target_vehicle)

                # 寻找距离目标车至少30米的点
                for ego_point in available_points[1:]:
                    if target_point.location.distance(ego_point.location) > 30.0:
                        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, ego_point)
                        self.vehicles.append(self.ego_vehicle)
                        break

            except RuntimeError as e:
                print(f"无法生成车辆: {e}")
                return

        # 启用自动驾驶（仅适用于目标车辆）
        if self.target_vehicle:
            tm = self.client.get_trafficmanager(8000)
            tm.set_synchronous_mode(True)
            self.target_vehicle.set_autopilot(True, tm.get_port())
            tm.auto_lane_change(self.target_vehicle, False)  # 禁用车道变换
            tm.vehicle_percentage_speed_difference(self.target_vehicle, 40.0)  # 让目标车辆慢一些（60%的速度限制）

        # 等待车辆稳定
        for _ in range(20):
            self.world.tick()

    def stanley_control(self, waypoints, k=0.5):
        """Stanley控制器"""
        if not waypoints:
            return 0.0

        # 获取自车的位置、朝向和速度
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        # 获取车速（避免除零错误）
        velocity = self.ego_vehicle.get_velocity()
        speed = max(0.1, math.sqrt(velocity.x ** 2 + velocity.y ** 2))

        # 找到最近的路点
        closest_waypoint = None
        min_distance = float('inf')

        for waypoint in waypoints:
            dist = vehicle_location.distance(waypoint.transform.location)
            if dist < min_distance:
                min_distance = dist
                closest_waypoint = waypoint

        if closest_waypoint:
            # 计算横向偏差
            wp_direction = closest_waypoint.transform.get_forward_vector()
            wp_direction_norm = math.sqrt(wp_direction.x ** 2 + wp_direction.y ** 2)

            # 车辆位置到路点的向量
            dx = vehicle_location.x - closest_waypoint.transform.location.x
            dy = vehicle_location.y - closest_waypoint.transform.location.y

            # 计算横向偏差（正负表示在路点的左侧或右侧）
            lateral_error = (dx * wp_direction.y - dy * wp_direction.x) / wp_direction_norm

            # 计算航向偏差
            waypoint_yaw = math.radians(closest_waypoint.transform.rotation.yaw)
            heading_error = self.normalize_angle(waypoint_yaw - vehicle_yaw)

            # Stanley控制律
            crosstrack_term = math.atan2(k * lateral_error, speed)
            steer = heading_error + crosstrack_term
            #打印heading_error和crosstrack_term
            print(f"横向偏差: {lateral_error:.2f}, 航向偏差: {heading_error:.2f}, 侧向控制: {crosstrack_term:.2f}")


            # 限制转向角
            return max(-1.0, min(1.0, steer))

        return 0.0

    def get_waypoints_ahead(self, distance=10.0, interval=1.0):
        """获取前方的车道路点"""
        if not self.ego_vehicle:
            return []

        # 获取当前车辆所在的路点
        map = self.world.get_map()
        vehicle_location = self.ego_vehicle.get_location()
        current_waypoint = map.get_waypoint(vehicle_location)

        # 生成前方路点
        waypoints = [current_waypoint]
        distance_accumulated = 0.0

        while distance_accumulated < distance:
            # 获取下一个路点
            next_waypoints = current_waypoint.next(interval)

            if not next_waypoints:
                break

            # 取第一个路点（保持在当前车道）
            current_waypoint = next_waypoints[0]
            waypoints.append(current_waypoint)
            distance_accumulated += interval

        return waypoints

    def normalize_angle(self, angle):
        """将角度标准化到[-pi, pi]范围内"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


    def get_vehicle_distance(self):
        """计算自车与前车之间的距离"""
        if self.ego_vehicle and self.target_vehicle:
            ego_location = self.ego_vehicle.get_location()
            target_location = self.target_vehicle.get_location()
            distance = ego_location.distance(target_location)
            return distance
        return None

    def setup_spectator(self):
        """设置观察视角"""
        self.spectator = self.world.get_spectator()
        self.update_spectator()

    def update_spectator(self):
        """更新观察视角"""
        if self.ego_vehicle and self.spectator:
            # 获取自车当前位置
            ego_transform = self.ego_vehicle.get_transform()
            ego_location = ego_transform.location

            # 获取自车所在的车道路径点
            waypoint = self.world.get_map().get_waypoint(ego_location)

            # 获取车道前方10米处的路径点
            forward_waypoint = waypoint.next(10.0)[0] if waypoint.next(10.0) else waypoint

            # 使用车道朝向计算观察者位置（后方8米，高度6米）
            lane_yaw = waypoint.transform.rotation.yaw
            lane_yaw_rad = math.radians(lane_yaw)

            # 计算后方15米的偏移
            offset_x = -15.0 * math.cos(lane_yaw_rad)
            offset_y = -15.0 * math.sin(lane_yaw_rad)

            spectator_location = carla.Location(
                x=ego_location.x + offset_x,
                y=ego_location.y + offset_y,
                z=ego_location.z + 6.0
            )

            # 计算观察者朝向：指向前方10米路径点
            direction = forward_waypoint.transform.location - spectator_location
            direction_norm = math.sqrt(direction.x ** 2 + direction.y ** 2 + direction.z ** 2)
            pitch = -math.degrees(math.asin(direction.z / direction_norm)) if direction_norm > 0 else 0
            yaw = math.degrees(math.atan2(direction.y, direction.x))

            spectator_rotation = carla.Rotation(pitch=pitch, yaw=yaw)

            # 设置观察者的位置和朝向
            spectator_transform = carla.Transform(spectator_location, spectator_rotation)
            self.spectator.set_transform(spectator_transform)

    def pid_control(self, error, dt):
        """PID控制器实现"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def follow_target_vehicle(self):
        """纵向和横向控制实现"""
        # 获取前方车道路点
        waypoints = self.get_waypoints_ahead()

        # 使用Stanley控制器计算横向控制
        steer = self.stanley_control(waypoints)

        # 纵向控制（保持你现有的PID控制器）
        actual_distance = self.get_vehicle_distance()

        if actual_distance is None:
            # 没有目标，减速停车
            control = carla.VehicleControl(throttle=0.0, steer=steer, brake=1.0)
            self.ego_vehicle.apply_control(control)
            print("未检测到目标，停车")
            return

        # 计算距离误差
        ego_location = self.ego_vehicle.get_location()
        target_location = self.target_vehicle.get_location()
        ego_transform = self.ego_vehicle.get_transform()

        # 将目标车位置转换到自车坐标系
        forward_vector = ego_transform.get_forward_vector()
        right_vector = ego_transform.get_right_vector()

        # 计算相对位置向量
        rel_pos = target_location - ego_location

        # 计算前向距离
        forward_distance = rel_pos.x * forward_vector.x + rel_pos.y * forward_vector.y

        # 计算距离误差
        distance_error = forward_distance - self.desired_distance

        # PID控制
        dt = 0.05  # 控制周期
        throttle_control = self.pid_control(distance_error, dt)

        # 转换为油门和刹车
        throttle = max(0.0, min(1.0, throttle_control))
        brake = max(0.0, min(1.0, -throttle_control))

        # 应用控制
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego_vehicle.apply_control(control)

        # 输出控制信息
        print(f"距离: {actual_distance:.2f}m, 目标: {self.desired_distance:.2f}m, "
              f"误差: {distance_error:.2f}m, 控制: 油门={throttle:.2f}, 转向={steer:.2f}, 刹车={brake:.2f}")

    def run(self):
        """主循环"""
        try:
            print("开始ACC跟车控制. 按Ctrl+C停止.")
            while True:
                # 更新模拟
                self.world.tick()

                # 更新观察者位置
                self.update_spectator()

                # 执行跟车控制
                self.follow_target_vehicle()

                # 控制频率
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("用户停止")
        finally:
            # 清理
            for vehicle in self.vehicles:
                if vehicle.is_alive:
                    vehicle.destroy()
            print("清理完成")


# 启动ACC控制器
if __name__ == "__main__":
    acc = SimpleACC()
    acc.run()