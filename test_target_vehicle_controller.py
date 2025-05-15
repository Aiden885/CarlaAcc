import carla
import time
import csv
import math
from target_vehicle_controller import TargetVehicleController


class TargetVehicleTest:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.world = self.client.load_world('Town05')
        self.map = self.world.get_map()
        self.target_vehicle = None
        self.spectator = None

    def setup_spectator(self):
        """设置观察视角"""
        self.spectator = self.world.get_spectator()
        self.update_spectator()

    def update_spectator(self):
        """更新观察视角"""
        if self.target_vehicle and self.spectator:
            # 获取目标车辆当前位置
            target_transform = self.target_vehicle.get_transform()
            target_location = target_transform.location

            # 获取目标车辆所在的车道路径点
            waypoint = self.world.get_map().get_waypoint(target_location)

            # 获取车道前方10米处的路径点
            forward_waypoint = waypoint.next(10.0)[0] if waypoint.next(10.0) else waypoint

            # 使用车道朝向计算观察者位置（后方15米，高度6米）
            lane_yaw = waypoint.transform.rotation.yaw
            lane_yaw_rad = math.radians(lane_yaw)

            # 计算后方15米的偏移
            offset_x = -15.0 * math.cos(lane_yaw_rad)
            offset_y = -15.0 * math.sin(lane_yaw_rad)

            spectator_location = carla.Location(
                x=target_location.x + offset_x,
                y=target_location.y + offset_y,
                z=target_location.z + 6.0
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

    def test_target_vehicle_controller(self):
        # 设置所有交通灯为绿灯并冻结
        for actor in self.world.get_actors().filter('traffic.traffic_light'):
            actor.set_state(carla.TrafficLightState.Green)  # 强制绿灯
            actor.freeze(True)  # 冻结状态，防止变回红灯

        # 生成目标车辆
        blueprint_library = self.world.get_blueprint_library()
        target_vehicle_bp = blueprint_library.filter('vehicle.audi.tt')[0]

        # 获取自定义坐标附近的最近车道点作为 spawn 点
        custom_location = carla.Location(x=207.362045, y=-23.449020, z=2.348918)
        waypoint = self.map.get_waypoint(custom_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        # 使用车道点的 transform 作为出生点，并略微抬高避免地面穿透
        spawn_transform = waypoint.transform
        spawn_transform.location.z += 0.3  # 稍微抬高

        # 生成车辆
        self.target_vehicle = self.world.spawn_actor(target_vehicle_bp, spawn_transform)

        # 启用自动驾驶并绑定到 Traffic Manager
        self.target_vehicle.set_autopilot(True, 8000)  # 绑定到端口 8000
        # 初始化控制器
        controller = TargetVehicleController(self.client, self.target_vehicle, self.map, speed_range=(30.0, 70.0))
        controller.start_update_thread()

        # 设置观察视角
        self.setup_spectator()

        # 记录数据
        data = []
        start_time = time.time()
        try:
            while True:

                current_time = time.time() - start_time
                speed = controller.get_current_speed()
                is_curved = controller.is_curved_road()
                pitch = controller.get_road_pitch()
                data.append([current_time, speed, is_curved, pitch])

                # 显示实时速度
                if self.target_vehicle:
                    speed_text = f"Speed: {speed:.1f} km/h"
                    vehicle_location = self.target_vehicle.get_location()
                    # 在车辆上方2米处显示速度
                    text_location = carla.Location(
                        x=vehicle_location.x,
                        y=vehicle_location.y,
                        z=vehicle_location.z + 2.0
                    )
                    self.world.debug.draw_string(
                        text_location,
                        speed_text,
                        draw_shadow=False,
                        color=carla.Color(r=255, g=0, b=0),  # 红色文字
                        life_time=0.1,  # 持续时间与循环同步
                        persistent_lines=False
                    )

                # 更新观察视角
                self.update_spectator()

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("测试中断")
        finally:
            controller.stop_update_thread()
            if self.target_vehicle:
                self.target_vehicle.destroy()
            # 保存数据到 CSV
            with open('target_vehicle_data.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Time (s)', 'Speed (km/h)', 'Is Curved', 'Pitch (deg)'])
                writer.writerows(data)
            print("数据已保存到 target_vehicle_data.csv")


if __name__ == '__main__':
    test = TargetVehicleTest()
    test.test_target_vehicle_controller()
