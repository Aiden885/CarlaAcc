"""
ACC手动控制模块 - 集成显示版本
将CARLA画面渲染到Pygame窗口中，实现单窗口控制
"""

import carla
import math
import numpy as np
import pygame
from pygame.locals import *
import argparse
import logging
import weakref

# 导入ACC相关模块
from acc_decision import ACCDecisionModule, ACCCommand
from acc_planning_control import ACCPlanningControl
from three_mode_controller import set_three_mode_parameters


class CarlaCameraManager:
    """CARLA相机管理器"""

    def __init__(self, parent_actor, hud_width, hud_height):
        self.sensor = None
        self._parent = parent_actor
        self._hud_width = hud_width
        self._hud_height = hud_height
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=-25.0)),  # 第三人称视角
            carla.Transform(carla.Location(x=1.6, z=1.7)),  # 第一人称视角
            carla.Transform(carla.Location(x=-8.0, z=3.0), carla.Rotation(pitch=-15.0)),  # 后视角
        ]
        self._transform_index = 0
        self._camera_image = None

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        # 创建相机
        self._camera_bp = bp_library.find('sensor.camera.rgb')
        self._camera_bp.set_attribute('image_size_x', str(hud_width))
        self._camera_bp.set_attribute('image_size_y', str(hud_height))
        self._camera_bp.set_attribute('fov', '90')

        # 生成相机
        self._spawn_camera()

    def _spawn_camera(self):
        """生成相机"""
        if self.sensor is not None:
            self.sensor.destroy()

        self.sensor = self._parent.get_world().spawn_actor(
            self._camera_bp,
            self._camera_transforms[self._transform_index],
            attach_to=self._parent
        )

        # 设置回调函数
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CarlaCameraManager._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        """解析相机图像"""
        self = weak_self()
        if not self:
            return

        # 将CARLA图像转换为pygame surface
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # 去掉alpha通道
        array = array[:, :, ::-1]  # BGR -> RGB

        self._camera_image = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def toggle_camera(self):
        """切换相机视角"""
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def get_camera_image(self):
        """获取当前相机图像"""
        return self._camera_image

    def destroy(self):
        """销毁相机"""
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()


class ACCManualControl:
    """
    ACC手动控制类 - 集成显示版本
    将CARLA画面集成到Pygame窗口中
    """

    def __init__(self):
        # 初始化pygame
        pygame.init()
        pygame.font.init()

        # 显示设置
        self.display_width = 1280
        self.display_height = 720
        self.display = pygame.display.set_mode(
            (self.display_width, self.display_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("ACC Manual Control - Integrated View")

        # 字体设置
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self.small_font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.large_font = pygame.font.Font(pygame.font.get_default_font(), 24)

        # CARLA连接
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.target_vehicle = None
        self.camera_manager = None

        # ACC相关
        self.acc_decision = ACCDecisionModule(initial_V3_kmh=50.0, initial_G1_m=15.0, initial_time_gap=2.0)
        self.acc_decision.set_debug(True)
        self.acc_controller = None

        # 控制状态
        self.manual_control_active = True
        self.acc_control_active = False

        # 基础车辆控制
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0

        # 运行状态
        self.running = True
        self.clock = pygame.time.Clock()

        # 状态显示
        self.show_help = False
        self.show_info = True  # 显示信息覆盖层
        self.help_text = self._create_help_text()

    def _create_help_text(self):
        """创建帮助文本"""
        help_lines = [
            "=== ACC Manual Control Help ===",
            "",
            "Vehicle Control:",
            "  W/Up Arrow    : Throttle",
            "  S/Down Arrow  : Brake",
            "  A/Left Arrow  : Steer Left",
            "  D/Right Arrow : Steer Right",
            "  Space         : Hand Brake",
            "",
            "ACC Control:",
            "  1  : ACC Engage/开启",
            "  2  : ACC Exit/退出",
            "  3  : Cruise Mode/一键定速巡航",
            "  Q  : Increase Speed/增速",
            "  E  : Decrease Speed/降速",
            "  R  : Increase Distance/增距",
            "  T  : Decrease Distance/降距",
            "",
            "View Control:",
            "  C  : Toggle Camera View",
            "  I  : Toggle Info Display",
            "  H  : Toggle Help",
            "  P  : Toggle ACC Debug",
            "",
            "System:",
            "  ESC: Quit/退出",
        ]
        return help_lines

    def init_carla(self):
        """初始化CARLA环境"""
        try:
            # 连接CARLA
            print("正在连接CARLA服务器...")
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)

            # 检查连接

            self.world = self.client.get_world()
            self.world = self.client.load_world('Town05', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

            print(f"当前地图: {self.world.get_map().name}")

            # 设置同步模式
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / 60.0
            self.world.apply_settings(settings)

            # 获取地图和蓝图
            carla_map = self.world.get_map()
            blueprint_library = self.world.get_blueprint_library()

            # 生成车辆
            print("正在生成车辆...")
            success = self._spawn_vehicles(blueprint_library, carla_map)
            if not success:
                return False

            # 创建相机管理器
            if self.ego_vehicle:
                self.camera_manager = CarlaCameraManager(
                    self.ego_vehicle,
                    self.display_width,
                    self.display_height
                )
                print("相机初始化完成")

            # 初始化ACC控制器
            if self.ego_vehicle:
                self.acc_controller = ACCPlanningControl(
                    self.ego_vehicle,
                    target_speed_kmh=30.0,
                    time_gap=2.0,
                    max_follow_distance=50.0
                )

                # 设置三模式参数
                set_three_mode_parameters(V1_kmh=20, V2_kmh=30, V3_kmh=50, G1_m=15.0, G2_s=2.0)
                print("ACC控制器初始化完成")

            print("CARLA环境初始化完成")
            return True

        except Exception as e:
            print(f"CARLA初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _spawn_vehicles(self, blueprint_library, carla_map):
        """生成车辆"""
        try:
            # 获取车辆蓝图
            ego_bp = blueprint_library.filter('vehicle.audi.etron')[0] if blueprint_library.filter(
                'vehicle.audi.etron') else blueprint_library.filter('vehicle.*')[0]
            target_bp = blueprint_library.filter('vehicle.tesla.model3')[0] if blueprint_library.filter(
                'vehicle.tesla.model3') else blueprint_library.filter('vehicle.*')[1]

            # 获取生成点
            spawn_points = carla_map.get_spawn_points()
            if len(spawn_points) < 2:
                print("错误: 地图生成点不足")
                return False

            # 生成自车
            ego_spawn = spawn_points[0]
            self.ego_vehicle = self.world.try_spawn_actor(ego_bp, ego_spawn)
            if not self.ego_vehicle:
                print("错误: 无法生成自车")
                return False

            # 生成目标车辆（前方50米）
            target_spawn = carla.Transform()
            forward_vector = ego_spawn.get_forward_vector()
            target_spawn.location = ego_spawn.location + forward_vector * 50.0
            target_spawn.rotation = ego_spawn.rotation

            self.target_vehicle = self.world.try_spawn_actor(target_bp, target_spawn)
            if self.target_vehicle:
                self.target_vehicle.set_autopilot(True)
                print(f"成功生成目标车辆")

            return True

        except Exception as e:
            print(f"车辆生成失败: {e}")
            return False

    def handle_keyboard_input(self):
        """处理键盘输入"""
        keys = pygame.key.get_pressed()

        # 基础车辆控制（只在手动模式下有效）
        if self.manual_control_active:
            # 油门
            if keys[K_w] or keys[K_UP]:
                self.throttle = min(1.0, self.throttle + 0.01)
                if self.acc_control_active:
                    ego_speed = self.get_vehicle_speed()
                    self.acc_decision.process_command(ACCCommand.THROTTLE, ego_speed)
                    self.acc_control_active = False
            else:
                self.throttle = max(0.0, self.throttle - 0.05)

            # 刹车
            if keys[K_s] or keys[K_DOWN]:
                self.brake = min(1.0, self.brake + 0.05)
                if self.acc_control_active:
                    ego_speed = self.get_vehicle_speed()
                    self.acc_decision.process_command(ACCCommand.BRAKE, ego_speed)
                    self.acc_control_active = False
            else:
                self.brake = max(0.0, self.brake - 0.1)

            # 转向
            if keys[K_a] or keys[K_LEFT]:
                self.steer = max(-1.0, self.steer - 0.02)
            elif keys[K_d] or keys[K_RIGHT]:
                self.steer = min(1.0, self.steer + 0.02)
            else:
                self.steer = self.steer * 0.9

        # 手刹
        hand_brake = keys[K_SPACE]

        # 应用控制
        if self.ego_vehicle and self.manual_control_active:
            control = carla.VehicleControl()
            control.throttle = self.throttle
            control.brake = self.brake
            control.steer = self.steer
            control.hand_brake = hand_brake
            self.ego_vehicle.apply_control(control)

    def handle_events(self):
        """处理pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            elif event.type == pygame.KEYDOWN:
                # 退出
                if event.key == K_ESCAPE:
                    self.running = False

                # 视图控制
                elif event.key == K_c:
                    if self.camera_manager:
                        self.camera_manager.toggle_camera()

                elif event.key == K_i:
                    self.show_info = not self.show_info

                elif event.key == K_h:
                    self.show_help = not self.show_help

                elif event.key == K_p:
                    debug_state = not self.acc_decision.debug
                    self.acc_decision.set_debug(debug_state)
                    print(f"ACC调试模式: {'开启' if debug_state else '关闭'}")

                # ACC控制
                elif event.key == K_1:
                    self._process_acc_command(ACCCommand.ENGAGE)

                elif event.key == K_2:
                    self._process_acc_command(ACCCommand.EXIT)

                elif event.key == K_3:
                    self._process_acc_command(ACCCommand.CRUISE_MODE)

                elif event.key == K_q:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.INCREASE_SPEED)

                elif event.key == K_e:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.DECREASE_SPEED)

                elif event.key == K_r:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.INCREASE_DISTANCE)

                elif event.key == K_t:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.DECREASE_DISTANCE)

    def _process_acc_command(self, command):
        """处理ACC指令"""
        if not self.ego_vehicle:
            return

        ego_speed = self.get_vehicle_speed()
        target_distance = self.get_vehicle_distance()
        has_target = target_distance < 50.0

        state, mode, msg = self.acc_decision.process_command(
            command, ego_speed, has_target, target_distance if has_target else None)

        acc_params = self.acc_decision.get_current_parameters()
        self.acc_control_active = acc_params['is_active']
        self.manual_control_active = not self.acc_control_active

        print(f"ACC指令 {command.value}: {msg}")

    def get_vehicle_speed(self):
        """获取车辆速度 (km/h)"""
        if not self.ego_vehicle:
            return 0.0

        velocity = self.ego_vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return speed_ms * 3.6

    def get_vehicle_distance(self):
        """获取与前车距离"""
        if not self.ego_vehicle or not self.target_vehicle:
            return float('inf')

        ego_loc = self.ego_vehicle.get_location()
        target_loc = self.target_vehicle.get_location()

        distance = math.sqrt(
            (ego_loc.x - target_loc.x) ** 2 +
            (ego_loc.y - target_loc.y) ** 2
        )
        return distance

    def update_acc_control(self):
        """更新ACC控制"""
        if not self.acc_control_active or not self.acc_controller:
            return

        try:
            ego_speed = self.get_vehicle_speed()
            target_distance = self.get_vehicle_distance()
            decision_output = self.acc_decision.get_decision_output(ego_speed, target_distance)

            if decision_output['control_enabled']:
                target_info = None
                if not decision_output['force_cruise_mode'] and target_distance < 100:
                    target_velocity = self.target_vehicle.get_velocity() if self.target_vehicle else carla.Vector3D(0,
                                                                                                                    0,
                                                                                                                    0)
                    target_speed = math.sqrt(target_velocity.x ** 2 + target_velocity.y ** 2 + target_velocity.z ** 2)
                    ego_velocity = self.ego_vehicle.get_velocity()
                    ego_speed_ms = math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2)
                    relative_speed = target_speed - ego_speed_ms
                    target_info = [target_distance, relative_speed, 0, 0, 0, 0, 0, 0]

                control = self.acc_controller.cruise_control(0.0, target_info)
                if control.brake < 0.01:
                    control.brake = 0
                self.ego_vehicle.apply_control(control)

        except Exception as e:
            print(f"ACC控制更新错误: {e}")

    def render_display(self):
        """渲染显示内容"""
        # 显示CARLA相机画面
        if self.camera_manager:
            camera_image = self.camera_manager.get_camera_image()
            if camera_image:
                self.display.blit(camera_image, (0, 0))
        else:
            self.display.fill((0, 0, 0))

        # 显示信息覆盖层
        if self.show_info:
            self._render_info_overlay()

        # 显示帮助信息
        if self.show_help:
            self._render_help()

        # 更新显示
        pygame.display.flip()

    def _render_info_overlay(self):
        """渲染信息覆盖层"""
        # 创建半透明背景
        info_surface = pygame.Surface((400, 500))
        info_surface.set_alpha(180)
        info_surface.fill((0, 0, 0))

        # 获取状态信息
        ego_speed = self.get_vehicle_speed()
        target_distance = self.get_vehicle_distance()
        has_target = target_distance < 50.0
        acc_params = self.acc_decision.get_current_parameters()
        acc_status = self.acc_decision.get_status_info()

        # 渲染信息文本
        y_offset = 10
        line_height = 25

        # 标题
        title_text = self.font.render("ACC Control System", True, (0, 255, 255))
        info_surface.blit(title_text, (10, y_offset))
        y_offset += line_height + 10

        # 车辆状态
        info_texts = [
            ("Speed", f"{ego_speed:.1f} km/h", (255, 255, 255)),
            ("Target", f"{target_distance:.1f} m" if has_target else "No Target",
             (255, 255, 0) if has_target else (128, 128, 128)),
            ("Control", "ACC" if self.acc_control_active else "Manual",
             (0, 255, 0) if self.acc_control_active else (255, 255, 255)),
            ("", "", (255, 255, 255)),
            ("ACC State", acc_status['state_description'], (0, 255, 255)),
            ("Cruise Mode", "ON" if acc_params.get('cruise_mode_active', False) else "OFF",
             (255, 255, 0) if acc_params.get('cruise_mode_active', False) else (128, 128, 128)),
            ("", "", (255, 255, 255)),
            ("V3 (Max)", f"{acc_params['V3_kmh']:.1f} km/h", (255, 255, 255)),
            ("G1 (Min Dist)", f"{acc_params['G1_m']:.1f} m", (255, 255, 255)),
            ("G2 (Time Gap)", f"{acc_params['G2_s']:.1f} s", (255, 255, 255)),
        ]

        for label, value, color in info_texts:
            if label:
                label_text = self.small_font.render(f"{label}:", True, (200, 200, 200))
                value_text = self.small_font.render(value, True, color)
                info_surface.blit(label_text, (10, y_offset))
                info_surface.blit(value_text, (150, y_offset))
            y_offset += line_height

        # 控制提示
        y_offset += 20
        hint_text = self.small_font.render("Press H for Help", True, (255, 255, 0))
        info_surface.blit(hint_text, (10, y_offset))

        # 显示信息面板
        self.display.blit(info_surface, (10, 10))

        # 显示速度表
        self._render_speedometer(ego_speed)

    def _render_speedometer(self, speed):
        """渲染速度表"""
        # 速度表位置和大小
        center_x = self.display_width - 150
        center_y = self.display_height - 150
        radius = 100

        # 绘制速度表背景
        pygame.draw.circle(self.display, (50, 50, 50), (center_x, center_y), radius, 3)

        # 绘制速度刻度
        for i in range(0, 181, 20):
            angle = math.radians(180 - i)
            start_x = center_x + (radius - 10) * math.cos(angle)
            start_y = center_y - (radius - 10) * math.sin(angle)
            end_x = center_x + radius * math.cos(angle)
            end_y = center_y - radius * math.sin(angle)
            pygame.draw.line(self.display, (200, 200, 200), (start_x, start_y), (end_x, end_y), 2)

        # 绘制速度指针
        speed_angle = math.radians(180 - min(speed * 1.8, 180))  # 0-100 km/h映射到0-180度
        pointer_x = center_x + (radius - 20) * math.cos(speed_angle)
        pointer_y = center_y - (radius - 20) * math.sin(speed_angle)
        pygame.draw.line(self.display, (255, 0, 0), (center_x, center_y), (pointer_x, pointer_y), 3)

        # 显示速度数值
        speed_text = self.large_font.render(f"{int(speed)}", True, (255, 255, 255))
        text_rect = speed_text.get_rect(center=(center_x, center_y + 30))
        self.display.blit(speed_text, text_rect)

        unit_text = self.small_font.render("km/h", True, (200, 200, 200))
        unit_rect = unit_text.get_rect(center=(center_x, center_y + 50))
        self.display.blit(unit_text, unit_rect)

    def _render_help(self):
        """渲染帮助信息"""
        help_surface = pygame.Surface((700, 600))
        help_surface.set_alpha(200)
        help_surface.fill((0, 0, 0))

        y_offset = 10
        for line in self.help_text:
            if line.startswith("==="):
                color = (255, 255, 0)
            elif line.startswith("  "):
                color = (200, 200, 200)
            elif line and not line.startswith(" "):
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)

            if line.strip():
                text_surface = self.small_font.render(line, True, color)
                help_surface.blit(text_surface, (10, y_offset))
            y_offset += 18

        help_x = (self.display_width - 700) // 2
        help_y = (self.display_height - 600) // 2
        self.display.blit(help_surface, (help_x, help_y))
        pygame.draw.rect(self.display, (255, 255, 255), (help_x, help_y, 700, 600), 2)

    def run(self):
        """主运行循环"""
        print("=== ACC手动控制系统启动 (集成显示版本) ===")
        print("系统将在单个窗口中显示CARLA画面和ACC控制信息")
        print()

        if not self.init_carla():
            print("❌ CARLA初始化失败")
            return

        print("✅ 系统初始化成功")
        print("📋 按H键查看帮助信息")
        print("👁️  按I键切换信息显示")
        print("📷 按C键切换相机视角")
        print()

        try:
            while self.running:
                self.clock.tick(60)

                self.handle_events()
                if not self.running:
                    break

                self.handle_keyboard_input()

                if self.acc_control_active:
                    self.update_acc_control()

                if self.world:
                    self.world.tick()

                self.render_display()

        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")

        # 销毁相机
        if self.camera_manager:
            self.camera_manager.destroy()

        # 销毁车辆
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
        if self.target_vehicle:
            self.target_vehicle.destroy()

        # 恢复异步模式
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        # 退出pygame
        pygame.quit()

        print("资源清理完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ACC Manual Control - Integrated Display')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    control_system = ACCManualControl()
    if args.debug:
        control_system.acc_decision.set_debug(True)

    try:
        control_system.run()
    except Exception as e:
        print(f"程序运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
