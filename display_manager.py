import pygame
from pygame.locals import *
import numpy as np
import weakref
import carla
import math


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


class DisplayManager:
    """显示管理器 - 处理所有Pygame相关的显示功能"""

    def __init__(self, width=1280, height=720):
        # Pygame初始化
        pygame.init()
        pygame.font.init()

        # 显示设置
        self.display_width = width
        self.display_height = height
        self.display = pygame.display.set_mode(
            (self.display_width, self.display_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("ACC Integrated Control System")

        # 字体设置 - 支持中文显示
        self.font = self._get_chinese_font(20)
        self.small_font = self._get_chinese_font(16)
        self.large_font = self._get_chinese_font(24)

        # 显示控制
        self.show_help = False
        self.show_info = True
        self.help_text = self._create_help_text()

        # 相机管理器
        self.camera_manager = None

        # 运行控制
        self.clock = pygame.time.Clock()

    def _get_chinese_font(self, size):
        """获取支持中文的字体"""
        # 尝试不同的中文字体路径（Windows/Linux/Mac）
        chinese_fonts = [
            # Windows 系统字体
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            # Linux 系统字体
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/PingFang.ttc",  # Mac
            # 相对路径（如果有项目字体文件）
            "./fonts/simhei.ttf",
            "./fonts/msyh.ttf",
        ]

        # 尝试加载中文字体
        for font_path in chinese_fonts:
            try:
                font = pygame.font.Font(font_path, size)
                # 测试是否能正确渲染中文
                test_surface = font.render("中文测试", True, (255, 255, 255))
                return font
            except (FileNotFoundError, OSError):
                continue

        # 如果都失败了，尝试系统字体
        try:
            # 尝试使用系统默认字体渲染中文
            system_fonts = pygame.font.get_fonts()

            # 优先尝试一些常见的中文字体名称
            preferred_fonts = ['microsoftyaheui', 'simhei', 'simsun', 'dengxian', 'fangsong']
            for font_name in preferred_fonts:
                if font_name in system_fonts:
                    try:
                        font = pygame.font.SysFont(font_name, size)
                        test_surface = font.render("中文测试", True, (255, 255, 255))
                        return font
                    except:
                        continue

            # 最后尝试系统默认字体
            return pygame.font.SysFont('arial', size)

        except Exception as e:
            print(f"字体加载失败: {e}")
            # 返回默认字体作为后备
            return pygame.font.Font(None, size)

    def _create_help_text(self):
        """创建帮助文本"""
        help_lines = [
            "=== ACC Integrated Control System ===",
            "",
            "Operation Steps:",
            "  1. Drive manually to suitable speed (>30km/h)",
            "  2. Press SPACE to enable ACC system (standby mode)",
            "  3. Press E to activate ACC (current speed) or Q (requires history)",
            "",
            "Vehicle Control:",
            "  W/Up Arrow    : Throttle",
            "  S/Down Arrow  : Brake", 
            "  A/Left Arrow  : Steer Left",
            "  D/Right Arrow : Steer Right",
            "",
            "ACC Control:",
            "  SPACE : ACC System On/Off (MUST press first)",
            "  E     : Decrease Speed/Current Speed Activate",
            "  Q     : Increase Speed/Inherit Activate (needs history)",
            "  R/T   : Increase/Decrease Distance",
            "  C     : Cancel ACC",
            "  W     : Throttle (Torque Arbitration when ACC active)",
            "",
            "View Control:",
            "  I  : Toggle Info Display",
            "  O  : Toggle OpenCV Window", 
            "  H  : Toggle Help",
            "  P  : Toggle ACC Debug",
            "",
            "System:",
            "  ESC: Quit/退出",
        ]
        return help_lines

    def init_camera_manager(self, parent_actor):
        """初始化相机管理器"""
        if parent_actor:
            self.camera_manager = CarlaCameraManager(
                parent_actor,
                self.display_width,
                self.display_height
            )
            print("相机管理器初始化完成")

    def handle_display_events(self):
        """处理显示相关的事件，返回需要传递给主程序的事件"""
        main_events = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                main_events.append(('quit', None))

            elif event.type == pygame.KEYDOWN:
                # 显示相关的事件直接处理
                if event.key == K_c:
                    if self.camera_manager:
                        self.camera_manager.toggle_camera()
                        print("切换相机视角")

                elif event.key == K_i:
                    self.show_info = not self.show_info
                    print(f"信息显示: {'开启' if self.show_info else '关闭'}")

                elif event.key == K_h:
                    self.show_help = not self.show_help

                # 其他事件传递给主程序
                else:
                    main_events.append(('keydown', event.key))

            elif event.type == pygame.KEYUP:
                main_events.append(('keyup', event.key))

        return main_events

    def render_display(self, system_info):
        """渲染显示内容

        Args:
            system_info: 包含系统状态信息的字典
        """
        # 显示CARLA相机画面
        if self.camera_manager:
            camera_image = self.camera_manager.get_camera_image()
            if camera_image:
                self.display.blit(camera_image, (0, 0))
        else:
            self.display.fill((0, 0, 0))

        # 显示信息覆盖层
        if self.show_info:
            self._render_info_overlay(system_info)

        # 显示帮助信息
        if self.show_help:
            self._render_help()

        # 更新显示
        pygame.display.flip()

    def _render_info_overlay(self, info):
        """渲染信息覆盖层"""
        # 创建半透明背景
        info_surface = pygame.Surface((400, 550))
        info_surface.set_alpha(180)
        info_surface.fill((0, 0, 0))

        # 渲染信息文本
        y_offset = 10
        line_height = 25

        # 标题
        title_text = self.font.render("ACC Control System", True, (0, 255, 255))
        info_surface.blit(title_text, (10, y_offset))
        y_offset += line_height + 10

        # 车辆状态
        info_texts = [
            ("Speed", f"{info.get('ego_speed', 0):.1f} km/h", (255, 255, 255)),
            ("Target", f"{info.get('target_distance', 0):.1f} m" if info.get('has_target', False) else "No Target",
             (255, 255, 0) if info.get('has_target', False) else (128, 128, 128)),
            ("", "", (255, 255, 255)),
            ("ACC System", "ON" if info.get('acc_system_enabled', False) else "OFF (Press SPACE)",
             (0, 255, 0) if info.get('acc_system_enabled', False) else (255, 128, 128)),
            ("ACC Control", "ACTIVE" if info.get('acc_control_active', False) else "STANDBY",
             (0, 255, 0) if info.get('acc_control_active', False) else (255, 255, 0)),
            ("ACC State", info.get('acc_state', 'Unknown'), (0, 255, 255)),
            ("Torque Arbitr.", "ACTIVE" if info.get('torque_arbitration_active', False) else "OFF",
             (255, 0, 255) if info.get('torque_arbitration_active', False) else (128, 128, 128)),
            ("", "", (255, 255, 255)),
            ("Cruise Mode", "ON" if info.get('cruise_mode', False) else "OFF",
             (255, 255, 0) if info.get('cruise_mode', False) else (128, 128, 128)),
            ("", "", (255, 255, 255)),
            ("Cruise Speed", f"{info.get('cruise_speed_kmh', 0):.1f} km/h", (255, 255, 0)),  # 显示当前巡航速度
            ("V_target (Target)", f"{info.get('V_target_kmh', 0):.1f} km/h", (200, 200, 200)),  # 目标速度（两模式切换阈值）
            ("V_min (Min Speed)", f"{info.get('V_min_kmh', 0):.1f} km/h", (255, 255, 255)),  # 最低速度要求
            ("G2 (Time Gap)", f"{info.get('G2_s', 0):.1f} s", (255, 255, 255)),
            ("", "", (255, 255, 255)),
            ("Manual Input", "", (200, 200, 200)),
            ("  Throttle", f"{info.get('throttle', 0):.2f}", (255, 128, 0)),
            ("  Brake", f"{info.get('brake', 0):.2f}", (255, 0, 0)),
            ("  Steer", f"{info.get('steer', 0):.2f}", (128, 255, 128)),
        ]

        for label, value, color in info_texts:
            if label:
                label_text = self.small_font.render(f"{label}:", True, (200, 200, 200))
                value_text = self.small_font.render(value, True, color)
                info_surface.blit(label_text, (10, y_offset))
                if value:
                    info_surface.blit(value_text, (150, y_offset))
            y_offset += line_height

        # 控制提示
        y_offset += 20
        hint_text = self.small_font.render("Press H for Help", True, (255, 255, 0))
        info_surface.blit(hint_text, (10, y_offset))

        # 显示信息面板
        self.display.blit(info_surface, (10, 10))

        # 显示速度表
        self._render_speedometer(info.get('ego_speed', 0))

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

    def tick(self, fps=60):
        """更新时钟"""
        return self.clock.tick(fps)

    def destroy(self):
        """销毁显示管理器"""
        if self.camera_manager:
            self.camera_manager.destroy()
        pygame.quit()
