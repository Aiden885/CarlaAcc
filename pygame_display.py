import pygame
from pygame.locals import *
import math

class PygameDisplay:
    def __init__(self, width=1280, height=720, title="ACC Integrated Control System"):
        """初始化Pygame显示"""
        pygame.init()
        pygame.font.init()

        self.display_width = width
        self.display_height = height
        self.display = pygame.display.set_mode(
            (self.display_width, self.display_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption(title)

        # 字体设置 - 使用支持中文的字体
        try:
            self.font = pygame.font.SysFont('Microsoft YaHei', 20)
            self.small_font = pygame.font.SysFont('Microsoft YaHei', 16)
            self.large_font = pygame.font.SysFont('Microsoft YaHei', 24)
        except:
            font_path = 'simhei.ttf'
            self.font = pygame.font.Font(font_path, 20)
            self.small_font = pygame.font.Font(font_path, 16)
            self.large_font = pygame.font.Font(font_path, 24)

        # 显示控制
        self.show_help = False
        self.show_info = True
        self.help_text = self._create_help_text()

    def _create_help_text(self):
        """创建帮助文本"""
        help_lines = [
            "=== ACC Integrated Control System ===",
            "",
            "车辆控制:",
            "  W/上箭头    : 油门",
            "  S/下箭头    : 刹车",
            "  A/左箭头    : 左转",
            "  D/右箭头    : 右转",
            "  空格        : 手刹",
            "",
            "ACC控制:",
            "  1  : ACC开启",
            "  2  : ACC退出",
            "  3  : 定速巡航",
            "  Q  : 增速",
            "  E  : 降速",
            "  R  : 增距",
            "  T  : 降距",
            "",
            "视图控制:",
            "  C  : 切换视角",
            "  I  : 切换信息显示",
            "  O  : 切换OpenCV窗口",
            "  H  : 切换帮助",
            "  P  : 切换ACC调试",
            "",
            "系统:",
            "  ESC: 退出",
        ]
        return help_lines

    def render(self, camera_image, ego_speed, target_distance, has_target, acc_params, acc_status):
        """渲染Pygame显示内容"""
        if camera_image:
            self.display.blit(camera_image, (0, 0))
        else:
            self.display.fill((0, 0, 0))

        if self.show_info:
            self._render_info_overlay(ego_speed, target_distance, has_target, acc_params, acc_status)

        if self.show_help:
            self._render_help()

        pygame.display.flip()

    def _render_info_overlay(self, ego_speed, target_distance, has_target, acc_params, acc_status):
        """渲染信息覆盖层"""
        info_surface = pygame.Surface((400, 550))
        info_surface.set_alpha(180)
        info_surface.fill((0, 0, 0))

        y_offset = 10
        line_height = 25

        title_text = self.font.render("ACC控制系统", True, (0, 255, 255))
        info_surface.blit(title_text, (10, y_offset))
        y_offset += line_height + 10

        info_texts = [
            ("速度", f"{ego_speed:.1f} km/h", (255, 255, 255)),
            ("目标距离", f"{target_distance:.1f} m" if has_target else "无目标",
             (255, 255, 0) if has_target else (128, 128, 128)),
            ("控制模式", "ACC" if acc_params.get('is_active', False) else "手动",
             (0, 255, 0) if acc_params.get('is_active', False) else (255, 255, 255)),
            ("", "", (255, 255, 255)),
            ("ACC状态", acc_status['state_description'], (0, 255, 255)),
            ("定速巡航", "开启" if acc_params.get('cruise_mode_active', False) else "关闭",
             (255, 255, 0) if acc_params.get('cruise_mode_active', False) else (128, 128, 128)),
            ("", "", (255, 255, 255)),
            ("V3 (最大)", f"{acc_params['V3_kmh']:.1f} km/h", (255, 255, 255)),
            ("G1 (最小距离)", f"{acc_params['G1_m']:.1f} m", (255, 255, 255)),
            ("G2 (时间间隔)", f"{acc_params['G2_s']:.1f} s", (255, 255, 255)),
        ]

        for label, value, color in info_texts:
            if label:
                label_text = self.small_font.render(f"{label}:", True, (200, 200, 200))
                value_text = self.small_font.render(value, True, color)
                info_surface.blit(label_text, (10, y_offset))
                if value:
                    info_surface.blit(value_text, (150, y_offset))
            y_offset += line_height

        y_offset += 20
        hint_text = self.small_font.render("按 H 查看帮助", True, (255, 255, 0))
        info_surface.blit(hint_text, (10, y_offset))

        self.display.blit(info_surface, (10, 10))
        self._render_speedometer(ego_speed)

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

    def _render_speedometer(self, speed):
        """渲染速度表"""
        center_x = self.display_width - 150
        center_y = self.display_height - 150
        radius = 100

        pygame.draw.circle(self.display, (50, 50, 50), (center_x, center_y), radius, 3)

        for i in range(0, 181, 20):
            angle = math.radians(180 - i)
            start_x = center_x + (radius - 10) * math.cos(angle)
            start_y = center_y - (radius - 10) * math.sin(angle)
            end_x = center_x + radius * math.cos(angle)
            end_y = center_y - radius * math.sin(angle)
            pygame.draw.line(self.display, (200, 200, 200), (start_x, start_y), (end_x, end_y), 2)

        speed_angle = math.radians(180 - min(speed * 1.8, 180))
        pointer_x = center_x + (radius - 20) * math.cos(speed_angle)
        pointer_y = center_y - (radius - 20) * math.sin(speed_angle)
        pygame.draw.line(self.display, (255, 0, 0), (center_x, center_y), (pointer_x, pointer_y), 3)

        speed_text = self.large_font.render(f"{int(speed)}", True, (255, 255, 255))
        text_rect = speed_text.get_rect(center=(center_x, center_y + 30))
        self.display.blit(speed_text, text_rect)

        unit_text = self.small_font.render("km/h", True, (200, 200, 200))
        unit_rect = unit_text.get_rect(center=(center_x, center_y + 50))
        self.display.blit(unit_text, unit_rect)

    def quit(self):
        """退出Pygame"""
        pygame.quit()