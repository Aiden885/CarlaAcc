import pygame
from pygame.locals import *
import numpy as np
import weakref
import carla
import math


class CarlaCameraManager:
    """Helper that manages an RGB camera attached to the ego vehicle."""

    def __init__(self, parent_actor, hud_width, hud_height):
        self.sensor = None
        self._parent = parent_actor
        self._hud_width = hud_width
        self._hud_height = hud_height
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=-25.0)),
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=-8.0, z=3.0), carla.Rotation(pitch=-15.0)),
        ]
        self._transform_index = 0
        self._camera_image = None

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        self._camera_bp = bp_library.find('sensor.camera.rgb')
        self._camera_bp.set_attribute('image_size_x', str(hud_width))
        self._camera_bp.set_attribute('image_size_y', str(hud_height))
        self._camera_bp.set_attribute('fov', '90')

        self._spawn_camera()

    def _spawn_camera(self):
        if self.sensor is not None:
            self.sensor.destroy()

        self.sensor = self._parent.get_world().spawn_actor(
            self._camera_bp,
            self._camera_transforms[self._transform_index],
            attach_to=self._parent
        )

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CarlaCameraManager._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self._camera_image = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def get_camera_image(self):
        return self._camera_image

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()


class DisplayManager:
    """Manage all Pygame rendering for the HUD."""

    def __init__(self, width=1280, height=720):
        pygame.init()
        pygame.font.init()

        self.display_width = width
        self.display_height = height
        self.display = pygame.display.set_mode(
            (self.display_width, self.display_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("ACC Integrated Control System")

        self.font = self._load_font(20, weight='regular')
        self.font_bold = self._load_font(20, weight='semibold')
        self.small_font = self._load_font(17, weight='regular')
        self.large_font = self._load_font(28, weight='semibold')

        self.show_help = False
        self.show_info = True
        self.help_text = self._create_help_text()

        self.camera_manager = None

        self.clock = pygame.time.Clock()
        self.dashboard_panel_width = 1120
        self.dashboard_panel_height = 380
        self.dashboard_margin_bottom = 30

    def _load_font(self, size, weight='regular'):
        regular_candidates = [
            "C:/Windows/Fonts/Microsoft YaHei UI.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhl.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]
        semibold_candidates = [
            "C:/Windows/Fonts/Microsoft YaHei UI Bold.ttf",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        ]
        candidates = regular_candidates if weight == 'regular' else semibold_candidates
        for path in candidates:
            try:
                return pygame.font.Font(path, size)
            except (FileNotFoundError, OSError):
                continue
        return pygame.font.SysFont("Microsoft YaHei UI", size, bold=(weight != 'regular'))

    def _create_help_text(self):
        return [
            "=== ACC Integrated Control System ===",
            "",
            "Vehicle Control:",
            "  W / Up Arrow    : Throttle",
            "  S / Down Arrow  : Brake",
            "  A / Left Arrow  : Steer Left",
            "  D / Right Arrow : Steer Right",
            "",
            "ACC Control:",
            "  SPACE : Toggle ACC System (must enable first)",
            "  E     : Decrease Speed / Activate current speed",
            "  Q     : Increase Speed / Inherit previous speed",
            "  R / T : Increase / Decrease Distance",
            "  C     : Cancel ACC",
            "  W     : Manual Throttle (torque arbitration)",
            "",
            "View Control:",
            "  I  : Toggle Info Display",
            "  O  : Toggle OpenCV Window",
            "  H  : Toggle Help Overlay",
            "  P  : Toggle ACC Debug Mode",
            "",
            "System:",
            "  ESC : Quit",
        ]

    def init_camera_manager(self, parent_actor):
        if parent_actor:
            self.camera_manager = CarlaCameraManager(parent_actor, self.display_width, self.display_height)
            print("Camera manager initialized.")

    def handle_display_events(self):
        main_events = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                main_events.append(('quit', None))
            elif event.type == pygame.KEYDOWN:
                if event.key == K_h:
                    self.show_help = not self.show_help
                elif event.key == K_i:
                    self.show_info = not self.show_info
                elif event.key == K_c:
                    if self.camera_manager:
                        self.camera_manager.toggle_camera()
                else:
                    main_events.append(('keydown', event.key))
            elif event.type == pygame.KEYUP:
                main_events.append(('keyup', event.key))
        return main_events

    def render_display(self, system_info):
        if self.camera_manager:
            camera_image = self.camera_manager.get_camera_image()
            if camera_image:
                self.display.blit(camera_image, (0, 0))
            else:
                self.display.fill((0, 0, 0))
        else:
            self.display.fill((0, 0, 0))

        if self.show_info:
            self._render_info_overlay(system_info)
        if self.show_help:
            self._render_help()

        pygame.display.flip()

    def _render_info_overlay(self, info):
        panel_width = self.dashboard_panel_width
        panel_height = self.dashboard_panel_height
        panel_x = (self.display_width - panel_width) // 2
        panel_y = self.display_height - panel_height - self.dashboard_margin_bottom

        surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        outer_rect = pygame.Rect(0, 0, panel_width, panel_height)
        pygame.draw.rect(surface, (14, 14, 20, 120), outer_rect, border_radius=32)
        pygame.draw.rect(surface, (90, 90, 120, 110), outer_rect, width=2, border_radius=32)

        gauge_radius = 120
        gauge_side_padding = 40
        side_center_x = gauge_radius + gauge_side_padding
        speed_center = (side_center_x, panel_height - 150)
        acc_center = (panel_width - side_center_x, panel_height - 150)
        self._draw_speedometer(surface, speed_center, gauge_radius, info.get('ego_speed', 0.0))
        self._draw_acc_status_gauge(surface, acc_center, gauge_radius, info)

        inner_gap = 20
        content_margin_x = int(side_center_x + gauge_radius + inner_gap)
        content_width = panel_width - content_margin_x * 2

        status_rect = pygame.Rect(content_margin_x, 40, content_width, 72)
        self._draw_status_band(surface, status_rect, info)

        info_rect = pygame.Rect(content_margin_x, 118, content_width, 168)
        self._draw_info_table(surface, info_rect, info)

        bars_top = info_rect.bottom + 14
        bars_height = panel_height - bars_top - 16
        min_bars_height = 60
        if bars_height < min_bars_height:
            bars_height = min_bars_height
            bars_top = panel_height - bars_height - 16
        bars_rect = pygame.Rect(content_margin_x, bars_top, content_width, bars_height)
        self._draw_input_bars(surface, bars_rect, info)

        self.display.blit(surface, (panel_x, panel_y))

    def _draw_speedometer(self, surface, center, radius, speed):
        pygame.draw.circle(surface, (30, 30, 40, 140), center, radius)
        pygame.draw.circle(surface, (18, 18, 26, 140), center, radius - 10)

        max_speed = 180.0
        for tick in range(0, int(max_speed) + 1, 20):
            angle = math.radians(220 - (tick / max_speed) * 260)
            inner = (center[0] + (radius - 24) * math.cos(angle),
                     center[1] - (radius - 24) * math.sin(angle))
            outer = (center[0] + (radius - 6) * math.cos(angle),
                     center[1] - (radius - 6) * math.sin(angle))
            pygame.draw.line(surface, (95, 95, 110), inner, outer, 2)

        clamped = max(0.0, min(speed, max_speed))
        angle = math.radians(220 - (clamped / max_speed) * 260)
        pointer = (center[0] + (radius - 28) * math.cos(angle),
                   center[1] - (radius - 28) * math.sin(angle))
        pygame.draw.line(surface, (240, 100, 100), center, pointer, 4)
        pygame.draw.circle(surface, (230, 230, 235), center, 5)

        speed_text = self.large_font.render(f"{int(clamped)}", True, (240, 240, 245))
        surface.blit(speed_text, speed_text.get_rect(center=(center[0], center[1] + 20)))

        unit_text = self.small_font.render("km/h", True, (170, 170, 190))
        surface.blit(unit_text, unit_text.get_rect(center=(center[0], center[1] + 48)))

        label = self.font_bold.render("SPEED", True, (120, 180, 255))
        surface.blit(label, label.get_rect(center=(center[0], center[1] - radius + 32)))

    def _draw_acc_status_gauge(self, surface, center, radius, info):
        sectors = [
            ("Standby", (110, 110, 120)),
            ("Monitoring", (90, 150, 255)),
            ("Active", (70, 200, 150)),
            ("Torque", (180, 100, 240)),
            ("Fault", (240, 90, 100)),
        ]
        acc_state_text = (info.get('acc_state') or '').lower()
        acc_enabled = info.get('acc_system_enabled', False)
        acc_active = info.get('acc_control_active', False)
        torque_active = info.get('torque_arbitration_active', False)

        if 'fault' in acc_state_text:
            active_idx = 4
        elif torque_active:
            active_idx = 3
        elif acc_active:
            active_idx = 2
        elif acc_enabled:
            active_idx = 1
        else:
            active_idx = 0

        start_angle = math.radians(220)
        total_sweep = math.radians(260)
        sweep = total_sweep / len(sectors)
        steps = 24

        for idx, (_, color) in enumerate(sectors):
            sa = start_angle - idx * sweep
            points = [center]
            for step in range(steps + 1):
                angle = sa - (step / steps) * sweep
                points.append((center[0] + radius * math.cos(angle),
                               center[1] - radius * math.sin(angle)))
            shade = color if idx == active_idx else tuple(int(c * 0.3) for c in color)
            pygame.draw.polygon(surface, shade, points)

        pygame.draw.circle(surface, (22, 22, 30, 130), center, radius - 18)
        pygame.draw.circle(surface, (36, 36, 48, 150), center, radius - 64)

        title = self.font_bold.render("ACC STATUS", True, (130, 160, 255))
        surface.blit(title, title.get_rect(center=(center[0], center[1] - radius + 18)))

        mode_text = 'Torque' if torque_active else 'Active' if acc_active else 'Ready' if acc_enabled else 'Standby'
        mode_color = (180, 100, 240) if torque_active else ((70, 200, 150) if acc_active else (90, 150, 255) if acc_enabled else (160, 160, 170))
        mode_surface = self.font_bold.render(mode_text, True, mode_color)
        surface.blit(mode_surface, mode_surface.get_rect(center=(center[0], center[1] - radius + 42)))

        state_label = info.get('acc_state', 'Unknown')
        state_surface = self.small_font.render(state_label[:18], True, (235, 235, 240))
        surface.blit(state_surface, state_surface.get_rect(center=(center[0], center[1] + 12)))

    def _draw_status_band(self, surface, rect, info):
        pygame.draw.rect(surface, (32, 32, 45, 100), rect, border_radius=18)
        pygame.draw.rect(surface, (80, 80, 110, 90), rect, width=1, border_radius=18)

        items = [
            ("ACC SYS", info.get('acc_system_enabled', False), (70, 200, 140)),
            ("CONTROL", info.get('acc_control_active', False), (90, 160, 255)),
            ("TORQUE", info.get('torque_arbitration_active', False), (180, 110, 255)),
        ]
        segment_width = rect.width // len(items)

        for idx, (label, active, color) in enumerate(items):
            seg_rect = pygame.Rect(rect.x + idx * segment_width + 6,
                                   rect.y + 6,
                                   segment_width - 12,
                                   rect.height - 12)
            base_color = color if active else tuple(int(c * 0.3) for c in color)
            pygame.draw.rect(surface, base_color, seg_rect, border_radius=14)
            pygame.draw.rect(surface, (18, 18, 24), seg_rect, width=2, border_radius=14)

            status_text = 'ON' if active else 'OFF'
            status_surface = self.font_bold.render(status_text, True, (245, 245, 250))
            surface.blit(status_surface, status_surface.get_rect(center=(seg_rect.centerx, seg_rect.y + seg_rect.height * 0.38)))

            label_surface = self.small_font.render(label, True, (240, 240, 245))
            surface.blit(label_surface, label_surface.get_rect(center=(seg_rect.centerx, seg_rect.y + seg_rect.height * 0.78)))

    def _draw_info_table(self, surface, rect, info):
        pygame.draw.rect(surface, (38, 48, 70, 110), rect, border_radius=18)
        pygame.draw.rect(surface, (90, 100, 130, 90), rect, width=1, border_radius=18)

        entries = [
            ("V_target", f"{info.get('V_target_kmh', 0.0):.1f} km/h"),
            ("V_min", f"{info.get('V_min_kmh', 0.0):.1f} km/h"),
            ("G2", f"{info.get('G2_s', 0.0):.1f} s"),
            ("Target Dist", f"{info.get('target_distance', 0.0):.1f} m"),
            ("Lane Offset", f"{info.get('lane_offset', 0.0):+.2f} m"),
            ("Throttle", f"{info.get('throttle', 0.0):.2f}"),
            ("Brake", f"{info.get('brake', 0.0):.2f}"),
            ("Steer", f"{info.get('steer', 0.0):+.2f}"),
        ]
        if not entries:
            return

        columns = 2
        rows = (len(entries) + columns - 1) // columns
        cell_width = rect.width // columns
        cell_height = rect.height // rows
        padding_x = 18
        padding_y = 18

        for index, (label, value) in enumerate(entries):
            col = index // rows
            row = index % rows
            cell_x = rect.x + col * cell_width
            cell_y = rect.y + row * cell_height

            display_text = f"{label}: {value}"
            text_surface = self.font.render(display_text, True, (235, 235, 245))
            surface.blit(text_surface, (cell_x + padding_x, cell_y + padding_y))

    def _draw_input_bars(self, surface, rect, info):
        pygame.draw.rect(surface, (26, 26, 34, 130), rect, border_radius=18)
        pygame.draw.rect(surface, (80, 80, 100, 90), rect, width=1, border_radius=18)

        bar_width = int((rect.width - 60) / 3)
        bar_height = 18
        top = rect.y + 32

        throttle = max(0.0, min(float(info.get('throttle', 0.0)), 1.0))
        brake = max(0.0, min(float(info.get('brake', 0.0)), 1.0))
        steer = max(-1.0, min(float(info.get('steer', 0.0)), 1.0))

        bars = [
            ("Throttle", throttle, (80, 200, 140), rect.x + 15, 'normal'),
            ("Brake", brake, (220, 90, 90), rect.x + 30 + bar_width, 'normal'),
            ("Steer", steer, (90, 150, 255), rect.x + 45 + bar_width * 2, 'centered'),
        ]

        for label, value, color, x, mode in bars:
            bar_rect = pygame.Rect(int(x), top, bar_width, bar_height)
            self._draw_progress_bar(surface, bar_rect, label, value, color, mode)

    def _draw_progress_bar(self, surface, bar_rect, label, value, color, mode='normal'):
        label_text = f"{label} {value:+.2f}" if mode == 'centered' else f"{label} {value:.2f}"
        label_surface = self.small_font.render(label_text, True, (230, 230, 238))
        surface.blit(label_surface, label_surface.get_rect(midbottom=(bar_rect.centerx, bar_rect.y - 6)))

        pygame.draw.rect(surface, (18, 18, 24), bar_rect, border_radius=8)
        pygame.draw.rect(surface, (70, 70, 90), bar_rect, width=1, border_radius=8)

        value = float(value)
        if mode == 'normal':
            value_clamped = max(0.0, min(value, 1.0))
            fill_width = int(round(bar_rect.width * value_clamped))
            if 0 < fill_width < 1:
                fill_width = 1
            if fill_width > 0:
                fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_width, bar_rect.height)
                pygame.draw.rect(surface, color, fill_rect, border_radius=8)
        else:
            half_width = bar_rect.width / 2
            center_x = bar_rect.x + half_width
            zero_rect = pygame.Rect(int(center_x) - 1, bar_rect.y, 2, bar_rect.height)
            pygame.draw.rect(surface, (135, 135, 150), zero_rect)

            value_clamped = max(-1.0, min(value, 1.0))
            if value_clamped >= 0:
                fill_width = int(round(half_width * value_clamped))
                if 0 < fill_width < 1:
                    fill_width = 1
                if fill_width > 0:
                    fill_rect = pygame.Rect(int(center_x), bar_rect.y, fill_width, bar_rect.height)
                    pygame.draw.rect(surface, color, fill_rect, border_radius=8)
            else:
                fill_width = int(round(half_width * (-value_clamped)))
                if 0 < fill_width < 1:
                    fill_width = 1
                if fill_width > 0:
                    fill_rect = pygame.Rect(int(center_x) - fill_width, bar_rect.y, fill_width, bar_rect.height)
                    pygame.draw.rect(surface, color, fill_rect, border_radius=8)

    def _render_help(self):
        help_surface = pygame.Surface((700, 600))
        help_surface.set_alpha(200)
        help_surface.fill((0, 0, 0))

        y_offset = 10
        for line in self.help_text:
            if line.startswith("==="):
                color = (255, 255, 0)
            elif line.startswith("  "):
                color = (200, 200, 200)
            elif line:
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
        return self.clock.tick(fps)

    def destroy(self):
        if self.camera_manager:
            self.camera_manager.destroy()
        pygame.quit()
