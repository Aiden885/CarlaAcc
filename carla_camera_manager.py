import carla
import numpy as np
import pygame
import weakref


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