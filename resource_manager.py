"""
资源管理器
统一管理所有需要清理的资源，确保程序退出时资源正确释放
"""
from abc import ABC, abstractmethod
from typing import List, Callable
import carla


class Cleanable(ABC):
    """可清理资源的抽象接口"""

    @abstractmethod
    def cleanup(self):
        """清理资源"""
        pass


class ResourceManager:
    """
    资源管理器 - 统一管理所有需要清理的资源
    支持：
    1. CARLA actors (车辆、传感器)
    2. 文件句柄
    3. 线程/进程
    4. 其他需要清理的资源
    """

    def __init__(self):
        self._cleanup_callbacks: List[Callable] = []
        self._carla_actors: List[carla.Actor] = []
        self._cleanable_objects: List[Cleanable] = []
        self._file_handles: List = []
        self._client: carla.Client = None
        self._world: carla.World = None
        self._synchronous_mode_enabled = False

    def register_callback(self, callback: Callable):
        """注册清理回调函数"""
        self._cleanup_callbacks.append(callback)

    def register_carla_actor(self, actor: carla.Actor):
        """注册CARLA actor（车辆、传感器等）"""
        if actor and actor not in self._carla_actors:
            self._carla_actors.append(actor)

    def register_carla_actors(self, actors: List[carla.Actor]):
        """批量注册CARLA actors"""
        for actor in actors:
            self.register_carla_actor(actor)

    def register_cleanable(self, obj: Cleanable):
        """注册可清理对象"""
        if obj not in self._cleanable_objects:
            self._cleanable_objects.append(obj)

    def register_file(self, file_handle):
        """注册文件句柄"""
        if file_handle and file_handle not in self._file_handles:
            self._file_handles.append(file_handle)

    def set_carla_client(self, client: carla.Client):
        """设置CARLA客户端（用于批量销毁actors）"""
        self._client = client

    def set_carla_world(self, world: carla.World):
        """设置CARLA世界（用于恢复异步模式）"""
        self._world = world

    def set_synchronous_mode(self, enabled: bool):
        """记录同步模式状态"""
        self._synchronous_mode_enabled = enabled

    def cleanup_all(self):
        """清理所有资源（逆序清理）"""
        print("=" * 60)
        print("🧹 开始清理资源...")
        print("=" * 60)

        # 1. 执行自定义清理回调
        self._cleanup_callbacks_safely()

        # 2. 清理可清理对象
        self._cleanup_cleanable_objects()

        # 3. 批量销毁CARLA actors
        self._destroy_carla_actors()

        # 4. 恢复CARLA异步模式
        self._restore_carla_settings()

        # 5. 关闭文件句柄
        self._close_files()

        print("=" * 60)
        print("✅ 资源清理完成")
        print("=" * 60)

    def _cleanup_callbacks_safely(self):
        """安全执行所有清理回调"""
        if not self._cleanup_callbacks:
            return

        print(f"\n1️⃣ 执行清理回调 ({len(self._cleanup_callbacks)} 个)...")
        for i, callback in enumerate(reversed(self._cleanup_callbacks)):
            try:
                callback()
                print(f"   ✅ 回调 #{i + 1} 执行成功")
            except Exception as e:
                print(f"   ⚠️ 回调 #{i + 1} 执行失败: {e}")

    def _cleanup_cleanable_objects(self):
        """清理所有可清理对象"""
        if not self._cleanable_objects:
            return

        print(f"\n2️⃣ 清理对象 ({len(self._cleanable_objects)} 个)...")
        for i, obj in enumerate(reversed(self._cleanable_objects)):
            try:
                obj.cleanup()
                obj_name = obj.__class__.__name__
                print(f"   ✅ {obj_name} 清理成功")
            except Exception as e:
                print(f"   ⚠️ 对象 #{i + 1} 清理失败: {e}")

    def _destroy_carla_actors(self):
        """批量销毁CARLA actors"""
        if not self._carla_actors:
            print("\n3️⃣ 无CARLA actors需要销毁")
            return

        print(f"\n3️⃣ 销毁CARLA actors ({len(self._carla_actors)} 个)...")

        if self._client:
            # 使用批量销毁API（更高效）
            batch = []
            for actor in self._carla_actors:
                try:
                    if actor and actor.is_alive:
                        batch.append(carla.command.DestroyActor(actor))
                except Exception as e:
                    print(f"   ⚠️ 检查actor存活状态失败: {e}")

            if batch:
                try:
                    # 根据同步模式选择批量销毁方法
                    if self._synchronous_mode_enabled:
                        responses = self._client.apply_batch_sync(batch)
                    else:
                        responses = self._client.apply_batch(batch)

                    # 统计结果
                    error_count = sum(1 for r in responses if r.has_error())
                    success_count = len(responses) - error_count

                    if error_count > 0:
                        print(f"   ⚠️ {error_count} 个actors销毁失败")
                    print(f"   ✅ {success_count} 个actors销毁成功")

                except Exception as e:
                    print(f"   ❌ 批量销毁失败: {e}")
                    # 回退到逐个销毁
                    self._destroy_actors_one_by_one()
        else:
            # 没有client，逐个销毁
            self._destroy_actors_one_by_one()

    def _destroy_actors_one_by_one(self):
        """逐个销毁actors（回退方案）"""
        print("   ℹ️ 回退到逐个销毁模式...")
        success_count = 0
        for actor in self._carla_actors:
            try:
                if actor and actor.is_alive:
                    actor.destroy()
                    success_count += 1
            except Exception as e:
                print(f"   ⚠️ 销毁actor失败: {e}")
        print(f"   ✅ {success_count} 个actors销毁成功")

    def _restore_carla_settings(self):
        """恢复CARLA异步模式"""
        if not self._world:
            print("\n4️⃣ 无需恢复CARLA设置")
            return

        print("\n4️⃣ 恢复CARLA异步模式...")
        try:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self._world.apply_settings(settings)
            print("   ✅ 异步模式已恢复")
        except Exception as e:
            print(f"   ⚠️ 恢复设置失败: {e}")

    def _close_files(self):
        """关闭所有文件句柄"""
        if not self._file_handles:
            print("\n5️⃣ 无文件句柄需要关闭")
            return

        print(f"\n5️⃣ 关闭文件句柄 ({len(self._file_handles)} 个)...")
        for i, fh in enumerate(self._file_handles):
            try:
                if fh and not fh.closed:
                    fh.close()
                    print(f"   ✅ 文件 #{i + 1} 关闭成功")
            except Exception as e:
                print(f"   ⚠️ 文件 #{i + 1} 关闭失败: {e}")

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动清理"""
        self.cleanup_all()
        return False  # 不抑制异常