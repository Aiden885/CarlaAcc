#!/usr/bin/env python3
"""
加载配置中的地图，并将Spectator定位到前车生成点。
"""
import carla
from acc_config import ACCConfig


def main():
    cfg = ACCConfig()

    # 目标坐标：使用配置中的前车生成点（切入工况前车位置）
    loc = cfg.cut_in_target_spawn_location
    rot = carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0)

    client = carla.Client(cfg.carla_host, cfg.carla_port)
    client.set_timeout(cfg.carla_timeout)

    print(f"Loading map: {cfg.map_name}")
    world = client.load_world(cfg.map_name)

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(loc, rot))
    print(f"Spectator set to {loc} on map {cfg.map_name}")


if __name__ == "__main__":
    main()
