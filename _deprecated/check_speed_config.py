"""
检查前车速度配置
诊断为什么前车跑得太快
"""
from acc_config import ACCConfig

config = ACCConfig()

print("="*70)
print("前车速度配置诊断")
print("="*70)

print(f"\n当前配置:")
print(f"  假设道路限速: {config.assumed_road_speed_limit_kmh:.1f} km/h")
print(f"  前车目标速度: {config.target_speed_kmh:.1f} km/h")
print(f"  CARLA时间步长: {config.fixed_delta_seconds:.3f} 秒/帧")

# 计算Traffic Manager的百分比差值
assumed_limit = config.assumed_road_speed_limit_kmh
target_speed = config.target_speed_kmh

percentage_diff = ((assumed_limit - target_speed) / assumed_limit) * 100.0

print(f"\nTraffic Manager计算:")
print(f"  百分比差值: {percentage_diff:.1f}%")

if percentage_diff > 0:
    print(f"  → 前车会比限速慢{percentage_diff:.1f}%")
    actual_speed = assumed_limit * (1 - percentage_diff/100)
elif percentage_diff < 0:
    print(f"  → 前车会比限速快{abs(percentage_diff):.1f}%")
    actual_speed = assumed_limit * (1 + abs(percentage_diff)/100)
else:
    print(f"  → 前车按限速行驶")
    actual_speed = assumed_limit

print(f"  预期实际速度: {actual_speed:.1f} km/h")

# 诊断
print(f"\n" + "="*70)
print("诊断结果:")
print("="*70)

if actual_speed > 60:
    print(f"❌ 前车速度过快！({actual_speed:.1f} km/h)")
    print(f"\n可能原因:")

    if target_speed > assumed_limit * 1.5:
        print(f"  1. target_speed_kmh ({target_speed:.1f}) 远大于 assumed_road_speed_limit_kmh ({assumed_limit:.1f})")
        print(f"     建议: 将target_speed_kmh改为合理值（如30-50 km/h）")

    if assumed_limit < 20:
        print(f"  2. assumed_road_speed_limit_kmh设置太低 ({assumed_limit:.1f} km/h)")
        print(f"     建议: 根据实际道路限速调整（Town03通常是30-50 km/h）")

    print(f"\n推荐配置:")
    print(f"  # 如果希望前车速度为40 km/h:")
    print(f"  self.assumed_road_speed_limit_kmh = 30.0  # 假设限速30")
    print(f"  self.target_speed_kmh = 40.0              # 目标40 (超速33%)")
    print(f"  ")
    print(f"  # 或者更合理的配置:")
    print(f"  self.assumed_road_speed_limit_kmh = 50.0  # 假设限速50")
    print(f"  self.target_speed_kmh = 40.0              # 目标40 (比限速慢20%)")

elif actual_speed < 20:
    print(f"❌ 前车速度过慢！({actual_speed:.1f} km/h)")
    print(f"\n建议增加 target_speed_kmh 或减小 assumed_road_speed_limit_kmh")

else:
    print(f"✅ 前车速度合理 ({actual_speed:.1f} km/h)")

# 检查时间步长影响
print(f"\n" + "="*70)
print("时间步长影响分析:")
print("="*70)
print(f"当前设置: {config.fixed_delta_seconds:.3f} 秒/帧 ({1/config.fixed_delta_seconds:.1f} FPS)")

if config.fixed_delta_seconds != 0.05:
    print(f"⚠️  注意: 你修改了时间步长从0.05改为{config.fixed_delta_seconds}")
    print(f"   这不会直接影响车辆速度（Traffic Manager会自动适应）")
    print(f"   但会影响仿真的平滑度和响应速度")

    if config.fixed_delta_seconds > 0.05:
        print(f"   → 更大的步长 = 更低的FPS = 画面更卡顿")
    else:
        print(f"   → 更小的步长 = 更高的FPS = 画面更流畅但计算量更大")

print(f"\n建议:")
print(f"  - ACC系统使用0.05s步长（20 FPS）是合理的")
print(f"  - 如果觉得前车走太快，调整target_speed_kmh，不要调时间步长")
