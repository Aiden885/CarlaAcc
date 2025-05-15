# 文件名: plot_vehicle_data.py
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
data = pd.read_csv('target_vehicle_data.csv')

# 提取列
time = data['Time (s)']
speed = data['Speed (km/h)']
is_curved = data['Is Curved']
pitch = data['Pitch (deg)']

# 创建图形
plt.figure(figsize=(12, 8))

# 速度曲线
plt.subplot(2, 1, 1)
plt.plot(time, speed, label='Speed (km/h)', color='blue')
plt.fill_between(time, 0, 100, where=is_curved, color='gray', alpha=0.3, label='Curved Road')
plt.title('Target Vehicle Speed Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed (km/h)')
plt.legend()
plt.grid(True)

# 坡度曲线
plt.subplot(2, 1, 2)
plt.plot(time, pitch, label='Pitch (deg)', color='green')
plt.title('Road Pitch Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Pitch (deg)')
plt.legend()
plt.grid(True)

# 调整布局并保存
plt.tight_layout()
plt.savefig('vehicle_behavior.png')
print("图像已保存到 vehicle_behavior.png")