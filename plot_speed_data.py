import csv
import matplotlib.pyplot as plt

# Read data from CSV
times = []
ego_speeds = []
target_speeds = []
target_desired_speeds = []
distances = []

try:
    with open('speed_data.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            times.append(float(row[0]))
            ego_speeds.append(float(row[1]))
            target_speeds.append(float(row[2]))
            target_desired_speeds.append(float(row[3]))
            distances.append(float(row[4]))
except FileNotFoundError:
    print("Error: 'speed_data.csv' not found. Please ensure the file exists.")
    exit(1)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# Create a figure with two subplots
plt.figure(figsize=(15, 10))

# First subplot - Speed vs Time
plt.subplot(2, 1, 1)
plt.plot(times, ego_speeds, label='Ego Vehicle Speed (km/h)', color='green', linewidth=2)
plt.plot(times, target_speeds, label='Target Vehicle Speed (km/h)', color='red', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Speed (km/h)')
plt.title('Vehicle Speeds Over Time')
plt.legend()
plt.grid(True)

# Second subplot - Distance vs Time
plt.subplot(2, 1, 2)
plt.plot(times, distances, label='Inter-vehicle Distance (m)', color='blue', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Distance Between Vehicles Over Time')
plt.legend()
plt.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('vehicle_data_plot.png', dpi=300)
plt.show()
print("Plot saved as 'vehicle_data_plot.png'")

# Optionally create a separate higher-resolution plot for each
# Speed plot
plt.figure(figsize=(12, 6))
plt.plot(times, ego_speeds, label='Ego Vehicle Speed (km/h)', color='green', linewidth=2)
plt.plot(times, target_speeds, label='Target Vehicle Speed (km/h)', color='red', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Speed (km/h)')
plt.title('Vehicle Speeds Over Time')
plt.legend()
plt.grid(True)
plt.savefig('speed_plot.png', dpi=300)
plt.close()

# Distance plot
plt.figure(figsize=(12, 6))
plt.plot(times, distances, label='Inter-vehicle Distance (m)', color='blue', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Distance Between Vehicles Over Time')
plt.legend()
plt.grid(True)
plt.savefig('distance_plot.png', dpi=300)
plt.close()

print("Additional individual plots saved as 'speed_plot.png' and 'distance_plot.png'")