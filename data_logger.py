import csv

class DataLogger:
    def __init__(self, filename):
        """初始化CSV文件"""
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Time(s)',
            'Ego_Speed(km/h)',
            'Target_Speed(km/h)',
            'Actual_Distance(m)',
            'Desired_Distance(m)',
            'Control_Mode',
            'Lane_Offset',
            'ACC_State',
            'ACC_Active',
            'V3_Setting',
            'G1_Setting',
            'G2_Setting',
            'Manual_Throttle',
            'Manual_Brake',
            'Manual_Steer'
        ])

    def log_data(self, current_time, ego_speed, target_speed, vehicle_distance, desired_distance,
                 control_mode, lane_offset, acc_state, acc_active, v3_kmh, g1_m, g2_s,
                 throttle, brake, steer):
        """记录数据到CSV"""
        self.csv_writer.writerow([
            current_time,
            ego_speed,
            target_speed,
            vehicle_distance,
            desired_distance,
            control_mode,
            lane_offset,
            acc_state,
            acc_active,
            v3_kmh,
            g1_m,
            g2_s,
            throttle,
            brake,
            steer
        ])
        self.csv_file.flush()

    def close(self):
        """关闭CSV文件"""
        self.csv_file.close()
