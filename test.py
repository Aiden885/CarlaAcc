# self.error_save_file_name = "cut_in_"
        # followed_car_point = carla.Location(x=-1593.145386, y=-5239.364258, z=0.5)
        # left_point = carla.Location(x=-1602.427734, y=-5219.831543, z=0.5)
        # ego_point = carla.Location(x=-1604.839478, y=-5196.538086, z=0.5)
        # self.end_point = carla.Location(x=492.051514, y=-6452.419922, z=0.5)

        # self.error_save_file_name = "cut_out_"
        # followed_car_point = carla.Location(x=-1593.882324, y=-5234.983398, z=0.5)
        # cutout_car_point = carla.Location(x=-1599.364136, y=-5215.684082, z=0.5)
        # ego_point = carla.Location(x=-1604.839478, y=-5196.538086, z=0.5)



if self.error_save_file_name != "cut_in_" and self.error_save_file_name != "cut_out_":
            # 设置前车生成点（基于 waypoint）
            spawn_point = waypoint.transform
            lanes  = []
            current_waypoint = waypoint
            lanes.append(current_waypoint.get_left_lane())
            lanes.append(current_waypoint)
            lanes.append(current_waypoint.get_right_lane())
            self.ego_gt_lane_points = []
            self.vehicles = []
            locations = []

            #add three cars
            for lane in lanes:
                if lane is not None:
                    transform = lane.transform
                    transform.location.z += 0.1
                    locations.append(transform.location)
                    vehicle = self.world.spawn_actor(vehicle_bp, transform)
                    self.vehicles.append(vehicle)
            self.followed_car = self.vehicles[1]

            # Spawn the ego vehicle
            ego_spawn_point = self.map.get_waypoint(ego_point, project_to_road=True, lane_type=carla.LaneType.Driving).transform
            ego_spawn_point.location = ego_point
            self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, ego_spawn_point)


            # Enable autopilot for all vehicles
            self.tm = self.client.get_trafficmanager(8000)
            self.tm.set_global_distance_to_leading_vehicle(1.0)
            self.tm.set_synchronous_mode(True)
            self.tm_port = self.tm.get_port()
            
            for vehicle in self.vehicles:
                vehicle.set_autopilot(True, self.tm_port)
                # self.tm.auto_lane_change(vehicle, False)
            if not self.manual_mode:
                self.ego_vehicle.set_autopilot(True, self.tm_port)
                # self.tm.auto_lane_change(self.ego_vehicle, False)
            # tm.auto_lane_change(self.ego_vehicle, False)
        else:
            spawn_point = waypoint.transform
            lanes  = []
            current_waypoint = waypoint
            lanes.append(current_waypoint)
            lanes.append(current_waypoint.get_right_lane())
            self.ego_gt_lane_points = []
            self.vehicles = []
            locations = []

            left_spawn_point = self.map.get_waypoint(left_point, project_to_road=True, lane_type=carla.LaneType.Driving).transform
            left_spawn_point.location = left_point
            self.vehicles.append(self.world.spawn_actor(vehicle_bp, left_spawn_point))
            for lane in lanes:
                if lane is not None:
                    transform = lane.transform
                    transform.location.z += 0.1
                    locations.append(transform.location)
                    vehicle = self.world.spawn_actor(vehicle_bp, transform)
                    self.vehicles.append(vehicle)
            self.followed_car = self.vehicles[1]
            # Spawn the ego vehicle
            ego_spawn_point = self.map.get_waypoint(ego_point, project_to_road=True, lane_type=carla.LaneType.Driving).transform
            ego_spawn_point.location = ego_point
            self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, ego_spawn_point)

            # Enable autopilot for all vehicles
            self.tm = self.client.get_trafficmanager(8000)
            self.tm.set_global_distance_to_leading_vehicle(1.0)
            self.tm.set_synchronous_mode(True)
            self.tm_port = self.tm.get_port()
            
            for vehicle in self.vehicles:
                vehicle.set_autopilot(True, self.tm_port)
                # self.tm.auto_lane_change(vehicle, False)
            
            if not self.manual_mode:
                self.ego_vehicle.set_autopilot(True, self.tm_port)
                # self.tm.auto_lane_change(self.ego_vehicle, False)
            # tm.auto_lane_change(self.ego_vehicle, False)





if (self.error_save_file_name == "cut_in_" or self.error_save_file_name == "cut_out_") and \
                    self.count > 20 and whe_change_right is False:
                    self.tm.force_lane_change(self.vehicles[0], True)
                    whe_change_right = True
                # if self.count > 100  and whe_change_left is False:
                #     self.tm.force_lane_change(self.vehicles[0], True)
                #     whe_change_left = True
                self.count += 1
                time.sleep(0.01)