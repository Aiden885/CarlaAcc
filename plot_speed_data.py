import carla
import math
import numpy as np
import threading
import pygame
import time
from pygame.locals import *
from acc_planning_control import ACCPlanningControl
from sinusoidal_speed_controller import SinusoidalSpeedController
from three_mode_controller import calculate_three_mode_desired_distance, set_three_mode_parameters
from acc_decision import ACCDecisionModule, ACCCommand, ACCState
from carla_camera_manager import CarlaCameraManager
from pygame_display import PygameDisplay
from data_logger import DataLogger
from sensor_manager import SensorManager


class acc:
    def __init__(self):
        # Pygame display initialization
        self.display_manager = PygameDisplay(width=1280, height=720, title="ACC Integrated Control System")

        # Sensor and detection modules
        self.tracker = kalman_filter.RadarTracker()
        self.lane_detector = lane_detection.LaneDetector()
        self.radar_point_cluster = radar_cluster.RadarClusterNode()
        self.max_follow_distance = 50
        self.radar_detections = []
        self.latest_camera_image = None
        self.radar_2_world = []
        self.world_2_camera = []
        self.cluster = []
        self.track_id = []
        self.image_width = 1280
        self.image_height = 720
        self.target_vehicle = None
        self.start_time = None
        self.target_speed_controller = None

        # ACC decision module
        self.acc_decision = ACCDecisionModule(initial_V3_kmh=50.0, initial_G1_m=15.0, initial_time_gap=2.0)
        self.acc_decision.set_debug(True)

        # Control states
        self.acc_control_active = False
        self.manual_control_active = True
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0

        # Runtime control
        self.running = True
        self.clock = pygame.time.Clock()

        # Camera manager
        self.camera_manager = None

        # Data logger
        self.data_logger = DataLogger('speed_data_integrated.csv')

        # Threading for sensor data
        self.radar_lock = threading.Lock()
        self.latest_cluster = []
        self.latest_track_id = []

        # Initialize CARLA
        self.init_carla()

        # Sensor manager (initialized after ego_vehicle and world)
        self.sensor_manager = SensorManager(
            tracker=self.tracker,
            lane_detector=self.lane_detector,
            radar_point_cluster=self.radar_point_cluster,
            ego_vehicle=self.ego_vehicle,
            world=self.world
        )

        # Sync three-mode parameters
        self._sync_three_mode_parameters()

    def init_carla(self):
        # Initialize CARLA client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        try:
            self.world = self.client.get_world()
            self.world = self.client.load_world('Town05', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load map Town05: {e}")

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 60.0
        self.world.apply_settings(settings)

        # Get blueprint library and map
        self.blueprint_library = self.world.get_blueprint_library()
        map = self.world.get_map()

        # Get vehicle blueprints
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_vehicle_bp = self.blueprint_library.filter('vehicle.audi.etron')[0]

        # Define spawn point
        fixed_point = carla.Location(x=0.663731, y=-203.651886, z=0.5)
        waypoint = map.get_waypoint(fixed_point, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            raise RuntimeError("Failed to find a valid waypoint near the specified location")

        # Spawn target vehicle
        spawn_point = waypoint.transform
        spawn_point.location.z += 0.05
        vehicles = []
        target_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if target_vehicle is None:
            raise RuntimeError("Failed to spawn target vehicle at waypoint location")
        vehicles.append(target_vehicle)
        self.target_vehicle = target_vehicle
        self.target_vehicle.set_autopilot(True)

        # Initialize sinusoidal speed controller
        self.target_speed_controller = SinusoidalSpeedController(
            vehicle=target_vehicle,
            base_speed=30,
            amplitude=5.0,
            period=10.0
        )

        # Spawn ego vehicle
        ego_spawn_point = carla.Transform()
        ego_spawn_point.location = spawn_point.location
        ego_spawn_point.location.x += 20
        ego_spawn_point.rotation = spawn_point.rotation
        self.ego_vehicle = self.world.try_spawn_actor(ego_vehicle_bp, ego_spawn_point)
        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        self.vehicles = vehicles
        self.ego_vehicle.set_autopilot(False)

        # Create camera manager
        if self.ego_vehicle:
            self.camera_manager = CarlaCameraManager(
                self.ego_vehicle,
                self.display_manager.display_width,
                self.display_manager.display_height
            )
            print("Camera manager initialized")

        # Set traffic manager
        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_synchronous_mode(False)
        self.tm_port = tm.get_port()
        tm.auto_lane_change(self.ego_vehicle, False)
        tm.ignore_vehicles_percentage(self.ego_vehicle, 100)  # Prevent TM interference

        # Target vehicle settings
        for vehicle in vehicles:
            vehicle.set_autopilot(True, self.tm_port)
            tm.auto_lane_change(vehicle, False)
            tm.vehicle_percentage_speed_difference(vehicle, 30.0)

        if self.target_speed_controller:
            self.target_speed_controller.set_traffic_manager(tm)

        # Set traffic lights
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)

        # Configure sensors
        radar_bp = self.blueprint_library.find('sensor.other.radar')
        RADAR_CONFIG = {
            'range': '100.0',
            'horizontal_fov': '120.0',
            'vertical_fov': '30.0',
            'points_per_second': '20000'
        }
        for attr, value in RADAR_CONFIG.items():
            radar_bp.set_attribute(attr, value)
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)

        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)

        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('points_per_second', '1000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-10')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)

    def _sync_three_mode_parameters(self):
        """Sync ACC decision parameters to three-mode controller"""
        acc_params = self.acc_decision.get_current_parameters()
        set_three_mode_parameters(
            V1_kmh=20,
            V2_kmh=30,
            V3_kmh=acc_params['V3_kmh'],
            G1_m=acc_params['G1_m'],
            G2_s=acc_params['G2_s']
        )

    def handle_keyboard_input(self):
        """Handle keyboard input"""
        keys = pygame.key.get_pressed()

        if self.manual_control_active:
            if keys[K_w] or keys[K_UP]:
                self.throttle = min(1.0, self.throttle + 0.02)
                if self.acc_control_active:
                    ego_speed = self.get_vehicle_speed(self.ego_vehicle)
                    self.acc_decision.process_command(ACCCommand.THROTTLE, ego_speed)
                    self.acc_control_active = False
                    print("Manual throttle engaged, ACC paused")
            else:
                self.throttle = max(0.0, self.throttle - 0.05)

            if keys[K_s] or keys[K_DOWN]:
                self.brake = min(1.0, self.brake + 0.05)
                if self.acc_control_active:
                    ego_speed = self.get_vehicle_speed(self.ego_vehicle)
                    self.acc_decision.process_command(ACCCommand.BRAKE, ego_speed)
                    self.acc_control_active = False
                    print("Manual brake engaged, ACC paused")
            else:
                self.brake = max(0.0, self.brake - 0.1)

            if keys[K_a] or keys[K_LEFT]:
                self.steer = max(-1.0, self.steer - 0.05)
            elif keys[K_d] or keys[K_RIGHT]:
                self.steer = min(1.0, self.steer + 0.05)
            else:
                self.steer = self.steer * 0.9

        hand_brake = keys[K_SPACE]

        if self.ego_vehicle and self.manual_control_active:
            control = carla.VehicleControl()
            control.throttle = self.throttle
            control.brake = self.brake
            control.steer = self.steer
            control.hand_brake = hand_brake
            self.ego_vehicle.apply_control(control)

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False

                elif event.key == K_c:
                    if self.camera_manager:
                        self.camera_manager.toggle_camera()
                        print("Toggled camera view")

                elif event.key == K_i:
                    self.display_manager.show_info = not self.display_manager.show_info
                    print(f"Info display: {'ON' if self.display_manager.show_info else 'OFF'}")

                elif event.key == K_o:
                    self.sensor_manager.show_opencv = not self.sensor_manager.show_opencv
                    print(f"OpenCV window: {'ON' if self.sensor_manager.show_opencv else 'OFF'}")

                elif event.key == K_h:
                    self.display_manager.show_help = not self.display_manager.show_help

                elif event.key == K_p:
                    debug_state = not self.acc_decision.debug
                    self.acc_decision.set_debug(debug_state)
                    print(f"ACC debug mode: {'ON' if debug_state else 'OFF'}")

                elif event.key == K_1:
                    self._process_acc_command(ACCCommand.ENGAGE)

                elif event.key == K_2:
                    self._process_acc_command(ACCCommand.EXIT)

                elif event.key == K_3:
                    self._process_acc_command(ACCCommand.CRUISE_MODE)

                elif event.key == K_q:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.INCREASE_SPEED)

                elif event.key == K_e:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.DECREASE_SPEED)

                elif event.key == K_r:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.INCREASE_DISTANCE)

                elif event.key == K_t:
                    if self.acc_control_active:
                        self._process_acc_command(ACCCommand.DECREASE_DISTANCE)

    def _process_acc_command(self, command):
        """Process ACC commands"""
        if not self.ego_vehicle:
            return

        ego_speed = self.get_vehicle_speed(self.ego_vehicle)
        target_distance = self.get_vehicle_distance(self.ego_vehicle, self.target_vehicle)
        has_target = target_distance < 50.0

        state, mode, msg = self.acc_decision.process_command(
            command, ego_speed, has_target, target_distance if has_target else None)

        acc_params = self.acc_decision.get_current_parameters()
        self.acc_control_active = acc_params['is_active']
        self.manual_control_active = not self.acc_control_active

        self._sync_three_mode_parameters()

        print(f"ACC command {command.value}: {msg}")

    def get_vehicle_speed(self, vehicle):
        """Get vehicle speed in km/h"""
        if vehicle is None:
            return 0.0
        velocity = vehicle.get_velocity()
        speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_kmh = speed_m_s * 3.6
        return speed_kmh

    def get_vehicle_distance(self, vehicle1, vehicle2):
        """Calculate distance between two vehicles in meters"""
        if vehicle1 is None or vehicle2 is None:
            return float('inf')
        loc1 = vehicle1.get_location()
        loc2 = vehicle2.get_location()
        distance = math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)
        return distance

    def get_lane_offset(self):
        """Get vehicle offset from lane center"""
        if self.world is None:
            return 0.0
        carla_map = self.world.get_map()
        if carla_map is None:
            return 0.0
        vehicle_location = self.ego_vehicle.get_location()
        waypoint = carla_map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            return 0.0
        lane_center = waypoint.transform.location
        lane_direction = waypoint.transform.get_forward_vector()
        to_center_vector = carla.Vector3D(
            lane_center.x - vehicle_location.x,
            lane_center.y - vehicle_location.y,
            0
        )
        right_direction = carla.Vector3D(
            -lane_direction.y,
            lane_direction.x,
            0
        ).make_unit_vector()
        offset = to_center_vector.dot(right_direction)
        return offset

    def calculate_desired_following_distance(self, ego_speed_kmh, time_gap=2.0, min_distance=5.0):
        """Calculate desired following distance using three-mode control"""
        ego_speed_ms = ego_speed_kmh / 3.6
        desired_distance, control_mode = calculate_three_mode_desired_distance(ego_speed_ms)
        return desired_distance, control_mode

    def generate_target(self):
        """Main loop - Integrated ACC decision, control, and display"""
        acc_controller = ACCPlanningControl(
            self.ego_vehicle,
            target_speed_kmh=30.0,
            time_gap=2.0,
            max_follow_distance=self.max_follow_distance
        )

        try:
            self.sensor_manager.get_extrinsic_params(self.radar, self.camera)
            self.start_time = time.time()
            frame_count = 0

            print("\n=== ACC Integrated Control System ===")
            print("System will display CARLA visuals and ACC info in Pygame window")
            print("\nKeyboard controls:")
            print("  1: Engage ACC  2: Exit ACC  3: Cruise mode")
            print("  Q/E: Increase/Decrease speed  R/T: Increase/Decrease distance")
            print("  W/S: Throttle/Brake  A/D: Steer")
            print("  C: Toggle camera  I: Toggle info  O: Toggle OpenCV window")
            print("  H: Help  P: Debug mode  ESC: Exit")
            print("\n")

            while self.running:
                self.handle_events()
                self.handle_keyboard_input()

                if self.target_speed_controller:
                    self.target_speed_controller.update()

                if self.world:
                    self.world.tick()

                self.clock.tick(60)

                ego_speed = self.get_vehicle_speed(self.ego_vehicle)
                target_speed = self.get_vehicle_speed(self.target_vehicle) if self.target_vehicle else 0.0
                vehicle_distance = self.get_vehicle_distance(self.ego_vehicle, self.target_vehicle)
                has_target = vehicle_distance < 50.0

                decision_output = self.acc_decision.get_decision_output(ego_speed, vehicle_distance)
                acc_params = self.acc_decision.get_current_parameters()
                acc_status = self.acc_decision.get_status_info()

                with self.radar_lock:
                    self.cluster = self.latest_cluster
                    self.track_id = self.latest_track_id

                image_with_radar = self.sensor_manager.process_camera_image(
                    self.latest_camera_image,
                    self.track_id,
                    acc_status,
                    self.acc_control_active,
                    acc_params
                )

                if self.acc_control_active and decision_output['control_enabled']:
                    try:
                        lane_center = self.sensor_manager.get_lane_center(image_with_radar)
                        # Calibrated scale factor based on camera FOV=90°
                        fx = 1280 / (2 * np.tan(90 * np.pi / 360))  # Focal length in pixels
                        scale_factor = fx / 1.0  # Assume lane is 1m away
                        lane_offset = (lane_center - 510) / scale_factor
                        print(f"Lane center: {lane_center}, Lane offset: {lane_offset}")
                        target_info = self.sensor_manager.get_target_info(self.track_id)
                        if decision_output['force_cruise_mode']:
                            control = acc_controller.cruise_control(lane_offset, None)
                        else:
                            control = acc_controller.cruise_control(lane_offset, target_info)
                        print(f"Steering command: {control.steer}")
                        if control.brake < 0.01:
                            control.brake = 0
                        self.ego_vehicle.apply_control(control)
                    except Exception as e:
                        print(f"ACC control error: {e}")

                current_time = time.time() - self.start_time
                desired_distance, control_mode = self.calculate_desired_following_distance(ego_speed)

                self.data_logger.log_data(
                    current_time,
                    ego_speed,
                    target_speed,
                    vehicle_distance,
                    desired_distance,
                    control_mode,
                    self.get_lane_offset(),
                    acc_status['state_description'],
                    self.acc_control_active,
                    acc_params['V3_kmh'],
                    acc_params['G1_m'],
                    acc_params['G2_s'],
                    acc_params.get('cruise_mode_active', False),
                    self.throttle,
                    self.brake,
                    self.steer
                )

                self.display_manager.render(
                    self.camera_manager.get_camera_image() if self.camera_manager else None,
                    ego_speed,
                    vehicle_distance,
                    has_target,
                    acc_params,
                    acc_status
                )

                frame_count += 1

        except KeyboardInterrupt:
            print("\nStopped by user.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up...")
            self.data_logger.close()
            self.display_manager.quit()
            self.destroy()

    def destroy(self):
        """Clean up resources"""
        if self.camera_manager:
            self.camera_manager.destroy()

        self.radar.stop()
        self.camera.stop()
        self.lidar.stop()

        self.radar.destroy()
        self.camera.destroy()
        self.lidar.destroy()

        for vehicle in self.vehicles:
            vehicle.destroy()
        self.ego_vehicle.destroy()

        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        print(f"Destroyed {len(self.vehicles)} vehicles, ego vehicle, sensors, and restored settings.")


def radar_callback_wrapper(sensor_manager, ego_vehicle, acc_instance):
    """Wrapper for radar callback to update shared data"""

    def callback(radar_data):
        try:
            cluster, track_id = sensor_manager.radar_callback(radar_data, ego_vehicle)
            with acc_instance.radar_lock:
                acc_instance.latest_cluster = cluster
                acc_instance.latest_track_id = track_id
        except Exception as e:
            print(f"Radar callback error: {e}")

    return callback


def main():
    """Main function to run the ACC system"""
    try:
        acc_actor = acc()
        if acc_actor.ego_vehicle is None:
            raise RuntimeError("Ego vehicle not initialized. Check init_carla().")

        # Create sensor listener threads
        thread_1 = threading.Thread(
            target=acc_actor.radar.listen,
            args=(radar_callback_wrapper(acc_actor.sensor_manager, acc_actor.ego_vehicle, acc_actor),),
            name='T1'
        )
        thread_2 = threading.Thread(
            target=acc_actor.camera.listen,
            args=(acc_actor.sensor_manager.camera_callback,),
            name='T2'
        )
        thread_3 = threading.Thread(
            target=acc_actor.lidar.listen,
            args=(acc_actor.sensor_manager.lidar_callback,),
            name='T3'
        )

        thread_1.start()
        thread_2.start()
        thread_3.start()

        acc_actor.generate_target()

    except KeyboardInterrupt:
        print("Program interrupted.")
    except Exception as e:
        print(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        acc_actor.radar.stop()
        acc_actor.camera.stop()
        acc_actor.lidar.stop()
        thread_1.join()
        thread_2.join()
        thread_3.join()
        print("All threads terminated.")


if __name__ == '__main__':
    main()