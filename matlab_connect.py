import matlab.engine
import carla
import time

# 启动 MATLAB 引擎
try:
    eng = matlab.engine.start_matlab()
    print("MATLAB 引擎启动成功")
except Exception as e:
    print("MATLAB 引擎启动失败:", e)
    exit()

# 设置 Simulink 模型路径
eng.cd(r'C:\Users\ZhaoY\Documents\MATLAB', nargout=0)

# 加载 Simulink 模型
model_name = 'test1'
try:
    eng.load_system(model_name)
    print(f"成功加载 Simulink 模型: {model_name}")
except Exception as e:
    print(f"加载 Simulink 模型失败: {e}")
    eng.quit()
    exit()

# 配置仿真参数
eng.set_param(model_name, 'SimulationMode', 'normal', nargout=0)
eng.set_param(model_name, 'StopTime', '0.1', nargout=0)

# 连接 CARLA
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print("成功连接到 CARLA 世界:", world.get_map().name)
except Exception as e:
    print("连接 CARLA 失败:", e)
    eng.quit()
    exit()

# 检查并生成车辆
actors = world.get_actors().filter('vehicle.*')
if not actors:
    print("世界中没有车辆，生成新车辆...")
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("错误：地图中没有生成点！加载 Town01...")
        client.load_world('Town01')
        spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    print("已生成车辆:", vehicle.type_id)
else:
    vehicle = actors[0]
    print("找到现有车辆:", vehicle.type_id)

# 仿真循环
for _ in range(100):
    try:
        location = vehicle.get_location()
        input_data = [location.x, location.y, location.z]
        eng.workspace['input_data'] = matlab.double(input_data)
        eng.sim(model_name)  # 运行仿真
        output = eng.eval("sim_out")  # 从 MATLAB 工作区提取 sim_out
        print("Simulink 输出:", output)
        # 取最新数据
        if output and len(output) > 0 and len(output[-1]) >= 2:
            throttle, steer = float(output[-1][0]), float(output[-1][1])
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        else:
            print("警告：Simulink 输出格式不正确:", output)
            throttle, steer = 0.0, 0.0
        time.sleep(0.1)
        world.tick()
    except Exception as e:
        print("仿真错误:", e)
        break

# 清理
vehicle.destroy()
eng.quit()
print("仿真结束")