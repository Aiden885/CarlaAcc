# CARLA ACC 集成控制系统 (CARLA ACC Integrated Control System)

## 项目简介
本项目是一个基于 CARLA 仿真器的自适应巡航控制 (ACC) 系统开发与验证框架。它采用 **Python + Simulink** 联合仿真的架构：
- **Python 端**：负责 CARLA 环境交互、车辆动力学接口、横向控制、场景构建及可视化。
- **Simulink 端**：负责 ACC 核心决策逻辑 (Decision) 和纵向控制算法 (SPPVT)。

两者通过 **UDP 协议** 进行实时闭环通信，实现了从感知、决策、控制到执行的完整链路。

## 核心特性
1.  **集成化架构**：采用单一 UDP 链路整合决策与控制模型 (`acc_integrated_model.slx`)，降低通信延迟与同步复杂度。
2.  **双模式控制**：支持 **定速巡航 (Speed Mode)** 和 **定距跟随 (Time Gap Mode)** 自动切换。
3.  **扭矩仲裁机制**：支持驾驶员介入（踩油门/刹车）优先于 ACC 控制，松开后自动恢复或待命。
4.  **物理级执行器**：内置 `TorqueToThrottleConverter`，基于车辆物理参数（扭矩曲线、阻力）将目标加速度/扭矩精确转换为油门/刹车信号。
5.  **增强横向控制**：基于预瞄 (Lookahead) 的 PID 横向控制器，适应弯道与换道场景。
6.  **测试场景生成**：支持配置 **Cut-in (切入)**、**Cut-out (切出)** 及 **Ramp Speed (变跟随目标)** 等典型测试工况。
7.  **实时可视化**：提供 Pygame 驾驶界面 (HUD) 和实时数据曲线绘图。

## 环境依赖
- **操作系统**: Linux (推荐) / Windows
- **Python**: 3.8+
- **CARLA**: 0.9.x (服务端)
- **MATLAB/Simulink**: 2020b+ (需安装 Simulink Coder, DSP System Toolbox 等)
- **Python 库**:
    ```bash
    pip install carla numpy pygame matplotlib
    ```

## 快速开始

### 1. 启动 CARLA 服务器
在终端中运行 CARLA：
```bash
./CarlaUE4.sh -quality-level=Low -world-port=2000
```

### 2. 启动 Simulink 模型
1.  打开 MATLAB，将工作目录切换到项目根目录。
2.  打开 `acc_integrated_model.slx`。
3.  **运行模型** (点击 Run 按钮)。
    *   *注意：模型需处于运行状态才能接收 Python 发送的 UDP 数据。*
    *   *UDP 配置：接收端口 27000，发送端口 27001。*

### 3. 运行 Python 主程序
在项目目录下运行：
```bash
python acc_updated.py
```

## 操作说明

### 键盘控制
| 按键 | 功能 | 说明 |
| :--- | :--- | :--- |
| **Space** | **ACC 开/关** | 激活或关闭 ACC 系统 |
| **W** | 油门 | 驾驶员加速 (覆盖 ACC，显示 "Torque Arbitration") |
| **S** | 刹车 | 驾驶员减速 (退出 ACC 控制，进入待命状态) |
| **A / D** | 转向 | 手动接管方向盘 |
| **Q / E** | 调整车速 | 增加/减少 目标巡航速度 |
| **R / T** | 调整时距 | 增加/减少 跟车时距 (Time Gap) |
| **C** | 取消控制 | 退出 ACC (Standby) |
| **Z / X** | 强制换道 | 指令前车向左/向右换道 (测试用) |
| **F** | 斜坡变速 | 触发前车速度线性变化场景 |
| **P** | 调试信息 | 切换终端调试信息输出 |
| **ESC** | 退出 | 关闭程序 |

### 界面显示 (HUD)
- **System State**: 显示当前 ACC 状态 (Control, Standby, Override 等)。
- **Speed**: 当前车速及目标车速。
- **Distance**: 与前车距离及目标跟随距离。
- **Throttle/Brake**: 当前油门与刹车开度。
- **Lead Vehicle**: 前车识别状态 (ID, 距离)。

## 项目配置 (`acc_config.py`)
所有核心参数均可在 `acc_config.py` 中调整，无需修改代码：

- **仿真设置**:
    - `synchronous_mode`: 是否开启同步模式 (推荐 True)。
    - `fixed_delta_seconds`: 仿真步长 (默认 0.05s, 即 20FPS)。
    - `enable_realtime`: 是否限制真实时间运行。

- **ACC 参数**:
    - `V_target_kmh`: 默认巡航速度。
    - `G2_s`: 默认跟车时距。
    - `sppvt_params`: 纵向控制器 PID 及物理参数。

- **场景配置**:
    - `enable_cut_in_scenario`: 启用切入场景。
    - `cut_in_trigger_time_s`: 切入触发时间。
    - `ego_vehicle_blueprint`: 自车车型 (默认 Audi e-tron)。

## 文件结构说明

| 文件名 | 描述 |
| :--- | :--- |
| `acc_updated.py` | **主程序入口**。负责 CARLA 循环、传感器回调、控制执行与 UI 更新。 |
| `acc_config.py` | **配置文件**。包含车辆、控制、场景、通信等所有参数。 |
| `integrated_simulink_manager.py` | **通信管理**。负责 Python 与 Simulink 之间的 UDP 数据封包与解析。 |
| `acc_integrated_model.slx` | **Simulink 模型**。包含状态机决策与 SPPVT 纵向控制算法。 |
| `two_mode_controller.py` | **双模式逻辑**。计算速度误差与距离误差，判断控制模式。 |
| `torque_to_throttle_converter.py` | **执行器转换**。将目标扭矩/加速度转换为油门/刹车信号。 |
| `enhanced_lateral_controller.py` | **横向控制**。实现车道保持与换道辅助。 |
| `cut_in_scenario.py` | **切入场景**。管理旁车切入的逻辑与状态。 |

## 常见问题 (FAQ)

**Q: 运行 `acc_updated.py` 后没反应或报错 UDP 连接失败？**
A: 请确保 Simulink 模型 `acc_integrated_model.slx` 正在运行。Python 需要等待 Simulink 的 UDP 回传才能开始控制循环。

**Q: 车辆行驶不稳定或画龙？**
A: 检查 `acc_config.py` 中的 `fixed_delta_seconds` 是否与 Simulink 的步长一致（默认 0.05s）。帧率过低会导致 PID 控制器震荡。

**Q: 如何修改目标车速？**
A: 可以在运行时按 `Q`/`E` 键，或在 `acc_config.py` 中修改 `V_target_kmh`。

---
*Copyright © 2025 CarlaAcc Project*
