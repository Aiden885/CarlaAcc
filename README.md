# CARLA ACC自适应巡航控制系统

## 项目概述

基于CARLA仿真器的ACC（Adaptive Cruise Control）自适应巡航控制系统，集成了先进的决策算法、多模式控制策略和实时状态管理，实现了研究级别的自动驾驶ACC功能。

### 🎯 核心特性
- 🧠 **智能决策系统**: 基于4状态×7指令的标准化状态机（详见decision.md）
- 🎛️ **多模式控制**: Two-Mode (TIME/SPEED) + SPPVT阶段递进控制算法
- 📡 **传感器融合**: 雷达聚类 + 相机车道检测 + 卡尔曼滤波跟踪
- 🔄 **Python-Simulink混合**: 实时状态外化，保证单步调用的状态持续性
- 🎮 **完整人机界面**: Pygame实时显示 + OpenCV调试视觉 + 键盘交互
- ⚡ **实时性能**: 60Hz仿真环境，20Hz控制频率，优化执行时间 （频率可能会后续修改）

## 🏗️ 系统架构

### 混合Python-Simulink架构
```
┌─────────────────────────────────────────────────────────────────────┐
│                    CARLA仿真环境 (Town05, 60Hz)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────────┐   │
│  │   自车辆    │  │   目标车辆   │  │      传感器数据                │   │
│  │   Audi     │  │   Tesla     │  │ Camera/Radar/LiDAR (同步采集) │   │
│  └─────────────┘  └─────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │     传感器融合        │
                        │  RadarCluster +     │
                        │  LaneDetection +    │
                        │  KalmanFilter       │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   Python主控制层     │
                        │ acc_updated.py      │
                        │ (20Hz控制循环)       │
                        └─────┬────────┬──────┘
                              │        │
                    ┌─────────▼──┐  ┌──▼─────────────┐
                    │ 混合决策层  │  │   人机交互层    │
                    │ACC+SPPVT   │  │Display+Input   │
                    └─────┬──────┘  └────────────────┘
                          │
          ┌───────────────▼───────────────┐
          │      Simulink集成层            │
          │ ACC_Decision_SPPVT_Integrated │
          │  (状态外化 + 单步调用)          │
          └──────────┬─────────────────────┘
                     │
          ┌──────────▼──────────┐
          │     控制执行层        │
          │ TwoMode + SPPVT +   │
          │ PlanningControl     │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   CARLA车辆控制      │
          │ Throttle/Brake/Steer│
          └─────────────────────┘
```

### 数据流架构
```
传感器数据 → Python感知 → 决策状态管理 ↔ Simulink计算 → 控制指令 → CARLA
     ↑                                                             ↓
     └─────────── 实时监控 ← 显示系统 ← 数据记录 ←─────────────────────┘
```

### 核心模块映射
| Python模块 | 功能职责 | Simulink交互 |
|-----------|---------|-------------|
| `acc_updated.py` | 系统主循环、传感器管理 | 调用集成模型 |
| `acc_decision_sppvt_interface.py` | Simulink接口层 | 14/15字段总线 |
| `realtime_sppvt_state_manager.py` | 状态外化管理 | 持久状态维护 |
| `two_mode_controller.py` | TIME/SPEED模式切换 | 控制策略选择 |
| `acc_planning_control.py` | 控制指令融合 | 纵向/横向控制 |
| `display_manager.py` | 实时显示界面 | 系统状态监控 |

## 📋 核心模块详解

### 🎛️ 控制算法层
- **ACC决策模块** (`acc_decision.py`): 4状态×7指令的完整状态机，详见decision.md
- **两模式控制器** (`two_mode_controller.py`): TIME模式(≤50km/h)和SPEED模式(>50km/h)自动切换
- **SPPVT控制器** (`sppvt_longitudinal_control.py`): 阶段递进速度跟踪，符号变化检测和级差动态计算
- **规划控制模块** (`acc_planning_control.py`): 纵向控制指令融合，横向车道保持，目标重检测安全协议

### 🔄 Simulink集成层
- **决策SPPVT接口** (`acc_decision_sppvt_interface.py`): Python-Simulink总线数据桥接
- **实时状态管理器** (`realtime_sppvt_state_manager.py`): 状态外化，解决单步调用状态丢失问题
- **Simulink模型**: `ACC_Decision_SPPVT_Integrated.slx` - 17模块集成决策+SPPVT计算

### 📡 感知融合层
- **雷达处理** (`radar_cluster.py`): DBSCAN聚类，ego motion补偿，多目标跟踪
- **相机视觉** (`lane_detection.py`): HSL车道检测，IPM变换，Bezier曲线拟合
- **状态估计** (`kalman_filter.py`): 多目标卡尔曼滤波，航迹管理
- **传感器管理** (`sensor_manager.py`): 传感器数据同步，坐标系转换

### 🎮 交互显示层
- **显示管理器** (`display_manager.py`): Pygame实时3D视图，信息面板，模拟仪表
- **数据记录** (`data_logger.py`): 16字段CSV数据记录，性能监控
- **系统主循环** (`acc_updated.py`): 60Hz CARLA同步，20Hz控制循环，键盘交互处理

## 🚀 快速开始

### 环境要求
- **Python 3.7+**
- **CARLA Simulator 0.9.x** (推荐0.9.13+)
- **MATLAB R2020b+** 含Simulink (用于混合控制算法)
- **Python依赖包**：
  ```bash
  pip install carla numpy opencv-python pygame matplotlib
  pip install matlab.engine  # MATLAB Python引擎接口
  pip install scikit-learn   # DBSCAN聚类算法
  ```

### Simulink模型准备
1. **加载总线定义**：
   ```matlab
   load('DecisionSPPVTBusDefinitions.mat');
   ```
2. **确认模型文件**：`ACC_Decision_SPPVT_Integrated.slx`

### 运行步骤
1. **启动CARLA服务器**:
   ```bash
   ./CarlaUE4.sh -quality-level=Low -world-port=2000
   ```

2. **启动MATLAB引擎** (可选，如不启动将使用Python fallback):
   ```matlab
   matlab.engine.shareEngine  % 共享MATLAB引擎
   ```

3. **运行主程序**:
   ```bash
   python acc_updated.py
   ```

3. **操作说明**:
   - `空格键`: ACC系统总开关
   - `Q`: I1 增速/继承启控 (有史待命下恢复历史设置，控制中时增速)
   - `E`: I0 降速/当速启控 (待命状态下以当前车速启动ACC，控制中时降速)
   - `R`: I3 增距 (增加跟车时距)
   - `T`: I2 减距 (减少跟车时距)
   - `C`: I6 取消ACC (进入待命状态)
   - `W`: I4 油门 (触发扭矩仲裁，人驾优先)
   - `S`: I5 刹车 (进入待命状态，人驾优先)
   - `A/D`: 向左/向右转向辅助
   - `方向键←→`: 转向控制
   - `O`: 切换OpenCV显示
   - `P`: 切换ACC调试信息显示
   - `ESC`: 退出程序

## ⚙️ 主要参数

- **目标巡航速度**: 50 km/h
- **时距参数**: 2.0秒
- **最低启控速度**: 30 km/h
- **默认地图**: Town05
- **控制频率**: 20Hz

## 📁 项目结构

```
CarlaAcc/
├── 🎯 核心控制模块
│   ├── acc_updated.py                    # 🚀 主程序入口 - 系统主循环(60Hz CARLA + 20Hz控制)
│   ├── acc_decision.py                   # 🧠 ACC决策模块 - 4状态×7指令状态机
│   ├── acc_planning_control.py           # 🎛️ 运动控制模块 - 纵向/横向控制融合
│   └── two_mode_controller.py            # ⚡ 两模式控制器 - TIME/SPEED动态切换
│
├── 🔄 Simulink集成层
│   ├── acc_decision_sppvt_interface.py   # 🔗 Python-Simulink接口桥接
│   ├── realtime_sppvt_state_manager.py   # 📊 实时状态外化管理器
│   ├── sppvt_longitudinal_control.py     # 📈 SPPVT控制算法(Python实现)
│   ├── ACC_Decision_SPPVT_Integrated.slx # 🔧 Simulink集成模型(17模块)
│   ├── sppvt_control_model.slx          # ⚙️ SPPVT控制子模型
│   └── DecisionSPPVTBusDefinitions.mat   # 📋 总线数据结构定义
│
├── 📡 传感器融合模块
│   ├── radar_cluster.py                  # 📡 雷达数据处理 - DBSCAN聚类
│   ├── lane_detection.py                 # 📷 车道检测 - HSL+IPM+Bezier拟合
│   ├── kalman_filter.py                  # 🎯 卡尔曼滤波 - 多目标状态估计
│   ├── sensor_manager.py                 # 🎛️ 传感器数据同步管理
│   └── carla_camera_manager.py           # 📸 CARLA相机接口封装
│
├── 🎮 人机交互界面
│   ├── display_manager.py                # 🖥️ Pygame实时显示 - 3D视图+仪表盘
│   ├── pygame_display.py                 # 🎨 游戏显示组件
│   ├── data_logger.py                    # 📝 CSV数据记录器(16字段)
│   └── get_cur_location.py               # 📍 位置信息获取工具
│
├── 🧪 测试验证模块
│   ├── test_two_mode_control.py          # 🧪 两模式控制器测试
│   ├── test_torque_arbitration.py        # ⚡ 扭矩仲裁逻辑测试
│   ├── test_realtime_integration.py      # 🔄 实时集成测试
│   ├── simple_test.py                    # 🔬 简单功能测试
│   ├── simple_analyze.py                 # 📊 性能分析工具
│   └── analyze_test_data.py              # 📈 测试数据分析器
│
├── 📚 文档和配置
│   ├── README.md                         # 📖 项目主文档
│   ├── decision.md                       # 🧠 决策逻辑详细说明
│   ├── REALTIME_SIMULINK_INTEGRATION.md  # 🔄 Simulink集成技术文档
│   ├── SPPVT_OUTPUT_ACQUISITION_GUIDE.md # 📋 SPPVT输出获取指南
│   └── speed_data_integrated.csv         # 📊 运行数据记录文件
│
└── 🛠️ MATLAB支持文件
    ├── build_acc_decision_sppvt_model.m   # 🔧 Simulink模型构建脚本
    ├── create_decision_sppvt_bus.m        # 📋 总线定义创建脚本
    ├── test_integrated_sppvt_model.m      # 🧪 集成模型测试脚本
    ├── comprehensive_state_transition_test.m # 🔍 状态转移完整性测试
    └── load_acc_sppvt_parameters.m        # ⚙️ 参数加载脚本
```

## 🔧 调试与开发

### 🧪 功能测试
```bash
# 决策系统独立测试
python acc_decision.py

# 两模式控制器测试
python test_two_mode_control.py

# 实时Simulink集成测试
python test_realtime_integration.py



### 🛠️ 调试模式
- **P键**: 切换ACC调试信息显示
- **O键**: 切换OpenCV传感器可视化
- **H键**: 显示帮助信息
- **I键**: 显示系统状态信息

### 📊 数据监控
- **实时数据**: 运行数据自动保存到`speed_data_integrated.csv` (16字段)
- **性能监控**: Simulink调用时间，控制循环频率
- **状态跟踪**: SPPVT阶段升级，决策状态转移

### 🔍 Simulink调试
```matlab
% 加载测试环境
load('DecisionSPPVTBusDefinitions.mat');

% 运行集成模型测试
test_integrated_sppvt_model;

% 验证状态转移逻辑
comprehensive_state_transition_test;
```

## 🛠️ 常见问题

### 🚨 连接问题
1. **CARLA连接失败**: 确保CARLA服务器运行在端口2000
2. **MATLAB引擎连接失败**: 检查matlab.engine安装，使用Python fallback模式
3. **传感器数据异常**: 检查CARLA版本兼容性，确认Town05地图加载

### ⚙️ 性能问题
1. **帧率下降**: 降低传感器采样率，关闭OpenCV显示窗口
2. **Simulink调用延迟**: 检查MATLAB许可证，优化总线数据大小
3. **内存占用过高**: 定期清理卡尔曼滤波器历史数据

### 🎛️ 控制问题
1. **ACC不响应**: 检查车速是否超过最低启控速度(30km/h)
2. **SPPVT阶段不升级**: 确认控制误差超过eta阈值(0.2)
3. **目标检测失效**: 验证雷达聚类参数，检查目标车辆位置

## 📈 技术特色

### 🔬 研究价值
- **混合架构**: Python-Simulink协同，兼顾灵活性和精确性
- **状态外化**: 解决实时系统中Simulink状态持续性问题
- **多模式控制**: TIME/SPEED动态切换，适应不同驾驶场景
- **符号变化检测**: SPPVT算法的重要安全特性

### 🏭 工程价值
- **模块化设计**: 清晰的层次结构，便于维护和扩展
- **错误处理**: 多层次fallback机制，提高系统可靠性
- **实时性能**: 优化的控制循环，满足实时控制要求
- **标准化接口**: 标准总线数据格式，支持硬件在环测试

---

**项目维护**: 持续更新中，欢迎贡献代码和反馈问题

**版本历史**:
- v3.0 (2024-09-26): 完整Simulink集成，状态外化架构，实时SPPVT控制
- v2.0 (2024-09-12): SPPVT算法集成，决策系统完善
- v1.0 (2024-05-29): 基础ACC功能，传感器融合框架

**最后更新**: 2024-09-26 (v3.0 混合架构完整版)
