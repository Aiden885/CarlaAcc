# CARLA ACC自适应巡航控制系统

## 项目概述

基于CARLA仿真器的ACC（Adaptive Cruise Control）自适应巡航控制系统，实现了完整的自动驾驶ACC功能。

### 🎯 核心特性
- 🚗 **智能时距控制**: 直接控制时距（秒），更精确的跟车逻辑
- 📡 **多传感器融合**: 雷达、相机、激光雷达
- 🎮 **实时人机交互**: Pygame界面 + OpenCV视觉反馈
- 🧠 **智能决策系统**: 基于状态机的标准化决策（详见decision.md）
- 🎛️ **SPPVT控制算法**: 阶段递进速度跟踪控制，符号自适应级差
- 🔄 **Python-Simulink混合**: 双重实现确保高可靠性

## 🏗️ 系统架构

### 核心架构图
```
┌─────────────────────────────────────────────────────────────┐
│                    CARLA仿真环境                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │   自车辆     │  │   目标车辆    │  │   环境对象        │    │
│  │   (Ego)     │  │  (Target)    │  │  (Road/Traffic)  │    │
│  └──────────────┘  └──────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │      传感器层        │
                    │  Radar│Camera│LiDAR │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │      感知层         │
                    │ 目标检测│车道检测    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │      决策层         │
                    │   ACC状态管理       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │      控制层         │
                    │  纵向控制│横向控制   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │      交互层         │
                    │ 显示界面│键盘控制    │
                    └─────────────────────┘
```

### 系统执行流程
```
用户输入 → 决策模块 → 控制模块 → Two-Mode控制器 → SPPVT → 车辆执行
```

### 模块关系
- **入口模块**: `acc_updated.py` - 主控制器，协调所有模块
- **决策模块**: `acc_decision.py` - ACC状态机和决策逻辑
- **控制模块**: `acc_planning_control.py` - 控制指令的融合与执行
- **模式控制**: `two_mode_controller.py` - 核心纵向控制算法
- **感知模块**: `lane_detection.py` + `radar_cluster.py` + `kalman_filter.py`
- **显示模块**: `display_manager.py` - Pygame界面管理

## 📋 核心模块

- **ACC决策模块** (`acc_decision.py`): 状态机管理，详见decision.md
- **规划控制模块** (`acc_planning_control.py`): 控制指令融合与执行
- **两模式控制器** (`two_mode_controller.py`): TIME/SPEED模式切换
- **SPPVT控制器** (`sppvt_longitudinal_control.py`): 阶段递进PID控制
- **传感器融合**: 雷达+相机+激光雷达多传感器融合
- **人机界面** (`display_manager.py`): Pygame实时显示

## 🚀 快速开始

### 环境要求
- Python 3.7+
- CARLA Simulator 0.9.x
- 相关Python依赖包：
  ```bash
  pip install carla numpy opencv-python pygame matplotlib
  ```

### 运行步骤
1. **启动CARLA服务器**:
   ```bash
   ./CarlaUE4.sh -quality-level=Low
   ```

2. **运行主程序**:
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
├── acc_updated.py                        # 主程序入口
├── acc_decision.py                       # ACC决策模块
├── acc_planning_control.py               # 运动控制模块
├── two_mode_controller.py                # 两模式控制器
├── sppvt_longitudinal_control.py         # SPPVT纵向控制器
├── display_manager.py                    # 界面显示管理
├── lane_detection.py                     # 车道检测
├── radar_cluster.py                      # 雷达聚类
├── kalman_filter.py                      # 卡尔曼滤波
├── decision.md                           # 决策逻辑说明文档
└── README.md                             # 项目主文档
```

## 🔧 调试与开发

### 测试决策系统
```bash
python acc_decision.py
```

### 调试模式
按P键切换ACC调试信息显示

### 数据记录
运行数据自动保存到`speed_data_integrated.csv`

## 🛠️ 常见问题

1. **CARLA连接失败**: 确保CARLA服务器运行在端口2000
2. **ACC不响应**: 检查控制状态，按P键查看调试信息
3. **性能问题**: 降低传感器采样率，优化显示分辨率

---

**项目维护**: 持续更新中，欢迎贡献代码和反馈问题

**最后更新**: 2024-09-12 (v3.0 SPPVT完整集成版本)
