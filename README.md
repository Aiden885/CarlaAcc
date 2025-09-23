# CARLA ACC自适应巡航控制系统

## 项目概述

基于CARLA仿真器的ACC（Adaptive Cruise Control）自适应巡航控制系统，实现了完整的自动驾驶ACC功能，包括智能跟车、定速巡航、传感器融合和人机交互。

### 🎯 核心特性
- 🚗 **智能时距控制**: 直接控制时距（秒），更精确的跟车逻辑
- 📡 **多传感器融合**: 雷达、相机、激光雷达
- 🎮 **实时人机交互**: Pygame界面 + OpenCV视觉反馈
- 🧠 **智能决策系统**: 状态机管理 + 人工介入检测
- 🎛️ **SPPVT控制算法**: 阶段递进速度跟踪控制，符号自适应级差
- 📊 **数据记录分析**: CSV格式实时数据记录
- 🔄 **Python-Simulink混合**: 双重实现确保高可靠性

## 🆕 最新更新 (v4.0)

### **时距控制架构升级**
- ✅ **直接时间控制**: 从距离控制升级为直接时距控制（秒）
- ✅ **SPPVT符号自适应**: 基于误差符号的动态级差计算
- ✅ **误差符号检测**: 自动检测符号变化并重置控制状态
- ✅ **接口完全兼容**: 保持所有现有代码接口不变

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

#### 1. 整体执行顺序
```
用户输入 → 决策模块 → 控制模块 → Two-Mode控制器 → SPPVT → 车辆执行
```

#### 2. 详细执行流程

**阶段1: 输入处理** (`acc_updated.py`)
- 键盘输入被捕获并转换为ACC指令 (I_0 到 I_6)
- 传感器数据 (雷达、相机) 被收集和处理

**阶段2: 决策层** (`acc_decision.py`) 
- 接收ACC指令和当前系统状态
- 根据状态转移表执行决策逻辑
- **输出标准化决策** (R_1 到 R_8)，包含目标速度调整、时距参数调整、控制模式切换指令

**阶段3: 控制层** (`acc_planning_control.py`)
- 接收决策模块的输出
- **调用Two-Mode控制器**进行具体的控制计算
- 处理横向控制（车道保持）和平滑滤波

**阶段4: Two-Mode控制器** (`two_mode_controller.py`)
- **被控制层调用**，执行核心控制算法
- 根据车速自动选择控制模式：TIME模式(时距跟车)或SPEED模式(定速巡航)
- **调用SPPVT底层控制器**进行PID计算

**阶段5: SPPVT控制器** (`sppvt_longitudinal_control.py`)
- 被Two-Mode控制器调用，执行具体的PID控制算法
- 输出最终的加速度指令给车辆执行器

### 模块关系
- **入口模块**: `acc_updated.py` - 主控制器，协调所有模块
- **决策模块**: `acc_decision.py` - ACC状态机和决策逻辑
- **控制模块**: `acc_planning_control.py` - 控制指令的融合与执行
- **模式控制**: `two_mode_controller.py` - 核心纵向控制算法
- **感知模块**: `lane_detection.py` + `radar_cluster.py` + `kalman_filter.py`
- **显示模块**: `display_manager.py` - Pygame界面管理

## 📋 核心模块详解

### 1. ACC决策模块 (`acc_decision.py`) - 基于新决策文档
**功能**: ACC系统的大脑，负责状态管理和标准化决策输出
- **4种核心状态**: 在控(S0)、适速有史待命(S1)、适速无史待命(S2)、低速(S3)
- **7种驾驶指令**: 降速/启控(I0)、增速/启控(I1)、减距(I2)、增距(I3)、油门(I4)、刹车(I5)、取消(I6)
- **8种标准决策**: R1-R8标准化决策输出，涵盖速度调整、时距调整、扭矩仲裁等
- **智能切换**: 基于车速和历史数据自动在待命状态间切换
- **扭矩仲裁**: 支持驾驶员和ACC系统协调控制(取最大油门开度)

### 2. 规划控制模块 (`acc_planning_control.py`)
**功能**: 控制指令的融合与执行器
- **纵向控制**: 调用`two_mode_controller`的核心跟驰模型算法
- **横向控制**: 车道保持辅助
- **平滑控制**: 避免控制输出的突变
- **鲁棒性增强**: 融合多帧雷达数据，稳定目标检测
- **安全策略**: 实现前车重检测后的强制安全跟车

### 3. 两模式控制器 (`two_mode_controller.py`)
**功能**: ACC纵向控制的核心算法
- **时距控制模式**: 低速时基于时距gap的跟车
- **定速控制模式**: 高速时的定速巡航
- **智能切换**: 基于速度阈值V_threshold的模式选择

### 4. 传感器融合
**雷达系统** (`radar_cluster.py` + `kalman_filter.py`):
- 目标检测和跟踪
- 卡尔曼滤波器状态估计
- 多目标聚类算法

**视觉系统** (`lane_detection.py`):
- 车道线检测
- 横向位置估计
- 车道保持参考

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

2. **部署SPPVT集成模型** (v3.0新增):
   ```matlab
   % 在MATLAB中运行完整集成部署
   create_decision_sppvt_bus(); 
   load_acc_sppvt_parameters(); 
   build_acc_decision_sppvt_model();
   ```

3. **运行主程序**:
   ```bash
   python acc_updated.py
   ```

3. **操作说明** (基于新决策逻辑):
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

## ⚙️ 配置参数

### ACC关键参数 (新决策系统)
```python
# 在acc_decision.py初始化中设置
V_min_kmh = 30.0         # 最低速度要求 (低于此值进入S3低速状态)
V_target_kmh = 50.0      # 目标巡航速度
G2_s = 2.0               # 时距参数（秒）
speed_step = 1.0         # 速度调整步长 (km/h per command)
distance_step = 2.0      # 距离调整步长 (m per command)

# 决策系统状态转移表已内置，无需手动配置
# 支持28种状态转移: 4状态 × 7指令 → 8种决策
```

### CARLA环境配置
```python
# 在acc_updated.py的init_carla()中设置
world_name = 'Town05'                # 地图选择
synchronous_mode = True              # 同步模式
fixed_delta_seconds = 1.0/60.0       # 仿真步长
ego_vehicle = 'vehicle.audi.etron'   # 自车型号
target_vehicle = 'vehicle.tesla.model3'  # 目标车型号
```

## 📁 项目结构

```
CarlaAcc/
├── acc_updated.py                        # 🎯 主程序入口
├── acc_decision.py                       # 🧠 ACC决策模块  
├── acc_planning_control.py               # 🎮 运动控制模块
├── two_mode_controller.py                # 🔄 两模式控制器
├── display_manager.py                    # 📺 界面显示管理
├── lane_detection.py                     # 👁️ 车道检测
├── radar_cluster.py                      # 📡 雷达聚类
├── kalman_filter.py                      # 🎯 卡尔曼滤波
├── sinusoidal_speed_controller.py        # 🌊 前车速度控制
├── matlab_connect.py                     # 🔗 Matlab接口
├── sppvt_longitudinal_control.py         # 🎛️ SPPVT纵向控制器
│
├── 🆕 SPPVT完整集成模块 (v3.0)
├── acc_decision_sppvt_interface.py       # 🔗 Python-Simulink统一接口
├── ACC_Decision_SPPVT_Integrated.slx     # 🏗️ 完整集成Simulink模型
├── sppvt_control_model.slx               # 📈 现有SPPVT模块 (保持不变)
├── generate_sppvt_adapter_code.m         # 🔌 SPPVT接口适配器
├── build_acc_decision_sppvt_model.m      # 🏭 集成模型构建脚本
├── create_decision_sppvt_bus.m           # 📊 总线定义脚本
├── load_acc_sppvt_parameters.m           # ⚙️ 参数加载脚本
├── test_integrated_sppvt_model.m         # 🧪 集成测试脚本
├── COMPLETE_SPPVT_INTEGRATION_GUIDE.md   # 📋 完整集成使用指南
├── QUICK_START_GUIDE.md                  # 🚀 快速启动指南
└── SIMULINK_MODEL_USAGE_GUIDE.md         # 📖 Simulink模型使用指南
│
├── speed_data_integrated.csv             # 📋 实验数据
└── README.md                             # 📖 项目主文档
```

## 🆕 新决策系统特性

### 基于decision.md文档的标准化决策
- **高内聚低耦合**: 决策模块专注状态管理，控制模块专注执行
- **标准化输出**: 所有决策都有统一的R_x编号和明确定义
- **完整状态覆盖**: 28种状态转移全覆盖测试验证
- **扭矩仲裁支持**: 油门指令实现驾驶员与ACC协调控制

### 决策系统验证
运行内置测试验证系统一致性：
```bash
python acc_decision.py
```
测试覆盖：
- ✅ 状态转移测试: 28/28 通过 (100.0%)
- ✅ 决策执行测试: 4/4 通过 (100.0%)  
- ✅ 自动状态切换: 通过
- ✅ 历史数据管理: 通过

## 🔧 开发指南

### 添加新功能的流程
1. **决策层修改**: 在`acc_decision.py`中添加新状态或指令（需更新状态转移表）
2. **控制层实现**: 在`acc_planning_control.py`中实现控制逻辑
3. **界面集成**: 在`display_manager.py`中添加显示内容
4. **主程序集成**: 在`acc_updated.py`中集成新功能

### 调试模式
```python
# 启用详细调试输出
acc_decision.set_debug(True)  # 按P键切换

# 查看ACC状态信息
system_info = acc_actor.get_system_info()
```

### 数据记录
系统自动记录运行数据到`speed_data_integrated.csv`，包含：
- 时间戳、车辆速度、距离信息
- ACC状态、控制模式、参数设置
- 传感器数据、控制输出

## 🛠️ 故障排除

### 常见问题

**1. CARLA连接失败**
```bash
# 检查CARLA服务器是否运行
netstat -an | grep 2000
# 确保端口2000可用
```

**2. 传感器数据异常**  
```python
# 检查传感器回调函数
def radar_callback(self, radar_data):
    print(f"Radar points: {len(radar_data)}")
```

**3. ACC控制不响应**
```python
# 检查控制状态
print(f"ACC Active: {acc_control_active}")
print(f"Manual Mode: {manual_control_active}")
print(f"Current State: {acc_decision.current_state}")
```

**4. 性能优化**
- 降低传感器采样率
- 减少图像处理分辨率
- 优化控制算法计算频率

### 关键调试点
- ACC状态转移逻辑检查
- 传感器数据有效性验证
- 控制输出平滑性检查
- 人工介入检测灵敏度

## 📊 技术指标

### 性能参数
- **控制频率**: 20Hz (0.05s控制周期)
- **传感器更新**: 60Hz同步
- **最大跟车距离**: 50m
- **速度范围**: 20-120 km/h
- **时距参数**: 1.0-3.0s可调

### 精度指标
- **速度控制精度**: ±1 km/h
- **距离控制精度**: ±0.5m
- **车道保持精度**: ±0.1m

## 🔄 版本历史

### v3.0 (当前版本) - 2024.09.12 🎉
**重大更新: ACC决策+SPPVT完整一体化集成**
- ✅ **完整SPPVT集成**: 将现有独立SPPVT模块集成到统一Simulink架构
- ✅ **Model Reference架构**: 使用高效的Model Reference方式保持模块完整性
- ✅ **Python-Simulink一体化**: 实现决策模块与SPPVT控制的紧密集成
- ✅ **MATLAB 2024b兼容**: 修复所有兼容性问题，支持最新MATLAB版本
- ✅ **双重架构支持**: 同时支持简化版和完整版两种集成方案
- ✅ **完整调试系统**: 集成版调试码(3000+)，完整的监控和测试工具

**新增核心文件**:
- `ACC_Decision_SPPVT_Integrated.slx` - 完整集成Simulink模型
- `generate_sppvt_adapter_code.m` - SPPVT接口适配器
- `acc_decision_sppvt_interface.py` - 统一Python接口
- `COMPLETE_SPPVT_INTEGRATION_GUIDE.md` - 完整集成使用指南

**技术突破**:
- 🔧 解决MATLAB Function Script参数设置问题
- 🔗 实现Model Reference自动集成
- 📊 建立完整的输入输出适配机制
- 🐛 修复所有MATLAB 2024b兼容性问题

### v2.0 - 2024.09.09
- ✅ 三模式控制简化为二模式控制
- ✅ 降低系统计算负荷
- ✅ 优化ACC决策逻辑
- ✅ 改进人工介入检测

### v1.0 - 2024.06.20
- ✅ 基础ACC功能实现
- ✅ CARLA-Matlab通信建立
- ✅ 多传感器融合

## 🤝 开发协作

### 代码规范
- 使用Python PEP8编码规范
- 添加详细的函数和类注释
- 重要修改需要更新README

### Git提交规范
```bash
git commit -m "feat: 添加新功能"
git commit -m "fix: 修复Bug"  
git commit -m "docs: 更新文档"
git commit -m "refactor: 代码重构"
```

---

**项目维护**: 持续更新中，欢迎贡献代码和反馈问题

**最后更新**: 2024-09-12 (v3.0 SPPVT完整集成版本)
