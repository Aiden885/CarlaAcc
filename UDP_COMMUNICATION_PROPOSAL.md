# Python-Simulink UDP通信优化方案

**项目名称：** CarlaAcc 自适应巡航控制系统
**方案版本：** v2.0
**创建日期：** 2025-12-03
**方案作者：** Claude Code
**审核状态：** 🟡 待审核

---

## 📋 目录

1. [当前架构分析](#1-当前架构分析)
2. [问题诊断](#2-问题诊断)
3. [技术方案设计](#3-技术方案设计)
4. [详细实施步骤](#4-详细实施步骤)
5. [文件修改清单](#5-文件修改清单)
6. [附录](#6-附录)

---

## 1. 当前架构分析

### 1.1 系统概览

```
┌────────────────────────────────────────────────────────┐
│             CARLA仿真环境 (20 FPS)                      │
│         自车 + 前车 + 环境感知                           │
└────────────────┬───────────────────────────────────────┘
                 │ 每帧 (50ms)
                 ▼
┌────────────────────────────────────────────────────────┐
│          Python主程序 (acc_updated.py)                  │
│  - 读取车辆状态                                          │
│  - 计算控制误差                                          │
│  - 调用ACC决策+控制                                      │
│  - 应用车辆控制指令                                      │
└────────────┬───────────────┬───────────────────────────┘
             │               │
             │ MATLAB Engine │ MATLAB Engine
             │ (每帧调用)     │ (每帧调用)
             ▼               ▼
┌────────────────────┐  ┌─────────────────────────┐
│  决策模型 (Simulink)│  │  控制模型 (Simulink)     │
│  acc_decision_core │  │  sppvt_control_model    │
│                    │  │                         │
│  状态机查找表       │  │  SPPVT三阶段控制器       │
│  输入: 4个         │  │  输入: 6个              │
│  输出: 5个         │  │  输出: 5个              │
└────────────────────┘  └─────────────────────────┘
```

### 1.2 当前数据流

**主循环（每帧50ms）：**

```python
while running:
    # 1. 获取车辆状态 (1-3ms)
    ego_speed, distance, error = get_vehicle_state()

    # 2. 调用决策模型 (40-60ms) ← 瓶颈
    decision_output = matlab_engine.sim('acc_decision_core', inputs)

    # 3. 调用控制模型 (40-60ms) ← 瓶颈
    control_output = matlab_engine.sim('sppvt_control_model', inputs)

    # 4. 应用控制指令 (1-2ms)
    apply_control(throttle, brake, steer)

    # 总耗时: 80-120ms/帧 (8-12 FPS)
```

### 1.3 两个Simulink模型详细接口

#### **模型1: acc_decision_core.slx**

| 端口 | 名称 | 类型 | 范围 | 说明 |
|------|------|------|------|------|
| **输入** |
| In1 | current_state | int | 0-3 | 当前状态 (S0-S3) |
| In2 | command_type | int | 0-7 | 键盘指令 (I0-I6) |
| In3 | has_history | bool | 0/1 | 是否有历史记录 |
| In4 | last_active_decision | int | 1-8 | 上次决策 (R1-R8) |
| **输出** |
| Out1 | next_state | int | 0-3 | 下一状态 |
| Out2 | decision | int | 1-8 | 当前决策 |
| Out3 | control_enabled | bool | 0/1 | 控制使能 |
| Out4 | next_has_history | bool | 0/1 | 更新的历史标志 |
| Out5 | next_last_decision | int | 1-8 | 更新的决策记录 |

#### **模型2: sppvt_control_model.slx**

| 端口 | 名称 | 类型 | 单位 | 说明 |
|------|------|------|------|------|
| **输入** |
| In1 | error_value | double | m | 控制误差（距离或速度差） |
| In2 | current_stage_offset | double | - | 级差状态累积值 |
| In3 | prev_error | double | m | 上一帧误差 |
| In4 | prev_velocity | double | m/s | 上一帧误差导数 |
| In5 | prev_accel | double | m/s² | 上一帧误差二阶导 |
| **输出** |
| Out1 | control_output | double | - | SPPVT原始输出 |
| Out2 | velocity_output | double | m/s | 速度输出 |
| Out3 | acceleration_output | double | m/s² | 加速度输出 |
| Out4 | jerk_output | double | m/s³ | 加加速度输出 |
| Out5 | should_upgrade | double | 0/1 | 升级标志（阶段切换条件） |

**⚠️ 注意：** control_mode_flag已从输入中删除，控制模式由Simulink内部逻辑决定。

---

## 2. 问题诊断

### 2.1 性能瓶颈

**根据性能分析报告：**

```
0_simulink          : 平均  50.23ms | 最大  85.12ms | 最小  38.45ms
1_events            : 平均   2.15ms
3_world_tick        : 平均   8.34ms
4_display_tick      : 平均   3.67ms
─────────────────────────────────────────────────────────
总计                : 平均  64.39ms/周期
理论帧率            :  15.53 FPS
```

**结论：** Simulink调用占用78%的总耗时，是主要瓶颈。

### 2.2 MATLAB Engine性能问题

1. **进程间通信开销：** Python ↔ MATLAB Engine 通过IPC通信，延迟高
2. **每次仿真启动成本：** 即使开启Fast Restart，仍需初始化仿真环境
3. **数据序列化/反序列化：** NumPy数组 ↔ MATLAB数组转换
4. **非实时性：** MATLAB Engine无法保证固定响应时间

### 2.3 为什么不能简单优化MATLAB Engine？

| 尝试方案 | 实施情况 | 效果 |
|---------|---------|------|
| 启用Accelerator模式 | ✅ 已启用 | 首次编译后略有改善 (~10%) |
| 启用Fast Restart | ✅ 已启用 | 避免重复编译，但帮助有限 |
| 减少数据传输量 | ❌ 无法减少 | 输入输出已是最小必需 |
| 使用代码生成 | ⚠️ 需要许可证 | 性能最优但成本高 |
| 并行仿真 | ❌ 不适用 | 模型间有依赖关系 |

**结论：** MATLAB Engine架构本身的限制，无法通过配置优化解决。

---

## 3. 技术方案设计

### 3.1 核心思路

**从"每帧调用仿真"改为"持续仿真+UDP通信"**

- **旧方式：** Python每帧启动→运行→停止Simulink仿真
- **新方式：** Simulink持续运行，Python通过UDP发送输入/接收输出

### 3.2 方案架构图

```
┌─────────────────────────────────────────────────────────┐
│                  Python主程序                            │
│               (acc_updated.py)                           │
│                                                          │
│  ┌──────────────────────┐  ┌──────────────────────┐    │
│  │ SimulinkUDPClient 1  │  │ SimulinkUDPClient 2  │    │
│  │   (决策通信)          │  │   (控制通信)          │    │
│  │ Send: 127.0.0.1:25000│  │ Send: 127.0.0.1:26000│    │
│  │ Recv: 127.0.0.1:25001│  │ Recv: 127.0.0.1:26001│    │
│  └──────────┬───────────┘  └──────────┬───────────┘    │
└─────────────┼──────────────────────────┼────────────────┘
              │ UDP数据包                 │ UDP数据包
              │ (4 doubles)               │ (6 doubles)
              ▼                           ▼
┌──────────────────────────┐  ┌───────────────────────────┐
│ Simulink模型1            │  │ Simulink模型2             │
│ acc_decision_core.slx    │  │ sppvt_control_model.slx   │
│                          │  │                           │
│ ┌────────────────────┐  │  │ ┌─────────────────────┐  │
│ │UDP Receive (25000) │  │  │ │UDP Receive (26000)  │  │
│ └─────────┬──────────┘  │  │ └─────────┬───────────┘  │
│           │ Demux        │  │           │ Demux        │
│           ▼              │  │           ▼              │
│   [原有决策逻辑]          │  │   [原有SPPVT逻辑]         │
│      状态机查找表         │  │      三阶段控制器         │
│           │              │  │           │              │
│           │ Mux          │  │           │ Mux          │
│           ▼              │  │           ▼              │
│ ┌────────────────────┐  │  │ ┌─────────────────────┐  │
│ │UDP Send (25001)    │  │  │ │UDP Send (26001)     │  │
│ └────────────────────┘  │  │ └─────────────────────┘  │
│                          │  │                           │
│ 仿真模式: Normal         │  │ 仿真模式: Normal          │
│ 步长: 0.05s (固定)       │  │ 步长: 0.05s (固定)        │
│ 持续运行 (StopTime=inf)  │  │ 持续运行 (StopTime=inf)   │
└──────────────────────────┘  └───────────────────────────┘
```

### 3.3 数据包格式定义

#### **决策模型数据包**

**Python → Simulink (端口25000):**
```
┌──────────┬──────────┬──────────┬──────────┐
│  8 bytes │  8 bytes │  8 bytes │  8 bytes │  共32字节
├──────────┼──────────┼──────────┼──────────┤
│ state    │ command  │ history  │ last_dec │
│ (double) │ (double) │ (double) │ (double) │
└──────────┴──────────┴──────────┴──────────┘
格式: '>4d' (big-endian, 4个double)
```

**Simulink → Python (端口25001):**
```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  8 bytes │  8 bytes │  8 bytes │  8 bytes │  8 bytes │  共40字节
├──────────┼──────────┼──────────┼──────────┼──────────┤
│next_state│ decision │ctrl_en   │next_hist │next_dec  │
│ (double) │ (double) │ (double) │ (double) │ (double) │
└──────────┴──────────┴──────────┴──────────┴──────────┘
格式: '>5d' (big-endian, 5个double)
```

#### **控制模型数据包**

**Python → Simulink (端口26000):**
```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  8 bytes │  8 bytes │  8 bytes │  8 bytes │  8 bytes │  共40字节
├──────────┼──────────┼──────────┼──────────┼──────────┤
│  error   │  offset  │prev_err  │prev_vel  │prev_acc  │
│ (double) │ (double) │ (double) │ (double) │ (double) │
└──────────┴──────────┴──────────┴──────────┴──────────┘
格式: '<5d' (little-endian, 5个double)
```

**Simulink → Python (端口26001):**
```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  8 bytes │  8 bytes │  8 bytes │  8 bytes │  8 bytes │  共40字节
├──────────┼──────────┼──────────┼──────────┼──────────┤
│ control  │ velocity │  accel   │   jerk   │should_upg│
│ (double) │ (double) │ (double) │ (double) │ (double) │
└──────────┴──────────┴──────────┴──────────┴──────────┘
格式: '<5d' (little-endian, 5个double)
```

**字节序说明：**
- 使用Little-Endian（`<`）与测试代码保持一致
- Simulink UDP块会自动处理字节序转换

### 3.4 工具箱选择

**使用工具箱：Instrument Control Toolbox UDP块**

---

## 4. 详细实施步骤

### 阶段0: 准备工作（预计1小时）

#### 0.1 备份当前工作

```bash
# 备份Python代码
cp -r /home/aiden/PycharmProjects/CarlaAcc /home/aiden/PycharmProjects/CarlaAcc_backup_$(date +%Y%m%d)

# 备份Simulink模型
cd /home/aiden/PycharmProjects/CarlaAcc
cp acc_decision_core.slx acc_decision_core_backup.slx
cp sppvt_control_model.slx sppvt_control_model_backup.slx
```

#### 0.2 创建新分支（Git管理）

```bash
git checkout -b feature/udp-communication
git commit -am "Backup before UDP refactor"
```

---

### 阶段1: Simulink模型改造（预计3-4小时）

#### 1.1 改造决策模型（acc_decision_core.slx）

**步骤：**

1. 在MATLAB中打开模型：
   ```matlab
   open_system('acc_decision_core.slx');
   ```

2. 添加UDP Receive块（输入端）：
   - 位置：Simulink Library Browser → Instrument Control Toolbox → UDP Receive
   - 拖拽到模型左侧
   - 配置参数：
     ```
     Remote IP address: 127.0.0.1
     Remote IP port: 25000
     Local IP port: 25001
     Data size: [4, 1]
     Data type: double
     Sample time: 0.05
     ```

3. 添加Demux块（解包输入）：
   - 位置：Simulink → Signal Routing → Demux
   - Number of outputs: 4
   - 连接：UDP Receive输出 → Demux输入

4. 连接到原有输入端口：
   ```
   Demux输出1 → 原In1 (current_state)
   Demux输出2 → 原In2 (command_type)
   Demux输出3 → 原In3 (has_history)
   Demux输出4 → 原In4 (last_active_decision)
   ```
   - 删除原有的In1-In4输入端口块

5. 添加Mux块（打包输出）：
   - 位置：Simulink → Signal Routing → Mux
   - Number of inputs: 5
   - 连接：原Out1-Out5 → Mux输入1-5

6. 添加UDP Send块（输出端）：
   - 位置：Instrument Control Toolbox → UDP Send
   - 配置参数：
     ```
     Remote IP address: 127.0.0.1
     Remote IP port: 25001
     Local IP port: 25000
     Sample time: 0.05
     ```
   - 连接：Mux输出 → UDP Send输入
   - 删除原有的Out1-Out5输出端口块

7. 配置仿真参数：
   ```matlab
   % 设置为持续运行模式
   set_param('acc_decision_core', 'StopTime', 'inf');

   % 确认固定步长
   set_param('acc_decision_core', 'Solver', 'FixedStepDiscrete');
   set_param('acc_decision_core', 'FixedStep', '0.05');

   % 保持Normal模式（不需要Accelerator，UDP块不支持）
   set_param('acc_decision_core', 'SimulationMode', 'normal');
   ```

8. 保存模型：
   ```matlab
   save_system('acc_decision_core');
   ```

#### 1.2 改造控制模型（sppvt_control_model.slx）

**步骤：**

重复1.1的步骤，但配置不同：

- **UDP Receive配置：**
  - Remote IP port: **26000**
  - Local IP port: **26001**
  - Data size: **[5, 1]** ⚠️ 改为5个输入
  - Demux输出数量: **5** ⚠️ 改为5个输出

- **UDP Send配置：**
  - Remote IP port: **26001**
  - Local IP port: **26000**
  - Mux输入数量: **5**

- **仿真参数：** 与决策模型相同

#### 1.3 创建启动脚本（start_simulink_servers.m）

**位置：** `/home/aiden/PycharmProjects/CarlaAcc/start_simulink_servers.m`

```matlab
%% Simulink UDP服务器启动脚本
% 启动决策和控制模型，持续运行等待UDP数据

fprintf('=== Simulink UDP服务器启动中 ===\n\n');

%% 1. 清理环境
close all;
bdclose('all');

%% 2. 加载决策模型
fprintf('1. 加载决策模型: acc_decision_core.slx\n');
load_system('acc_decision_core');

% 启动仿真（非阻塞）
set_param('acc_decision_core', 'SimulationCommand', 'start');
fprintf('   ✅ 决策模型已启动 (UDP端口: 25000/25001)\n\n');

%% 3. 加载控制模型
fprintf('2. 加载控制模型: sppvt_control_model.slx\n');
load_system('sppvt_control_model');

% 启动仿真（非阻塞）
set_param('sppvt_control_model', 'SimulationCommand', 'start');
fprintf('   ✅ 控制模型已启动 (UDP端口: 26000/26001)\n\n');

%% 4. 提示信息
fprintf('═══════════════════════════════════════════\n');
fprintf('🚀 Simulink UDP服务器已就绪\n');
fprintf('═══════════════════════════════════════════\n');
fprintf('端口映射:\n');
fprintf('  决策模型: Python→25000, Python←25001\n');
fprintf('  控制模型: Python→26000, Python←26001\n\n');
fprintf('现在可以启动Python主程序 (acc_updated.py)\n');
fprintf('按 Ctrl+C 停止服务器\n');
fprintf('═══════════════════════════════════════════\n\n');

%% 5. 保持MATLAB运行
% 循环等待，防止脚本退出
try
    while true
        pause(1);
    end
catch ME
    fprintf('\n⚠️ 服务器停止: %s\n', ME.message);
end

%% 6. 清理（Ctrl+C后执行）
fprintf('\n正在停止Simulink模型...\n');
set_param('acc_decision_core', 'SimulationCommand', 'stop');
set_param('sppvt_control_model', 'SimulationCommand', 'stop');
bdclose('all');
fprintf('✅ 清理完成\n');
```

#### 1.4 测试Simulink模型

```matlab
% 在MATLAB命令窗口运行
run('start_simulink_servers.m');

% 应该看到:
% ✅ 决策模型已启动 (UDP端口: 25000/25001)
% ✅ 控制模型已启动 (UDP端口: 26000/26001)
% 🚀 Simulink UDP服务器已就绪
```

**验证端口监听：**
```bash
# Linux/Mac
netstat -an | grep 25000
netstat -an | grep 26000

# Windows
netstat -ano | findstr 25000
netstat -ano | findstr 26000
```

---

### 阶段2: Python代码改造（预计4-5小时）

#### 2.1 创建UDP通信模块（新建文件）

**文件：** `simulink_udp_interface.py`

**内容要点：**
- `SimulinkUDPClient` 类：单个UDP客户端
  - `call(inputs)` 方法：发送输入，接收输出
  - 使用 `struct.pack/unpack` 处理big-endian double
  - 超时检测和错误处理
- `DualSimulinkUDPManager` 类：双UDP管理器
  - `call_decision()` 方法：调用决策模型
  - `call_control()` 方法：调用控制模型
  - `test_connections()` 方法：测试连接

#### 2.2 修改决策管理器

**文件：** `acc_decision_simulink_manager.py`

**关键修改：**

```python
class SimulinkACCDecisionManager:
    def __init__(self, debug=False, max_target_speed_kmh=150.0):
        # 移除matlab_engine参数
        self.debug = debug
        self.max_target_speed_kmh = max_target_speed_kmh

        # 初始化UDP客户端
        from simulink_udp_interface import SimulinkUDPClient
        self.udp_client = SimulinkUDPClient(
            send_port=25000,
            recv_port=25001,
            num_inputs=4,
            num_outputs=5,
            timeout=0.1,
            debug=debug
        )

        # ... 其余初始化保持不变 ...

    def _run_simulink_decision(self, command_type: int):
        """运行Simulink决策模型 - UDP版本"""
        inputs = [
            float(self.current_state),
            float(command_type),
            float(1 if self.has_history else 0),
            float(self.last_active_decision)
        ]

        start_time = time.time()
        outputs = self.udp_client.call(inputs)
        elapsed_ms = (time.time() - start_time) * 1000.0

        # 解析输出
        next_state = int(round(outputs[0]))
        decision = int(round(outputs[1]))
        control_enabled = bool(int(round(outputs[2])))
        next_has_history = int(round(outputs[3]))
        next_last_decision = int(round(outputs[4]))

        # 更新状态
        self.current_state = next_state
        self.has_history = bool(next_has_history)
        self.last_active_decision = next_last_decision

        if self.debug:
            print(f"🔧 UDP决策: S{self.current_state}, R{decision}, "
                  f"enabled={control_enabled} ({elapsed_ms:.1f}ms)")

        return control_enabled, decision

    def cleanup(self):
        """清理UDP资源"""
        if hasattr(self, 'udp_client'):
            self.udp_client.cleanup()
```

**删除的方法：**
- `_ensure_model_loaded()` - 不再需要加载模型
- 所有与`matlab_engine`相关的代码

#### 2.3 修改控制管理器

**文件：** `sppvt_manager_simulink.py`

**关键修改：**

```python
class SimulinkSPPVTManager(BaseSPPVTManager):
    def __init__(self, model_name='sppvt_control_model',
                 params=None, debug=False):
        super().__init__(params=params, debug=debug)
        self.model_name = model_name

        # 初始化UDP客户端
        from simulink_udp_interface import SimulinkUDPClient
        self.udp_client = SimulinkUDPClient(
            send_port=26000,
            recv_port=26001,
            num_inputs=6,
            num_outputs=5,
            timeout=0.1,
            debug=debug
        )

    def _run_simulink(self, inputs: List[float]) -> tuple[list[float], float]:
        """运行Simulink控制模型 - UDP版本"""
        start_time = time.time()
        outputs = self.udp_client.call(inputs)
        elapsed_ms = (time.time() - start_time) * 1000.0

        if self.debug:
            print(f"✅ UDP控制输出: control={outputs[0]:.4f}, "
                  f"should_upgrade={outputs[4]:.0f} ({elapsed_ms:.1f}ms)")

        return outputs, elapsed_ms

    def cleanup(self):
        """清理UDP资源"""
        if hasattr(self, 'udp_client'):
            self.udp_client.cleanup()
```

**删除的方法：**
- `initialize_matlab_engine()`
- `_ensure_model_loaded()`
- `_sync_params_to_constant_blocks()`

**注意：** SPPVT参数现在只能通过Simulink模型内的Constant块配置，Python端不再动态同步。

#### 2.4 修改外观类（Facade）

**文件：** `acc_control_facade.py`

**关键修改：**

```python
class ACCControlFacade:
    def __init__(self, config=None, mode='hybrid', debug=False):
        # 移除matlab_engine参数
        self.config = config or ACCConfig()
        self.mode = mode
        self.debug = debug

        # 决策模块 - UDP版本
        self.acc_controller = SimulinkACCDecisionManager(
            debug=debug,
            max_target_speed_kmh=self.config.max_target_speed_kmh
        )

        # 控制模块 - UDP版本
        self.sppvt_manager = SimulinkSPPVTManager(
            model_name='sppvt_control_model',
            debug=debug
        )

        # ... 其余代码保持不变 ...
```

**删除的代码：**
- `matlab_engine` 参数
- 从 `matlab_engine_factory` 的导入

#### 2.5 修改主程序

**文件：** `acc_updated.py`

**关键修改：**

```python
class acc:
    def __init__(self, use_result_plotter=None, scenario_mode=None):
        # ... 现有初始化代码 ...

        # === ACC混合控制器模块（UDP版本）===
        self.acc_decision_sppvt = ACCControlFacade(
            config=self.config,
            debug=self.config.acc_decision_debug
        )

        # 注意：不再需要传递matlab_engine

    def destroy(self):
        print("Cleaning up resources...")

        # 1. 停止实时绘图器
        # ... 现有代码 ...

        # 2. 清理UDP连接（新增）
        try:
            if hasattr(self, 'acc_decision_sppvt') and self.acc_decision_sppvt:
                self.acc_decision_sppvt.cleanup()
                print("✅ UDP连接已关闭")
        except Exception as e:
            print(f"⚠️ Error cleaning up UDP: {e}")

        # 3. 清理车辆和传感器
        # ... 现有代码保持不变 ...
```

#### 2.6 删除MATLAB Engine工厂（可选）

**文件：** `matlab_engine_factory.py`

**操作：** 可以保留此文件作为备份，但不再被任何代码调用。

---

### 阶段3: 启动流程（预计1小时）

#### 3.1 手动启动流程（用户当前需求）

**步骤：**

1. **终端1（MATLAB）：**
   ```matlab
   cd /home/aiden/PycharmProjects/CarlaAcc
   run('start_simulink_servers.m');
   ```
   等待显示：`🚀 Simulink UDP服务器已就绪`

2. **终端2（Python）：**
   ```bash
   cd /home/aiden/PycharmProjects/CarlaAcc
   python acc_updated.py
   ```

3. **停止：**
   - 先停Python（Ctrl+C）
   - 再停MATLAB（Ctrl+C）

#### 3.2 验证连接

Python启动时应该看到：
```
✅ UDP客户端初始化: 发送→25000, 接收←25001
✅ UDP客户端初始化: 发送→26000, 接收←26001
```

如果看到超时错误，说明Simulink服务器未启动或端口配置错误。

---

## 5. 文件修改清单

### 5.1 新建文件

| 文件名 | 类型 | 作用 | 行数估计 |
|--------|------|------|---------|
| `simulink_udp_interface.py` | Python | UDP通信基类和双模型管理器 | ~300 |
| `start_simulink_servers.m` | MATLAB | Simulink UDP服务器启动脚本 | ~60 |

### 5.2 修改文件

| 文件名 | 修改类型 | 主要改动 | 行数变化 |
|--------|----------|---------|---------|
| `acc_decision_simulink_manager.py` | 重构 | 删除MATLAB Engine，改用UDP | -100/+50 |
| `sppvt_manager_simulink.py` | 重构 | 删除MATLAB Engine，改用UDP | -120/+40 |
| `acc_control_facade.py` | 简化 | 移除matlab_engine参数 | -20/+5 |
| `acc_updated.py` | 微调 | cleanup()中添加UDP清理 | +10 |

### 5.3 Simulink模型修改

| 模型文件 | 修改内容 | 新增块数量 |
|---------|---------|-----------|
| `acc_decision_core.slx` | 替换In/Out端口为UDP块 | 4个 (UDP Rx/Tx, Mux/Demux) |
| `sppvt_control_model.slx` | 替换In/Out端口为UDP块 | 4个 (UDP Rx/Tx, Mux/Demux) |

### 5.4 备份文件

| 文件名 | 位置 |
|--------|------|
| `acc_decision_core_backup.slx` | 项目根目录 |
| `sppvt_control_model_backup.slx` | 项目根目录 |

### 5.5 删除/废弃的代码

| 文件 | 操作 | 说明 |
|------|------|------|
| `matlab_engine_factory.py` | 保留但不再调用 | 可作为历史参考 |
| `acc_*_manager.py`中的`_ensure_model_loaded()` | 删除 | 不再需要加载模型 |
| 所有`matlab_engine.sim()`调用 | 替换为UDP | 核心改动 |

---

## 6. 附录

### 6.1 术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| UDP | User Datagram Protocol | 用户数据报协议，无连接网络协议 |
| IPC | Inter-Process Communication | 进程间通信 |
| ACC | Adaptive Cruise Control | 自适应巡航控制 |
| SPPVT | Speed Profile Planning with Variable Time gap | 可变时距速度规划 |
| Big-Endian | - | 大端字节序，网络字节序标准 |

### 6.2 端口映射表

| 端口 | 方向 | 用途 | 数据格式 |
|------|------|------|---------|
| 25000 | Python → Simulink | 决策模型输入 | 4 doubles (32字节) |
| 25001 | Simulink → Python | 决策模型输出 | 5 doubles (40字节) |
| 26000 | Python → Simulink | 控制模型输入 | 6 doubles (48字节) |
| 26001 | Simulink → Python | 控制模型输出 | 5 doubles (40字节) |

### 6.3 故障排查指南

#### 问题1: "UDP接收超时"

**症状：**
```
TimeoutError: ❌ UDP接收超时(0.1s): Simulink模型可能未运行或端口配置错误
```

**可能原因：**
1. Simulink服务器未启动
2. 端口配置错误
3. 防火墙阻止

**解决方案：**
```bash
# 1. 检查Simulink是否运行
# 在MATLAB中应该看到两个正在仿真的模型

# 2. 检查端口监听
netstat -an | grep 25000

# 3. 检查防火墙（临时关闭测试）
sudo ufw disable  # Linux
```

#### 问题2: "数据解析失败"

**症状：**
```
ValueError: ❌ UDP数据解析失败: unpack requires a buffer of 40 bytes
```

**可能原因：**
1. Simulink输出数量配置错误
2. 字节序不匹配

**解决方案：**
```python
# 验证Simulink输出数量
# 在MATLAB中运行:
get_param('acc_decision_core/UDP_Send', 'DataSize')
# 应该返回 [5, 1]

# 检查Python端配置
num_outputs = 5  # 必须与Simulink一致
```

#### 问题3: "Simulink模型停止运行"

**症状：**
Simulink仿真自动停止，UDP连接断开

**可能原因：**
1. 模型StopTime设置错误
2. Simulink错误导致停止

**解决方案：**
```matlab
% 检查StopTime
get_param('acc_decision_core', 'StopTime')
% 应该返回 'inf'

% 重新启动
set_param('acc_decision_core', 'SimulationCommand', 'start');
```

### 6.4 参考资料

**MathWorks官方文档：**
- [UDP Send/Receive Blocks](https://www.mathworks.com/help/instrument/udpreceive.html)
- [Simulink External Mode](https://www.mathworks.com/help/rtw/ug/external-mode-simulation-with-tcpip-or-serial-communication.html)
- [Fixed-Step Discrete Solver](https://www.mathworks.com/help/simulink/ug/solvers.html)

**开源项目参考：**
- [RealTime-UDP-Communication-with-Simulink-and-Python](https://github.com/RitterD/RealTime-UDP-Communication-with-Simulink-and-Python)
- [TCP-communication-between-Python-and-Maltab-Simulink](https://github.com/z1223343/TCP-communication-between-Python-and-Maltab-Simulink)

**Python文档：**
- [socket — Low-level networking](https://docs.python.org/3/library/socket.html)
- [struct — Interpret bytes as packed binary data](https://docs.python.org/3/library/struct.html)

---

## 📝 审核检查清单

在审核本方案时，请关注以下关键点：

- [ ] **架构设计** - 双UDP通道设计是否合理？
- [ ] **接口定义** - 数据包格式是否完整？（特别是should_upgrade端口）
- [ ] **实施步骤** - 步骤是否清晰可执行？
- [ ] **文件修改清单** - 是否覆盖所有需要修改的文件？
- [ ] **删除MATLAB Engine** - 是否完全移除依赖？
- [ ] **启动流程** - 手动启动是否可行？
- [ ] **文档完整性** - 是否遗漏关键信息？

---

## ✅ 下一步行动

**审核通过后：**

1. 开始实施阶段0：备份和准备
2. 实施阶段1：Simulink模型改造
3. 实施阶段2：Python代码改造
4. 实施阶段3：测试验证

**如有疑问或需要修改方案，请提出具体意见。**

---

**方案结束** 📄