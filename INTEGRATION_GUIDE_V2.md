# Simulink模型整合指南 V2
## 设计理念：Python维护状态，Simulink纯计算

---

## 核心原则

✅ **Simulink只做纯计算**：每帧输入→计算→输出，无内部状态
✅ **Python维护所有状态**：current_state, prev_error, stage_offset等
✅ **避免因果环**：Decision和SPPVT完全解耦，通过Python协调

---

## 整合方案：两个子模块 + 统一UDP接口

### 架构图

```
Python端状态管理
    ↓ (准备输入)
[统一UDP输入 9+5=14个double]
    ↓
┌─────────────────────────────┐
│  acc_integrated_model.slx   │
│  ┌──────────────────────┐   │
│  │ UDP Receive (14路)   │   │
│  └──────┬──────┬────────┘   │
│         │      │             │
│    ┌────┘      └────┐        │
│    ↓                ↓        │
│ [Decision子系统]  [SPPVT子系统] │  ← 完全独立，无连接
│    │ (4输入5输出)  │ (5输入5输出)│
│    └────┬──────┬────┘        │
│         │      │             │
│  ┌──────┴──────┴────────┐   │
│  │ Mux (10路输出)       │   │
│  └──────────┬───────────┘   │
│             ↓               │
│  ┌──────────────────────┐   │
│  │ UDP Send (10路)      │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
    ↓
Python端接收并更新状态
```

---

## Step 1: 创建新的集成模型

### 1.1 在MATLAB中创建新模型

```matlab
% 在MATLAB命令窗口执行
cd /home/aiden/PycharmProjects/CarlaAcc
new_system('acc_integrated_model');
open_system('acc_integrated_model');
save_system('acc_integrated_model');
```

---

## Step 2: 添加UDP接收模块（14个输入）

### 2.1 添加UDP Receive模块

1. 在模型窗口，按 **Ctrl+Shift+L** 打开Library Browser
2. 搜索 `UDP Receive`
3. 拖入模型左侧
4. **双击模块**打开配置窗口

### 2.2 配置UDP Receive参数

```
Remote IP address: 127.0.0.1
Local IP port: 27000
Data size: 14
Data type: double
Sample time: 0.05
Enable blocking mode: ☐ (不勾选，非阻塞)
```

点击 **OK** 保存。

### 2.3 添加Demux模块（信号拆分）

1. 搜索 `Demux`，拖入UDP Receive右侧
2. **双击Demux**配置：
   ```
   Number of outputs: 14
   Display option: bar
   ```

### 2.4 连接UDP Receive到Demux

- 点击UDP Receive的输出端口
- 拖动到Demux的输入端口
- 松开鼠标完成连接

### 2.5 给Demux输出添加信号标签（可选但推荐）

**方法**：
1. **右键点击Demux的某个输出线**
2. 选择 **Properties**
3. 在Signal Properties窗口，找到 **Signal name** 字段
4. 输入信号名称（例如 `ego_speed_kmh`）
5. 点击 **OK**

**14个输出信号名称**（按顺序）：
```
输出1-4: Decision输入
  1: current_state
  2: command_type
  3: has_history
  4: last_active_decision

输出5-9: SPPVT输入
  5: error_value
  6: stage_offset
  7: prev_error
  8: prev_velocity
  9: prev_accel

输出10-14: 辅助输入（用于后处理/调试）
  10: ego_speed_kmh
  11: V_min_kmh
  12: control_mode_flag
  13: control_error_raw (原始误差，未经enable过滤)
  14: manual_throttle_active
```

**注意**：信号标签是可选的，主要用于提高可读性。如果不添加，Simulink会自动使用默认名称。

---

## Step 3: 创建Decision子系统

### 3.1 方法：从现有模型复制内容

1. **打开现有Decision模型**：
   ```matlab
   open_system('acc_decision_core.slx');
   ```

2. **选中核心计算部分**：
   - 在模型中，**Ctrl+A** 全选
   - **取消选择** UDP Receive 和 UDP Send 模块（按住Ctrl点击）
   - 剩下的就是核心逻辑（Lookup Tables, Data Type Conversion等）

3. **创建子系统**：
   - 右键选中的部分 → **Create Subsystem from Selection**
   - Simulink会自动创建子系统，并生成Inport/Outport

4. **复制子系统**：
   - **Ctrl+C** 复制子系统
   - 切换到 `acc_integrated_model`
   - **Ctrl+V** 粘贴
   - 将子系统命名为 `Decision_Core`

### 3.2 配置子系统的输入输出端口

**检查子系统端口**：
1. **双击子系统**进入内部
2. 确认有4个 **Inport** 模块：
   ```
   Inport1: current_state
   Inport2: command_type
   Inport3: has_history
   Inport4: last_active_decision
   ```
3. 确认有5个 **Outport** 模块：
   ```
   Outport1: next_state
   Outport2: decision
   Outport3: control_enabled
   Outport4: next_has_history
   Outport5: next_last_decision
   ```

**如果没有Inport/Outport**（手动添加）：
1. 搜索 `Inport`，拖入4个
2. 搜索 `Outport`，拖入5个
3. 连接到原有逻辑的输入输出

**返回顶层模型**：
- 点击模型窗口上方的 **Up to Parent** 按钮（或按 Ctrl+U）

---

## Step 4: 创建SPPVT子系统

### 4.1 同样方法，从sppvt_control_model复制

1. **打开现有SPPVT模型**：
   ```matlab
   open_system('sppvt_control_model.slx');
   ```

2. **选中核心计算部分**（排除UDP模块）

3. **创建并复制子系统**到 `acc_integrated_model`

4. 命名为 `SPPVT_Core`

### 4.2 确认SPPVT端口

**输入（5个）**：
```
Inport1: error_value (控制误差)
Inport2: stage_offset (级差累积值)
Inport3: prev_error (上一帧误差)
Inport4: prev_velocity (上一帧速度)
Inport5: prev_accel (上一帧加速度)
```

**输出（5个）**：
```
Outport1: control_output
Outport2: velocity_output
Outport3: acceleration_output
Outport4: jerk_output
Outport5: should_upgrade
```

---

## Step 5: 连接Demux到子系统

### 5.1 连接Decision子系统

```
Demux输出1 (current_state) → Decision_Core输入1
Demux输出2 (command_type) → Decision_Core输入2
Demux输出3 (has_history) → Decision_Core输入3
Demux输出4 (last_active_decision) → Decision_Core输入4
```

**操作方法**：
- 点击Demux的输出端口1
- 拖动到Decision_Core的输入端口1
- 重复4次

### 5.2 连接SPPVT子系统

```
Demux输出5 (error_value) → SPPVT_Core输入1
Demux输出6 (stage_offset) → SPPVT_Core输入2
Demux输出7 (prev_error) → SPPVT_Core输入3
Demux输出8 (prev_velocity) → SPPVT_Core输入4
Demux输出9 (prev_accel) → SPPVT_Core输入5
```

**关键点**：
- ❌ **不要**连接Decision的control_enabled到SPPVT
- ✅ Python端根据上一帧的control_enabled决定这一帧传给SPPVT的error_value
- ✅ 如果control_enabled=False，Python传error_value=0

---

## Step 6: 添加输出Mux和UDP Send

### 6.1 添加Mux模块

1. 搜索 `Mux`，拖入右侧
2. 配置：
   ```
   Number of inputs: 10
   Display option: bar
   ```

### 6.2 连接子系统输出到Mux

```
Decision_Core输出1 (next_state) → Mux输入1
Decision_Core输出2 (decision) → Mux输入2
Decision_Core输出3 (control_enabled) → Mux输入3
Decision_Core输出4 (next_has_history) → Mux输入4
Decision_Core输出5 (next_last_decision) → Mux输入5

SPPVT_Core输出1 (control_output) → Mux输入6
SPPVT_Core输出2 (velocity_output) → Mux输入7
SPPVT_Core输出3 (acceleration_output) → Mux输入8
SPPVT_Core输出4 (jerk_output) → Mux输入9
SPPVT_Core输出5 (should_upgrade) → Mux输入10
```

### 6.3 添加UDP Send模块

1. 搜索 `UDP Send`，拖入最右侧
2. 配置：
   ```
   Remote IP address: 127.0.0.1
   Remote IP port: 27001
   Sample time: 0.05
   Enable blocking mode: ☐ (不勾选)
   ```

### 6.4 连接Mux到UDP Send

```
Mux输出 → UDP Send输入
```

---

## Step 7: 配置模型仿真参数

### 7.1 打开Configuration Parameters

- 快捷键：**Ctrl+E**
- 或菜单：**Simulation → Model Configuration Parameters**

### 7.2 Solver设置

```
Solver
  ├─ Type: Fixed-step
  ├─ Solver: discrete (no continuous states)
  └─ Fixed-step size: 0.05
```

### 7.3 Data Import/Export设置

```
Data Import/Export
  ├─ ☐ Time (取消勾选)
  ├─ ☐ Output (取消勾选)
  └─ ☐ States (取消勾选)
```

### 7.4 诊断设置（避免警告）

```
Diagnostics → Connectivity
  └─ Signal resolution: none
```

### 7.5 点击OK保存设置

---

## Step 8: 保存并检查模型

### 8.1 保存模型

```matlab
save_system('acc_integrated_model');
```

### 8.2 检查模型连接

```matlab
% 检查是否有未连接的端口
unconnected = find_system('acc_integrated_model', 'Type', 'Port', 'Line', -1);
if isempty(unconnected)
    disp('✅ 所有端口已连接');
else
    disp('❌ 发现未连接端口：');
    disp(unconnected);
end
```

### 8.3 检查UDP配置

```matlab
% 检查UDP Receive端口
recv_port = get_param('acc_integrated_model/UDP Receive', 'LocalPort');
fprintf('UDP Receive端口: %s\n', recv_port);

% 检查UDP Send端口
send_port = get_param('acc_integrated_model/UDP Send', 'RemotePort');
fprintf('UDP Send端口: %s\n', send_port);
```

---

## Step 9: 可视化最终布局（可选）

### 9.1 自动排列模块

```matlab
open_system('acc_integrated_model');
Simulink.BlockDiagram.arrangeSystem('acc_integrated_model');
```

### 9.2 推荐布局

```
从左到右：
[UDP Receive] → [Demux] → [Decision_Core]
                          [SPPVT_Core]   → [Mux] → [UDP Send]

垂直排列Decision和SPPVT，便于查看
```

---

## 常见问题解答

### Q1: 如何给信号线添加标签？

**方法1：通过Properties**
1. 右键点击信号线
2. Properties → Signal name
3. 输入名称，确定

**方法2：双击信号线**
1. 双击信号线弹出编辑框
2. 直接输入名称
3. 回车确定

### Q2: 子系统输入输出必须是Inport/Outport吗？

**是的**，Simulink子系统的规则：
- **内部输入**必须用 `Inport` 模块
- **内部输出**必须用 `Outport` 模块
- 顶层模型中，子系统会自动显示对应的输入输出端口

### Q3: Python端如何处理control_enabled的时序？

**推荐方案（延迟一帧，更安全）**：

```python
# 在Python端维护上一帧的control_enabled
self._last_control_enabled = False

# 准备输入时
if self._last_control_enabled:
    error_value = control_error  # 使用实际误差
else:
    error_value = 0.0  # 强制为0，SPPVT不工作

inputs = [
    current_state,
    command_type,
    has_history,
    last_active_decision,
    error_value,  # ← 已根据上一帧的enable过滤
    stage_offset,
    prev_error,
    prev_velocity,
    prev_accel,
    # ... 其他辅助输入
]

# 调用Simulink
outputs = self.udp_client.call(inputs)

# 更新control_enabled供下一帧使用
self._last_control_enabled = bool(outputs[2])  # control_enabled在输出3
```

**优点**：
- ✅ 避免Simulink内部因果环
- ✅ 逻辑清晰，便于调试
- ✅ 延迟一帧对控制影响极小（0.05s）

### Q4: SPPVT的状态（prev_error等）在哪更新？

**完全在Python端**：

```python
class IntegratedSimulinkManager:
    def __init__(self):
        # SPPVT状态
        self.sppvt_state = {
            'stage_offset': 0.0,
            'prev_error': 0.0,
            'prev_velocity': 0.0,
            'prev_accel': 0.0,
        }

    def process_cycle(self, control_error, ...):
        # 准备SPPVT输入（使用上一帧的状态）
        inputs = [
            # ...
            error_value,
            self.sppvt_state['stage_offset'],
            self.sppvt_state['prev_error'],
            self.sppvt_state['prev_velocity'],
            self.sppvt_state['prev_accel'],
        ]

        # 调用Simulink
        outputs = self.udp_client.call(inputs)

        # 更新SPPVT状态（使用当前帧的输出）
        self.sppvt_state['prev_error'] = error_value
        self.sppvt_state['prev_velocity'] = outputs[6]  # velocity_output
        self.sppvt_state['prev_accel'] = outputs[7]     # acceleration_output

        # 根据should_upgrade更新stage_offset（Python实现升级逻辑）
        if outputs[9] > 0.5:  # should_upgrade
            self.sppvt_state['stage_offset'] += 某个增量
```

### Q5: 为什么不在Simulink中用Unit Delay？

**原因**：
1. ❌ 增加模型复杂度
2. ❌ 状态分散在Python和Simulink两处，难以调试
3. ❌ 违反"Simulink无状态"原则
4. ✅ Python端维护状态更灵活，易于保存/恢复
5. ✅ 便于实现热重启、状态快照等高级功能

---

## 下一步

完成Simulink模型整合后，需要修改Python端代码以适配新接口。

请参考：`python_integration_update.md`（即将创建）