# Simulink SPPVT 改造计划（基于现有 acc_integrated_model.slx）

## 1. 需求摘要（确认版）
- 控制方程改为：`Y(k) = Kp * e_i(k) + Y0`，**control_output 直接输出 N·m**。
- Python 端提前完成误差符号约定与 `reset_flag` 逻辑。
- **保留** `Derivatives_Subsystem` 和 `Upgrade_Condition_Subsystem`（升级逻辑必须依赖导数链）。
- `control_enabled` 由 Decision 子系统输出，在 Simulink 内部作为控制使能/复位信号使用。
- Simulink 端输出仅需 `control_output`（加上 Decision 的 5 个输出）。
- `create_decision_sppvt_bus.m` **已废弃**，保留文件但不作为后续分析依据。

---

## 2. 新接口定义（Python ↔ Simulink）

### 2.1 输入（Python → Simulink）
1. `current_state` (Decision)
2. `command_type` (Decision)
3. `has_history` (Decision)
4. `last_active_decision` (Decision)
5. `control_error_signed`（Python 已做符号约定）
6. `Y0`（Python 估算的初始扭矩，N·m）
7. `reset_flag`（Python 计算：是否清零）

> 备注：如后续需要保留 G2 突变检测，也可新增第 8 个输入 `G2_s`。当前方案中由 Python 统一通过 `reset_flag` 触发，因此 **G2_s 不再强制作为 Simulink 输入**。

### 2.2 输出（Simulink → Python）
1. `next_state`
2. `decision`
3. `control_enabled`
4. `next_has_history`
5. `next_last_decision`
6. `control_output`（N·m）

---

## 3. Simulink 端改造步骤（逐步搭建）

### Step 1：调整顶层端口
- 将 SPPVT 输入端口改为：`control_error_signed / Y0 / reset_flag`（+ Decision 四个输入）。
- 将 SPPVT 输出端口缩减为：`control_output`（只保留 1 个）。
- Decision 子系统端口保持不变。

### Step 2：Derivatives_Subsystem 仅保留一阶导数（速度）
升级条件只需要一阶导数，因此 Derivatives_Subsystem 只计算 **error 的一阶导数**：
- 删除加速度与三阶导数相关链路
- 仅保留误差差分与低通滤波（若保留滤波）

内部状态建议：
```
prev_error = Unit Delay(error_value)
```
- 若滤波器使用内部状态（Discrete Filter），不再需要 `prev_velocity`
- Unit Delay 开启 Reset 端口  
- Reset 信号 = `reset_flag OR ~control_enabled`

### Step 3：保留 Upgrade_Condition_Subsystem（仅使用一阶导数）
- 输入来自 Derivatives 的 `velocity` 与 `error_value`  
- 已手动修改升级条件：不再使用加速度或更高阶导数  
- 输出仍为 `should_upgrade`

### Step 4：新增 StageManager 子系统（替代 Python 升级记录）
**Inputs**
- `error_value`
- `should_upgrade`
- `reset_flag`
- `control_enabled`

**States（Unit Delay 或 MATLAB Function persistent）**
- `stage_offset`
- `stage`
- `error_sign`
- `cooldown`

**逻辑伪码（与原 Python 行为一致）**
```
if reset_flag || ~control_enabled:
    stage_offset = 0
    stage = 1
    cooldown = 0
    error_sign = 0
else:
    current_sign = sign(error_value)
    if sign_changed:
        stage_offset = 0
        stage = 1
        cooldown = 0
    elif should_upgrade && cooldown == 0:
        stage += 1
        stage_offset += sign(error_value) * rho * abs(error_value)
        stage_offset = clamp(stage_offset, -500, 500)
        cooldown = upgrade_cooldown_frames
    else:
        if cooldown > 0: cooldown -= 1
    if current_sign != 0: error_sign = current_sign
```

**参数建议（常量块）**
- `rho`：沿用原 `integrated_sppvt_rho`（例如 0.1）
- `upgrade_cooldown_frames`：沿用原 `sppvt_upgrade_cooldown`（例如 2）

### Step 5：简化 Core Control（保留 SPPVT 核心功能）
```
enhanced_error = error_value + stage_offset
control_output = Kp * enhanced_error + Y0
```
- `Kp` 作为常量块（值与原 `sppvt_accel_scale` 一致时单位为 N·m / 误差单位）
- 可选：在 `control_enabled` 为 False 时输出 0（安全优先）

---

## 4. Python 端同步修改（清单）
- `integrated_simulink_manager.py`
  - 输入输出长度按新接口调整（Decision 4 + 3 = 7 输入；Decision 5 + 1 = 6 输出）
  - 删除所有 SPPVT 状态维护逻辑（stage_offset、prev_*、cooldown 等）
  - 仅保留 `control_output` 的解析
- `control_loop_manager.py`
  - 使用 `control_output` 作为**扭矩 N·m**
  - 不再做 `sppvt_accel_scale` 缩放
  - 保留纵向限幅 `longitudinal_constraint_limiter.py`
- 绘图/日志
  - 去除对 `sppvt_velocity_output / sppvt_acceleration_output / sppvt_enhanced_error / sppvt_should_upgrade` 的依赖

---

## 5. 验证与检查点
1. **控制使能/失能**：`control_enabled = 0` 时输出应清零或保持安全值
2. **reset_flag**：触发时 stage/offset/cooldown 清零
3. **升级逻辑**：error 收敛时 `should_upgrade` 能触发 stage 递增
4. **扭矩单位**：`control_output` 应直接为 N·m（Python 不再二次缩放）

---

## 6. 风险与注意事项
- 若 `Kp` 单位未同步，会导致输出扭矩量级错误
- StageManager 内部状态必须与 reset_flag 同步，否则会出现"历史污染"
- 删除二阶/三阶导数后，调试数据减少，必要时可临时保留 debug 端口

---

## 7. Simulink 离散系统时序行为总结（经验记录）

基于 StageManager 模块的测试验证，总结以下时序规律：

### 7.1 核心概念：Unit Delay 的作用

```
┌─────────────────────────────────────────────────────────────────┐
│  时刻 k 的执行流程                                              │
│                                                                 │
│  1. 读取 Unit Delay 输出 ──► 获得 x(k-1)（上一帧写入的值）      │
│  2. 组合逻辑计算 ──────────► 基于输入 u(k) 和 x(k-1) 计算       │
│  3. 写入 Unit Delay 输入 ──► 存储 x(k)（供下一帧使用）          │
│  4. 输出端口 ──────────────► 输出当前帧计算结果                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 输出端口连接方式决定时序

| 连接方式 | 行为 | 适用场景 |
|----------|------|----------|
| 输出 ← Switch/计算结果 | **即时输出**：同一帧计算完立即可见 | 控制信号（推荐） |
| 输出 ← Unit Delay 输出 | **延迟输出**：下一帧才能看到变化 | 特殊需求 |

**StageManager 采用即时输出**：所有输出端口连接到 Switch 的输出，而非 Unit Delay 的输出。

### 7.3 实测时序验证结果

以 `dt = 0.05s` 为例：

```
时刻      输入                    输出（即时）
────────────────────────────────────────────────────────
t=0.05    reset=1                 stage=1, offset=0, cooldown=0
t=0.10    reset=0, err=0.5        stage=1, offset=0, cooldown=0, sign=+1
t=0.15    upgrade=1, err=0.5      stage=2, offset=0.05, cooldown=2  ← 同帧生效
t=0.20    upgrade=0               stage=2, offset=0.05, cooldown=1  ← cooldown递减
t=0.55    err从+变-               stage=1, offset=0 ← 符号翻转，同帧重置
```

**关键结论**：
- 升级触发 → **同帧** 输出新的 stage/offset/cooldown
- 符号翻转 → **同帧** 输出重置后的状态
- reset_flag=1 → **同帧** 输出归零状态

### 7.4 边界条件注意事项

| 场景 | 行为 | 是否影响控制 |
|------|------|--------------|
| control_enabled 从 0→1 | error_sign 可能延迟一帧更新 | 否（不触发误判） |
| reset_flag 脉冲 | 所有状态同帧清零 | 否 |
| 死区内 (error ≈ 0) | current_sign=0，不更新 error_sign | 否 |

### 7.5 调试技巧

1. **添加 To Workspace 块**：将内部信号导出到 MATLAB 工作区
2. **使用 Scope 观察**：实时查看状态变化
3. **检查采样时间**：确保所有块的 SampleTime 一致（使用 `-1` 继承）
4. **From Workspace 设置**：注意插值模式，离散系统建议用 `Zero-Order Hold`

### 7.6 验证脚本使用方法

```matlab
% 运行 StageManager 独立验证
results = run_stage_manager_validation();

% 查看期望值 vs 实际值
sm_expected    % MATLAB 模拟的期望值
sm_stage_offset % Simulink 实际输出

% 如有失败，自动绘制对比图
```

### 7.7 集成到主模型的检查清单

- [ ] 确认 StageManager 的 4 个输入正确连接
- [ ] 确认 stage_offset 输出连接到 Core Control 的加法器
- [ ] 确认 reset_signal = `reset_flag OR NOT(control_enabled)` 传递给 Derivatives 的 Unit Delay Reset 端口
- [ ] 运行端到端仿真，检查 control_output 量级是否合理
