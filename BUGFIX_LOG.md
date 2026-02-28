# ACC 问题修复记录

记录从 2026-02-27 起发现的问题及对应解决方案。

---

## BUG-001：误差变号时扭矩突然下降

### 问题描述

当控制误差由正变负（或由负变正）时，即使变化量很小，输出扭矩出现大幅骤降。

### 根本原因

控制方程：`Y(k) = Kp × (error + stage_offset) + Y0`

`StageManager` 内部逻辑：当检测到 `AND_sign_changed = True` 时，触发 `OR_clear`，将 `stage_offset` **瞬间归零**。

扭矩跳变量 = `Kp × stage_offset`，在 stage_offset 积累较大时可达数百 N·m。

### 解决方案（待实现）：Y0 持续追踪当前工作点

**核心思路**：将 Y0 从"仅在启控瞬间锁存"改为"启控时瞬间对齐 + 控制中持续缓慢追踪 `current_engine_torque`"。

**控制律**：
```
启控瞬间（control_enabled 上升沿）：
    Y0 = current_engine_torque  （瞬间对齐，行为与现在相同）

ACC 激活期间（每帧）：
    Y0(k) = Y0(k-1) + α × (current_engine_torque - Y0(k-1))

ACC 未激活：
    Y0 冻结，保持上次值
```

**参数**：`α = 0.03`（可调范围 0.02~0.08，约 1.5s 完成跟踪）

**为何能自然处理所有场景，无需针对变号单独判断**：

| 场景 | 误差过零速度 | Y0 追踪程度 | 结果 |
|------|------------|------------|------|
| 自然收敛（慢速） | 慢（多帧） | 充分，Y0 ≈ 当前输出 | 过零时 stage_offset 归零，Y0 已吸收其贡献，无跳变 ✅ |
| 前车急刹（快速） | 快（1~2帧） | 几乎未追踪 | stage_offset 归零后扭矩骤降 → 制动及时生效 ✅ |
| 司机踩油门介入 | 快 | 追踪缓慢 | 不会锁存司机高扭矩，接管后不超调 ✅ |

**需修改的文件**：
- `create_simplified_subsystems.m`：`create_y0_latch_v2` 函数，将 Switch+Memory 锁存替换为追踪滤波器
- `apply_sign_change_fix.m`：更新重建脚本

**Simulink 内部结构变化**（Y0_Latch 子系统）：
```
新增块：
  Sub_track  = current_engine_torque - Y0_prev
  Prod_alpha = α × Sub_track
  Add_track  = Y0_prev + Prod_alpha          ← 追踪值
  Sw_snap    = rising_edge ? snap : tracked  ← 启控瞬间对齐
  Sw_freeze  = control_enabled ? Sw_snap : Y0_prev  ← 未激活冻结

接口不变（仍为2输入）：
  /1 control_enabled
  /2 current_engine_torque
```

**状态**：⏳ 待实现

---
