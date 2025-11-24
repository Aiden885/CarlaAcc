# ACC决策类型分类

## R1-R6: 有效的ACC控制决策（会被记录）

这些是**ACC系统自主执行的控制策略**：

| 决策 | 名称 | 触发条件 | 含义 |
|------|------|----------|------|
| R1 | DECREASE_SPEED | S0 + E键 | ACC降低目标速度 |
| R2 | INCREASE_SPEED | S0 + Q键 | ACC提高目标速度 |
| R3 | DECREASE_DISTANCE | S0 + T键 | ACC缩短跟车时距 |
| R4 | INCREASE_DISTANCE | S0 + R键 | ACC延长跟车时距 |
| R5 | ACTIVATE_CURRENT_SPEED | S1/S2 + E键 | 以当前车速启控 |
| R6 | ACTIVATE_INHERITED_SPEED | S1 + Q键 | 以历史速度启控 |

**特点**：
- 这些决策代表**ACC的持续控制策略**
- 用户设定后，ACC会一直执行，直到下一个指令
- 应该被记录到`last_active_decision`，以便无指令时保持

**示例**：
```python
用户按E键降速 → R1 → last_active_decision = R1
松开所有键     → cmd=0 → 返回R1 → ACC继续降速控制
```

---

## R7: 扭矩仲裁（不记录）

**性质**：**临时的驾驶员介入**，不是ACC策略

| 决策 | 名称 | 触发条件 | 含义 |
|------|------|----------|------|
| R7 | TORQUE_ARBITRATION | S0 + W键按住 | 驾驶员油门 + ACC协同 |

**特点**：
- 只在W键**按住期间**有效
- 松开W键立即失效
- 不改变ACC的基础控制策略
- 不应被记录（避免污染ACC决策历史）

**为什么不记录？**

假设R7被记录，会导致：
```python
# 错误场景（如果记录R7）
时刻1: 按E键降速 → R1 → last_active_decision = R1
时刻2: 按W键超车 → R7 → last_active_decision = R7 ✗错误记录
时刻3: 松开W键   → cmd=0 → 返回R7 ✗错误！
  → ACC会认为还在"扭矩仲裁模式"
  → 但manual_throttle_input=0，失去意义
  → ACC不知道应该执行什么策略
```

**正确行为（不记录R7）**：
```python
时刻1: 按E键降速 → R1 → last_active_decision = R1
时刻2: 按W键超车 → R7 → last_active_decision = R1 ✓保持
时刻3: 松开W键   → cmd=0 → 返回R1 ✓正确！
  → ACC回归"降速"策略，继续正常工作
```

---

## R8: 系统待命（不记录）

| 决策 | 名称 | 触发条件 | 含义 |
|------|------|----------|------|
| R8 | SYSTEM_STANDBY | S0 + S/C键 | ACC退出控制 |

**特点**：
- 代表**ACC未激活**状态
- 不是控制决策，而是"无控制"
- 不应被记录

---

## 总结

```
有效的ACC控制决策 (R1-R6)
  ↓
会被记录到 last_active_decision
  ↓
用于 S0 + cmd0 时保持控制策略
  ↓
确保ACC在无新指令时继续执行上一个有效策略

临时介入 (R7)
  ↓
不被记录，不污染决策历史
  ↓
松开后回归上一个有效策略

待命状态 (R8)
  ↓
不被记录，不是控制决策
```

## 代码位置

`acc_controller.py:200-201`:
```python
# 注意：不记录R7(扭矩仲裁)和R8(待命)，只记录R1-R6(有效的ACC控制决策)
if decision >= 1 and decision <= 6:
    self.last_active_decision = decision
```

这个设计确保了**ACC策略的连续性**和**手动介入的临时性**。