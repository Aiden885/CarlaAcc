# ACC 纵向限制实现原理说明（当前版本）

本文说明当前版本 ACC 纵向限制的实现原理，重点解释公式来源、变量意义与功能目的。
所有限制在“扭矩指令层”完成，保证符合国标平均约束，同时尽量减少控制震荡。

---

## 1. 信号与总体流程

**输入：**
- `torque_cmd`：纵向控制器输出的发动机扭矩指令（N·m，正值驱动/负值制动）。
- `speed_ms`：车辆纵向速度（m/s）。
- `accel_ms2`：纵向加速度（m/s^2）。优先使用 CARLA 实测加速度向前投影值；不可用时用速度差分估算。

**流程：**
1. 根据加速度信号更新滑动窗口（只对正加速度/负加速度分别统计）。
2. 计算窗口平均值：
   - 正加速度平均（2 s）
   - 负加速度平均（2 s）
   - 减速 jerk 平均（1 s）
3. 如果平均值超过阈值，就限制扭矩指令。

---

## 2. 关键物理关系与变量定义

### 2.1 基本动力学

牛顿第二定律：

$$
F_{\text{total}} = m a
$$

车辆纵向动力学中，牵引力需要同时克服阻力，所以：

$$
\begin{aligned}
F_{\text{drive}} - F_{\text{resist}} &= m a \\
F_{\text{drive}} &= m a + F_{\text{resist}}
\end{aligned}
$$

### 2.2 阻力模型

阻力由滚阻与空气阻力组成：

$$
F_{\text{resist}} = F_{\text{roll}} + F_{\text{drag}}
$$

$$
F_{\text{roll}} = c_{rr} m g, \qquad
F_{\text{drag}} = \tfrac{1}{2}\rho (C_d A) v^2
$$

**变量说明：**
- `c_rr`：滚阻系数（无量纲）
- `m`：车辆质量（kg）
- `g`：重力加速度（9.80665 m/s^2）
- `rho`：空气密度（kg/m^3）
- `CdA`：风阻系数与迎风面积的乘积（m^2）
- `v`：车速（m/s）

### 2.3 轮端扭矩与发动机扭矩关系

轮端扭矩与驱动力关系：

$$
T_{\text{wheel}} = F_{\text{drive}}\, r
$$

传动系统放大关系：

$$
T_{\text{engine}} = \frac{T_{\text{wheel}}}{G\,\eta}
$$

其中 $G$ 为总传动比（对应变量 `gear_ratio`），$\eta$ 为传动效率（对应 `driveline_efficiency`）。

**变量说明：**
- `r`：车轮半径（m）
- `gear_ratio`：总传动比（无量纲）
- `eta`：传动效率（0~1，<1 表示损耗）

### 2.4 等效质量因子

考虑轮胎、传动系转动惯量，使用等效质量放大系数：

$$
m_{\text{eff}} = k_{\text{eff}}\, m
$$

**变量说明：**
- `k_eff`：等效质量系数（>=1），用于提高同等加速度下的所需扭矩

---

## 3. 加速度上限的扭矩推导（加速侧）

国标限制的核心：平均加速度不得超过 `a_max`。
当平均加速度超限时，计算允许的最大驱动力：

$$
F_{\text{drive,max}} = m_{\text{eff}}\, a_{\max} + F_{\text{resist}}
$$

转成发动机扭矩上限：

$$
T_{\max} = \frac{F_{\text{drive,max}}\, r}{G\,\eta}
$$

**作用：**
该 `T_max` 是在考虑阻力与传动损耗后，为实现目标加速度上限所需的“最大允许发动机扭矩”。

---

## 4. 减速度上限的扭矩推导（制动侧）

减速时，阻力本身会帮助减速，所以最大允许制动力需要扣掉阻力贡献：

$$
F_{\text{brake,max}} = m\, a_{\text{decel,avg,max}} - F_{\text{resist}}
$$

$$
F_{\text{brake,max}} = \max\left(0, F_{\text{brake,max}}\right)
$$

对应的最大制动扭矩：

$$
T_{\text{brake,max}} = \frac{F_{\text{brake,max}}\, r}{G\,\eta}
$$

**作用：**
保证“平均减速度”不超过上限，同时不会过度制动。

---

## 5. 平均 jerk（减速度变化率）限制推导

jerk 定义：

$$
j = \frac{d a}{d t}
$$

由动力学关系（忽略阻力变化）：

$$
F = m_{\text{eff}}\, a
$$

$$
T = \frac{F\, r}{G\,\eta}
$$

因此加速度变化对应的扭矩变化：

$$
\Delta T = \frac{m_{\text{eff}}\, \Delta a \, r}{G\,\eta}
$$

在离散时间步长 `dt` 下：

$$
\Delta a_{\max} = j_{\max}\, \Delta t
$$

$$
\Delta T_{\max} =
\frac{m_{\text{eff}}\, j_{\max}\, r}{G\,\eta}\, \Delta t
$$

**作用：**
限制“扭矩变化速度”，从而限制平均 jerk，避免强烈制动冲击。

---

## 6. 滑动平均与触发条件

当前采用“滑动窗口平均值”的方式满足国标要求：

- **加速平均窗口**（默认 2 s）  
  只统计 `a > 0` 样本，得到 `avg_accel`

- **减速平均窗口**（默认 2 s）  
  只统计 `a < 0` 样本（取绝对值），得到 `avg_decel`

- **jerk 平均窗口**（默认 1 s）  
  只在减速阶段统计，得到 `avg_jerk`

**触发条件：**
- 加速限幅：`avg_accel > accel_avg_max` 且 `torque_cmd > 0`
- 减速限幅：`avg_decel > decel_avg_max` 且 `torque_cmd < 0`
- jerk 限幅：`avg_jerk > jerk_avg_max` 且 `torque_cmd < 0`

---

## 7. 软限幅与硬限幅

### 7.1 加速侧软限幅
触发后不直接硬截断，而是缓慢拉回：

$$
T \leftarrow T + \alpha \left(T_{\max} - T\right)
$$

**意义：**
避免瞬时硬限幅导致控制震荡，允许短时超调。

### 7.2 减速侧硬限幅
减速侧采用硬夹紧，确保安全性与国标约束：

$$
T \leftarrow \max\left(T, -T_{\text{brake,max}}\right)
$$

---

## 8. 变量单位与意义汇总

| 变量 | 单位 | 含义 |
|------|------|------|
| `torque_cmd` | N·m | 控制器输出发动机扭矩指令 |
| `speed_ms` | m/s | 车辆纵向速度 |
| `accel_ms2` | m/s^2 | 车辆纵向加速度 |
| `a_max` | m/s^2 | 加速平均上限 |
| `a_decel_avg_max` | m/s^2 | 减速平均上限 |
| `j_max` | m/s^3 | 平均 jerk 上限 |
| `m` | kg | 车辆质量 |
| `k_eff` | - | 等效质量系数 |
| `m_eff` | kg | 等效质量 |
| `r` | m | 车轮半径 |
| `gear_ratio` | - | 总传动比 |
| `eta` | - | 传动效率 |
| `c_rr` | - | 滚阻系数 |
| `rho` | kg/m^3 | 空气密度 |
| `CdA` | m^2 | 风阻系数与面积乘积 |

---

## 9. 当前实现使用的具体数值与“实际”差异

下面列出文档中涉及的关键参数，并给出**当前实现取值**与**CARLA 车辆真实参数（若可获取）**的差异。

> 说明：  
> - “CARLA 实际值”来自 `vehicle.get_physics_control()`（见 `vehicle_physics_params.json` 与 `torque_to_throttle_converter.py`）。  
> - 对于 CARLA 未提供的物理参数（如滚阻、CdA、传动效率），只能作为**经验值**，与真实值差异未知。  
> - 国标限制（a_max, decel_avg, jerk）不是车辆“物理真实值”，而是**法规约束值**，不进行“实际差异”比较。

| 参数 | 当前实现值 | CARLA 实际值 | 差异 | 来源/说明 |
|------|------------|--------------|------|-----------|
| 车辆质量 $m$ | 2370 kg | 2370 kg | 0 | CARLA 物理参数 |
| 轮半径 $r$ | 0.37 m | 0.37 m | 0 | CARLA 物理参数 |
| 总传动比 $G$ | 9.204 | 9.204 | 0 | CARLA 物理参数（单档） |
| 传动效率 $\eta$ | 0.90 | N/A | N/A | 经验值（电驱常见 0.90~0.95） |
| 等效质量系数 $k_{\text{eff}}$ | 1.08 | N/A | N/A | 经验值（考虑转动惯量） |
| 滚阻系数 $c_{rr}$ | 0.012 | N/A | N/A | CARLA 未提供，经验值 |
| 风阻 $C_dA$ | 0.74 m^2 | N/A | N/A | CARLA 未提供，经验值 |
| 空气密度 $\rho$ | 1.225 kg/m^3 | N/A | N/A | 标准海平面值 |
| 加速平均上限 $a_{\max}$ | 2.0 m/s^2 | N/A | N/A | 国标限制值 |
| 减速平均上限 $a_{\text{decel,avg,max}}$ | 3.0 m/s^2 | N/A | N/A | 国标限制值 |
| jerk 平均上限 $j_{\max}$ | 2.5 m/s^3 | N/A | N/A | 国标限制值 |
| 加速平均窗口 | 2.0 s | N/A | N/A | 国标/实现选取 |
| 减速平均窗口 | 2.0 s | N/A | N/A | 国标/实现选取 |
| jerk 平均窗口 | 1.0 s | N/A | N/A | 国标/实现选取 |


---
