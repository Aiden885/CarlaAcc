# CARLA ACC项目技术指南

## 🎯 项目当前状态 (v4.0)

### **最新架构更新**
- ✅ **时距控制升级**: 从距离控制完全替换为直接时距控制（秒）
- ✅ **SPPVT符号自适应**: 实现基于误差符号的动态级差计算
- ✅ **误差符号检测**: 自动检测符号变化并重置控制状态
- ✅ **接口完全兼容**: 所有现有代码接口保持不变

## 🏗️ 核心架构详解

### **时距控制逻辑 (v4.0新特性)**

#### 控制目标变化
```python
# v3.0 (旧版本): 距离控制
desired_distance = ego_speed * G2  # 期望距离随速度变化
distance_error = desired_distance - current_distance
control_output = sppvt_longitudinal_control(distance_error)

# v4.0 (新版本): 时距控制
desired_time_gap = G2  # 期望时距固定(秒)
actual_time_gap = current_distance / ego_speed  # 实际时距
time_error = desired_time_gap - actual_time_gap  # 时间误差
control_output = sppvt_longitudinal_control(time_error)
```

#### SPPVT级差计算更新
```python
# v3.0 (旧版本): 基于控制模式
if control_mode == 'distance':
    new_offset = prev_offset - rho * error_value  # 负级差
else:
    new_offset = prev_offset + rho * error_value  # 正级差

# v4.0 (新版本): 基于误差符号
if error_value > 0:  # 正误差
    new_offset = prev_offset + rho * abs(error_value)  # 增加正级差
else:  # 负误差
    new_offset = prev_offset - rho * abs(error_value)  # 增加负级差
```

#### 误差符号变化检测
```python
def _check_error_sign_change(self, error_value):
    # 计算当前误差符号
    current_sign = 1 if error_value > 0 else (-1 if error_value < 0 else 0)

    # 检测符号变化
    if (self.prev_error_sign != 0 and current_sign != 0 and
        self.prev_error_sign != current_sign):
        # 重置到初始级
        self.stage = 1
        self.target_stages = {1: 0.0}
        self.current_stage_offset = 0.0
        # ... 其他重置操作
```

## 🎛️ SPPVT控制系统详解

### **核心控制原理**
SPPVT (Set Point Pre-Positioning Velocity Tracking) 采用阶段递进控制：

1. **初始级**: stage=1, offset=0（标准比例控制）
2. **升级条件满足**: 增大误差绝对值，让控制更积极
3. **符号变化检测**: 误差符号变化时重置到初始级

### **升级条件**
```python
# 距离/时间跟踪模式
condition1 = acceleration < 0  # 加速度 < 0
condition2 = abs(velocity) <= delta  # 速度 <= δ
condition3 = abs(control_error) > eta  # 控制差 > η

# 速度跟踪模式
condition1 = jerk < 0  # 加加速度 < 0
condition2 = abs(acceleration) <= delta  # 加速度 <= δ
condition3 = abs(control_error) > eta  # 控制差 > η
```

### **参数调优指南**
```python
# SPPVT关键参数
sppvt_kp = 1.0      # 比例控制系数
sppvt_delta = 0.05  # 变目标阈值（接近0的正值）
sppvt_eta = 0.2     # 定速控制精度
sppvt_rho = 0.25    # 惩罚系数 (0 < ρ < 0.5)

# 两模式控制参数
V_threshold = 50/3.6  # 模式切换阈值速度 (m/s)
G2 = 2.0             # 期望时距 (秒)
```

## 🔄 控制流程详解

### **两模式控制切换**
```python
def determine_control_mode(ego_speed):
    if ego_speed <= V_threshold:
        return 'TIME'    # 时距控制模式
    else:
        return 'SPEED'   # 定速控制模式
```

### **TIME模式控制逻辑**
```python
# 有目标车辆时
if current_distance is not None:
    desired_time_gap = G2  # 固定期望时距
    if ego_speed > 0.1:
        actual_time_gap = current_distance / ego_speed
    else:
        actual_time_gap = float('inf')  # 低速保护

    time_error = desired_time_gap - actual_time_gap
    control_output = sppvt_longitudinal_control(time_error)

# 无目标车辆时
else:
    speed_error = target_speed - ego_speed
    control_output = sppvt_longitudinal_control(speed_error)
```

### **SPEED模式控制逻辑**
```python
speed_error = target_speed - ego_speed
set_sppvt_parameters(control_mode='speed')
control_output = sppvt_longitudinal_control(speed_error)
```

## 📊 决策系统详解

### **状态定义**
| 状态 | 符号 | 含义 | 触发条件 |
|------|------|------|----------|
| 在控 | S0 | ACC主动控制 | 启动控制指令 |
| 适速有史待命 | S1 | 有历史数据的待命 | 从S0退出并保存数据 |
| 适速无史待命 | S2 | 无历史数据的待命 | 初始状态或清除历史 |
| 低速 | S3 | 低速待命 | 车速 < V_min_kmh |

### **指令处理**
| 指令 | 符号 | 键盘 | 功能 | 优先级 |
|------|------|------|------|--------|
| 降速/启控 | I0 | E | 降速或当前速度启控 | 机驾执行 |
| 增速/启控 | I1 | Q | 增速或继承历史启控 | 机驾执行 |
| 减距 | I2 | T | 减少跟车距离 | 机驾执行 |
| 增距 | I3 | R | 增加跟车距离 | 机驾执行 |
| 油门 | I4 | W | 人工加速 | 人驾优先 |
| 刹车 | I5 | S | 人工减速 | 人驾优先 |
| 取消 | I6 | C | 取消ACC | 无 |

### **决策输出**
| 决策 | 符号 | 功能 | 状态变化 |
|------|------|------|----------|
| 速度降低 | R1 | 目标速度减少 | 保持S0 |
| 速度增加 | R2 | 目标速度增加 | 保持S0 |
| 时距降低 | R3 | 目标距离减少 | 保持S0 |
| 时距增加 | R4 | 目标距离增加 | 保持S0 |
| 无继控制 | R5 | 当前速度启控 | 进入S0 |
| 继承控制 | R6 | 历史设置启控 | 进入S0 |
| 扭矩仲裁 | R7 | 人驾协调控制 | 保持S0 |
| 系统待命 | R8 | 进入待命状态 | S0→S1 |

## 🛠️ 调试与监控

### **调试信息输出**
```python
# 启用SPPVT调试
enable_sppvt_debug(True)

# 启用两模式调试
enable_two_mode_debug(True)

# 获取状态信息
sppvt_status = get_sppvt_status()
two_mode_status = get_two_mode_status()
```

### **关键调试输出格式**
```
SPPVT Stage 2(distance)[Python]:
原始时间误差=0.300s, 级差=0.075s, 扩大误差=0.375s, 控制输出=0.38m/s²

检测到误差符号变化: 正 → 负, 重置到初始级

升级到第3级: 负误差(-0.200) → 增加负级差 0.050 → -0.100, 增量=0.150, 总升级次数=2
```

### **状态监控要点**
- 监控误差符号变化频率
- 检查SPPVT升级次数和阶段稳定性
- 验证时距控制的收敛性能
- 观察模式切换的平滑性

## 🚀 性能优化建议

### **SPPVT参数调优**
1. **kp调优**: 影响响应速度，建议0.5-2.0范围
2. **delta调优**: 影响升级敏感度，建议0.01-0.1范围
3. **eta调优**: 影响升级阈值，建议0.1-0.5范围
4. **rho调优**: 影响级差增量，建议0.1-0.4范围

### **两模式参数调优**
1. **V_threshold**: 模式切换点，影响控制策略选择
2. **G2**: 期望时距，影响跟车距离，建议1.5-3.0秒
3. **模式切换缓冲**: 可添加滞后防止频繁切换

### **异常处理机制**
```python
# 低速除零保护
if ego_speed < 0.1:
    actual_time_gap = float('inf')

# 控制输出限制
control_output = np.clip(control_output, max_decel, max_accel)

# 模式切换时状态重置
if mode != self.current_mode:
    reset_sppvt_controller()
```

## 📈 测试与验证

### **单元测试**
```python
# 测试时距控制逻辑
def test_time_gap_control():
    controller = TwoModeController()
    ego_speed = 10.0  # m/s
    current_distance = 15.0  # m

    control_output, info = controller.calculate_control_output(
        ego_speed, current_distance)

    assert info['mode'] == 'TIME'
    assert abs(info['error'] - 0.5) < 0.1  # 期望2s，实际1.5s

# 测试SPPVT符号检测
def test_sppvt_sign_change():
    controller = SPPVTLongitudinalController()

    # 正误差
    result1 = controller.sppvt_longitudinal_control(1.0)

    # 负误差 - 应该重置
    result2 = controller.sppvt_longitudinal_control(-1.0)

    assert controller.stage == 1  # 重置到初始级
```

### **集成测试场景**
1. **跟车场景**: 测试时距控制的稳定性
2. **模式切换**: 测试TIME/SPEED模式的平滑切换
3. **符号变化**: 测试误差符号变化时的重置机制
4. **极限工况**: 测试低速、高速边界条件

## 🔧 故障排除

### **常见问题与解决方案**

**1. 控制振荡**
- 检查SPPVT参数是否合理
- 降低kp值或增大delta值
- 检查误差符号是否频繁变化

**2. 响应速度慢**
- 增大kp值
- 减小eta值以更容易升级
- 检查时距计算是否正确

**3. 模式切换不平滑**
- 添加速度滞后缓冲区
- 检查V_threshold设置
- 确认重置机制正常工作

**4. 时距控制不准确**
- 验证时距计算公式
- 检查ego_speed的单位和精度
- 确认除零保护正常工作

---

*文档版本: v4.0*
*最后更新: 2024-09-23*
*适用于: 时距控制架构升级版本*