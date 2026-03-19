# Simulink决策+控制测试指南

## 概述

本测试框架用于完整验证 `ACC_Decision_SPPVT_Integrated.slx` 的:
1. **状态转移逻辑** (S0-S3状态机)
2. **决策输出正确性** (R1-R8决策规则)
3. **SPPVT控制输出** (升级条件和控制性能)


## 测试文件说明

### 1. `test_simulink_decision_control.py`
**基础测试框架** - 命令行输出，适合快速验证

**功能:**
- 覆盖所有状态转移路径(S0↔S1↔S2, S3)
- 验证所有决策输出(R1-R8)
- 测试SPPVT升级机制
- 命令行详细输出


**输出示例:**
```
🧪 测试: S2_I0_to_S0_R5
📝 描述: S2状态下按E键(I0降速) -> 当速启控
📥 输入条件:
   - 车速: 45.0 km/h
   - 指令: I0 (code=1)
   - 激活: True
📤 Simulink输出:
   - 状态: S0
   - 决策: R5
   - 控制使能: True
✅ 测试通过!
```

---

### 2. `test_simulink_with_visualization.py`
**可视化测试框架** - 生成图表和HTML报告，适合完整分析

**功能:**
- 包含基础测试的所有功能
- 生成状态转移序列图
- 生成决策分布统计图
- 生成状态转移覆盖矩阵
- 生成交互式HTML测试报告


**生成文件:**
- `simulink_test_report.html` - 交互式HTML报告(推荐优先查看)
- `simulink_state_transitions.png` - 状态转移序列图
- `simulink_decision_distribution.png` - 决策分布图
- `simulink_test_coverage.png` - 状态转移覆盖矩阵

---

## 测试用例详解

### 第一部分: 状态转移逻辑测试

#### 测试组1: S2 -> S0 (无史启控)
```python
测试场景: 车辆处于S2(适速无史待命)状态,按E键(I0降速)
预期结果: 转移到S0(在控)状态,决策R5(无继控制)
验证内容: 当速启控功能
```

#### 测试组2: S0 -> S1 (退出控制)
```python
测试场景: 车辆处于S0(在控)状态,按C键(I6取消)或S键(I5刹车)
预期结果: 转移到S1(有史待命)状态,决策R8(待命)
验证内容: 退出控制并保存历史
```

#### 测试组3: S1 -> S0 (继承启控)
```python
测试场景: 车辆处于S1(有史待命)状态,按Q键(I1增速)
预期结果: 转移到S0(在控)状态,决策R6(继承控制)
验证内容: 继承历史速度启控
```

#### 测试组4: S0状态内的决策
```python
测试场景: 在S0状态下测试所有调整指令
- I0降速 -> R1(速度降低)
- I1增速 -> R2(速度增加)
- I2降距 -> R3(时距降低)
- I3增距 -> R4(时距增加)
- I4油门 -> R7(扭矩仲裁)
```

#### 测试组5: S3低速状态
```python
测试场景: 车速 < V_min_kmh
预期结果: 自动进入S3状态,任何指令都保持R8待命
验证内容: 低速保护逻辑
```

---

### 第二部分: 决策输出测试(R1-R8)

| 决策 | 触发条件 | 验证方法 |
|-----|---------|---------|
| R1 | S0+I0降速 | 检查目标速度是否降低 |
| R2 | S0+I1增速 | 检查目标速度是否增加 |
| R3 | S0+I2降距 | 检查时距参数是否减小 |
| R4 | S0+I3增距 | 检查时距参数是否增加 |
| R5 | S2+I0降速 | 检查当速启控(V_target=当前速度) |
| R6 | S1+I1增速 | 检查继承启控(V_target=历史速度) |
| R7 | S0+I4油门 | 检查扭矩仲裁激活 |
| R8 | S1/S2/S3+其他指令 | 检查系统待命 |

---

### 第三部分: SPPVT控制输出测试

#### 3.1 时距控制模式(control_mode_flag=1)
```python
测试场景: 时距误差 = 1.5秒
验证内容:
- 控制输出在合理范围[-4.0, 3.0] m/s²
- 误差符号正确对应加减速
```

#### 3.2 速度控制模式(control_mode_flag=2)
```python
测试场景: 速度误差 = 2.5 m/s
验证内容:
- 控制输出在合理范围[-4.0, 3.0] m/s²
- 误差符号正确对应加减速
```

#### 3.3 SPPVT升级机制
```python
测试场景: 连续输入稳定误差
验证条件:
- 二阶导数 < 0
- abs(一阶导数) < delta (0.05)
- abs(误差) > eta (0.2)
验证内容: 观察级差是否正确升级
```

---

## 测试结果解读

### 命令行输出
```
📊 测试总结报告
==================================================
总测试数: 15
✅ 通过: 14
❌ 失败: 1
通过率: 93.3%

❌ 失败的测试:
   - S0_I3_to_S0_R4


### 可视化报告解读

#### 1. 状态转移序列图
- **X轴**: 测试序列编号
- **Y轴**: 状态(S0-S3)
- **标注**: 每个点显示状态名称和决策
- **颜色**: 绿色=测试通过, 红色=测试失败

#### 2. 决策分布图
- **左图**: 各决策(R1-R8)的触发次数统计
- **右图**: SPPVT控制输出时间序列
  - 绿线: 控制输出
  - 红虚线: 上下限(2.0/-3.0 m/s²)

#### 3. 状态转移覆盖矩阵
- **行**: 源状态
- **列**: 目标状态
- **颜色深度**: 转移次数(越深次数越多)
- **标注**: 显示触发该转移的指令

---

## 常见问题排查

### Q1: 测试失败 - 状态转移不正确
**可能原因:**
1. Simulink的`decision_function.m`逻辑与设计不符
2. 初始状态设置错误
3. 输入参数(V_min, V_target)配置不合理



### Q2: 测试失败 - 决策输出不正确
**可能原因:**
1. 指令码映射错误(command_type)
2. decision_function.m的决策表有误

**验证方法:**
查看测试输出中的"📥 输入条件"和"📤 Simulink输出",对比`decision.md`

### Q3: SPPVT控制输出异常
**可能原因:**
1. 控制模式标志(control_mode_flag)设置错误
2. 升级条件参数(delta, eta)配置不合理
3. SPPVT算法实现有问题

**检查点:**
- 控制输出是否在[-4.0, 3.0]范围内
- 误差符号与加减速方向是否一致

---

## 自定义测试用例

### 添加新测试
```python
# 在SimulinkDecisionControlTester类中添加方法
def test_custom_scenario(self):
    """自定义测试场景"""
    self.interface.reset()

    # 设置初始状态
    init_input = self.create_test_input(
        ego_speed_kmh=40.0,
        command_type=1,  # I0降速
        command_active=True
    )
    self.interface.process_decision_and_control(init_input)

    # 执行测试
    self.run_single_test(
        "Custom_Test",
        self.create_test_input(
            ego_speed_kmh=40.0,
            command_type=2,  # I1增速
            command_active=True,
            control_error=1.0,
            V_target_kmh=50.0
        ),
        expected_state=0,  # 期望状态
        expected_decision=2,  # 期望决策
        description="自定义测试场景描述"
    )
```

---

## 指令码参考

| command_type | 指令名称 | 键盘按键 | 含义 |
|-------------|---------|---------|-----|
| 0 | NONE | - | 无指令 |
| 1 | I0 | E键 | 降速/当速启控 |
| 2 | I1 | Q键 | 增速/继承启控 |
| 3 | I2 | T键 | 降距 |
| 4 | I3 | R键 | 增距 |
| 5 | I4 | W键 | 油门(扭矩仲裁) |
| 6 | I5 | S键 | 刹车 |
| 7 | I6 | C键 | 取消ACC |

---

## 测试最佳实践



   ```

3. **查看报告**: 用浏览器打开`simulink_test_report.html`

4. **问题定位**: 如果有失败测试,启用debug模式重新运行
   ```python
   tester = VisualizedSimulinkTester(debug=True)
   ```

5. **迭代优化**: 根据测试结果修改Simulink模型,重新测试

---

## 技术支持

如有问题,请检查:
1. `decision.md` - 决策逻辑设计文档
2. `REALTIME_SIMULINK_INTEGRATION.md` - Simulink集成文档
3. `acc_decision_sppvt_interface.py:192` - 接口实现

---

**最后更新**: 2025-10-10
**适用版本**: ACC_Decision_SPPVT_Integrated.slx v2.0+