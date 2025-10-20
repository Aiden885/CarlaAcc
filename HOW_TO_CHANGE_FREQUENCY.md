# 如何修改 Simulink 模型频率

## 方法1：使用MATLAB脚本（推荐）⭐

### 快速使用
```matlab
% 在MATLAB命令窗口中运行：
change_model_frequency(20)   % 修改为20Hz (0.05秒)
change_model_frequency(50)   % 修改为50Hz (0.02秒) - 当前值
change_model_frequency(100)  % 修改为100Hz (0.01秒)
```

### 优点
- ✅ 自动验证修改结果
- ✅ 自动保存模型
- ✅ 显示修改前后对比
- ✅ 安全可靠

---

## 方法2：使用MATLAB命令行

### 步骤
```matlab
% 1. 加载模型
load_system('ACC_Decision_SPPVT_Integrated');

% 2. 设置固定步长（例如：20Hz = 0.05秒）
set_param('ACC_Decision_SPPVT_Integrated', 'FixedStep', '0.05');

% 3. 验证
fprintf('新频率: %.2f Hz\n', 1/str2double(get_param('ACC_Decision_SPPVT_Integrated', 'FixedStep')));

% 4. 保存
save_system('ACC_Decision_SPPVT_Integrated');
close_system('ACC_Decision_SPPVT_Integrated', 0);
```

### 常用频率对应的固定步长
| 频率 | 固定步长 | 说明 |
|------|---------|------|
| 10 Hz | 0.1000 秒 | 低频，计算负担小 |
| 20 Hz | 0.0500 秒 | 标准控制频率 |
| 50 Hz | 0.0200 秒 | 当前设置，较高精度 |
| 100 Hz | 0.0100 秒 | 高频，精度高但计算密集 |
| 200 Hz | 0.0050 秒 | 超高频 |

---

## 方法3：在Simulink图形界面中修改

### 步骤
1. **打开模型**
   ```matlab
   open_system('ACC_Decision_SPPVT_Integrated')
   ```

2. **打开配置参数对话框**
   - 方法A：菜单栏 → `Modeling` → `Model Settings` → `Model Properties`
   - 方法B：快捷键 `Ctrl + E`
   - 方法C：在模型窗口空白处双击

3. **修改求解器配置**
   - 导航到：`Solver` 选项卡
   - 确保 `Type` 设置为：`Fixed-step`
   - 修改 `Fixed-step size` 的值：
     - 输入 `0.05` → 20Hz
     - 输入 `0.02` → 50Hz
     - 输入 `0.01` → 100Hz

4. **设置仿真时间**
   - `Start time`: `0.0`
   - `Stop time`: `inf` (或根据需要设置具体时间)

5. **应用并保存**
   - 点击 `OK` 或 `Apply`
   - 保存模型：`Ctrl + S`

### 示意图位置
```
Configuration Parameters
├── Solver
│   ├── Solver selection
│   │   ├── Type: [Fixed-step] ← 确保选择这个
│   │   └── Solver: ode3 (Bogacki-Shampine)
│   ├── Solver details
│   │   └── Fixed-step size: [0.02] ← 在这里修改频率
│   └── Tasking and sample time options
│       └── ...
```

---

## 方法4：批量修改脚本

如果你需要测试多个频率，可以使用这个脚本：

```matlab
% test_multiple_frequencies.m
frequencies = [10, 20, 50, 100];  % 要测试的频率

for i = 1:length(frequencies)
    freq = frequencies(i);
    fprintf('\n测试频率: %d Hz\n', freq);

    change_model_frequency(freq);

    % 在这里运行仿真测试
    % sim('ACC_Decision_SPPVT_Integrated');

    pause(1);  % 等待1秒
end
```

---

## 频率选择建议

### 🎯 根据应用场景选择

| 场景 | 推荐频率 | 原因 |
|------|---------|------|
| **CARLA仿真** | 20-50 Hz | 匹配CARLA的tick频率 |
| **实时控制** | 50-100 Hz | 快速响应，精确控制 |
| **调试/测试** | 10-20 Hz | 便于观察，计算快 |
| **高精度控制** | 100+ Hz | 最高精度，计算密集 |

### ⚠️ 注意事项

1. **频率越高**
   - ✅ 控制精度越高
   - ✅ 响应速度越快
   - ❌ 计算负担越大
   - ❌ 可能导致实时性问题

2. **与Python代码匹配**
   - 如果模型设置为50Hz (0.02秒)
   - Python中的`acc_updated.py`也应该设置相应的控制周期
   - 查找Python代码中的`sleep()`或`tick()`调用

3. **CARLA集成**
   - CARLA的默认tick通常是50Hz或60Hz
   - 建议模型频率与CARLA tick频率匹配
   - 在`acc_updated.py`中检查：
     ```python
     # 查找类似这样的代码
     world.tick()  # 或
     time.sleep(0.02)  # 0.02秒 = 50Hz
     ```

---

## 验证频率修改

### 方法1：使用提供的脚本
```matlab
check_model_frequency  % 运行之前创建的检查脚本
```

### 方法2：手动验证
```matlab
load_system('ACC_Decision_SPPVT_Integrated');
step = str2double(get_param('ACC_Decision_SPPVT_Integrated', 'FixedStep'));
freq = 1.0 / step;
fprintf('当前频率: %.2f Hz (步长: %.4f 秒)\n', freq, step);
close_system('ACC_Decision_SPPVT_Integrated', 0);
```

---

## 常见问题

### Q1: 修改频率后模型运行出错？
**A:** 检查以下几点：
- 确保所有子模块的SampleTime设置为`-1`（继承）
- 确保StopTime不要设置得太小（建议`inf`或具体仿真时长）
- 重新编译模型（如果使用代码生成）

### Q2: 频率修改后Python接口不工作？
**A:** 需要同步修改Python端的控制周期：
```python
# 在 acc_updated.py 中查找并修改
CONTROL_PERIOD = 0.02  # 修改为与Simulink匹配的值
```

### Q3: 如何恢复原始频率？
**A:**
```matlab
change_model_frequency(50)  % 当前的频率
# 或
change_model_frequency(20)  % build脚本原始的频率
```

---

## 实际示例

### 示例1：修改为20Hz并测试
```matlab
% 1. 修改频率
change_model_frequency(20);

% 2. 运行测试
test_integrated_sppvt_model();

% 3. 检查结果
check_model_frequency();
```

### 示例2：优化CARLA集成
```matlab
% CARLA通常以50Hz运行
change_model_frequency(50);

% 然后在acc_updated.py中设置匹配的周期
% CONTROL_PERIOD = 0.02  # 50Hz
```

---

## 总结

**推荐使用方法1（脚本）**，只需一行命令：
```matlab
change_model_frequency(你想要的频率)
```

简单、安全、自动验证！