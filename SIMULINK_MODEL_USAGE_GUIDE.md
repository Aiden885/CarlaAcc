# ACC决策+SPPVT一体化Simulink模型使用指南

## 📋 概述

本指南详细说明了如何使用ACC决策+SPPVT一体化Simulink模型，该模型适配MATLAB 2024b版本，实现了完整的自适应巡航控制决策逻辑与SPPVT纵向控制的集成。

## 🚀 快速开始

### 1. 环境要求
- MATLAB 2024b 或更高版本
- Simulink 工具箱
- 推荐: Simulink Coder（用于代码生成）

### 2. 文件清单
确保以下文件位于同一目录：

**MATLAB核心文件：**
- `create_decision_sppvt_bus.m` - 总线定义脚本
- `decision_function.m` - 决策逻辑MATLAB函数
- `sppvt_function.m` - SPPVT控制算法
- `build_acc_decision_sppvt_model.m` - 模型构建脚本
- `load_acc_sppvt_parameters.m` - 参数配置加载脚本
- `test_acc_decision_sppvt_model.m` - 完整测试套件

**Python接口文件：**
- `acc_decision_sppvt_interface.py` - Python一体化接口
- `acc_updated.py` - 修改后的主控制器
- `two_mode_controller.py` - 增强的两模式控制器

**文档文件：**
- `decision.md` - 决策逻辑说明文档
- `SIMULINK_MODEL_USAGE_GUIDE.md` - 本使用指南
- `MODIFICATION_PLAN_TRACKER.md` - 项目修改跟踪

### 3. 🎯 标准运行顺序

**按以下顺序执行MATLAB脚本：**

```matlab
% === 第一步：基础环境准备 ===
% 1. 创建总线定义（必须首先执行）
create_decision_sppvt_bus();

% 2. 加载参数配置（第二步执行）
load_acc_sppvt_parameters();

% === 第二步：模型构建 ===
% 3. 构建Simulink模型（在总线和参数准备后）
build_acc_decision_sppvt_model();

% === 第三步：测试验证 ===
% 4. 运行完整测试套件（最后执行）
test_acc_decision_sppvt_model();
```

**⚠️ 重要说明：**
- **必须严格按照顺序执行**，因为后续脚本依赖前面的结果
- 如果中间有错误，请从出错的步骤重新开始
- 每个步骤完成后会有成功提示

### 4. 一键快速部署

如果所有文件就绪，可以用一条命令完成全部部署：

```matlab
% 一键部署（按顺序执行所有步骤）
create_decision_sppvt_bus(); load_acc_sppvt_parameters(); build_acc_decision_sppvt_model(); test_acc_decision_sppvt_model();
```

## 🔧 详细使用步骤

### 步骤1: 创建总线定义

```matlab
create_decision_sppvt_bus();
```

这将创建两个总线对象：
- **DecisionSPPVTInput** (11个字段): Python到Simulink的输入接口
- **DecisionSPPVTOutput** (10个字段): Simulink到Python的输出接口

### 步骤2: 构建Simulink模型

```matlab
build_acc_decision_sppvt_model();
```

自动创建的Simulink模型包含：

**主要模块：**
- **Input_Validator**: 输入数据验证和预处理
- **Decision_Function**: ACC决策状态机（S0-S3状态，R1-R8决策）
- **SPPVT_Function**: SPPVT纵向控制算法
- **Output_Formatter**: 输出数据格式化
- **Parameter_Manager**: 参数管理子系统
- **Debug_Display**: 调试信息显示
- **State_Monitor**: 状态监控示波器

### 步骤3: 参数配置

```matlab
% 使用修复后的参数加载脚本
load_acc_sppvt_parameters();
```

**关键参数：**

**决策参数：**
- `V_target_default_kmh = 50.0` - 默认目标速度
- `G2_default_s = 2.0` - 默认时距参数
- `min_activation_speed_kmh = 30.0` - 最低启控速度

**SPPVT参数：**
- `kp = 0.8, ki = 0.1, kd = 0.05` - PID控制增益
- `max_accel_ms2 = 2.0` - 最大加速度限制
- `max_decel_ms2 = -4.0` - 最大减速度限制

### 步骤4: 运行仿真

```matlab
% 打开模型
open_system('ACC_Decision_SPPVT_Integrated');

% 配置仿真时间
set_param('ACC_Decision_SPPVT_Integrated', 'StopTime', '10');

% 运行仿真
sim('ACC_Decision_SPPVT_Integrated');
```

## 📊 输入输出接口规范

### 输入接口 (DecisionSPPVTInput)

```matlab
input_data = struct();
input_data.ego_speed_kmh = 50.0;          % 自车速度 km/h
input_data.ego_speed_ms = 13.89;          % 自车速度 m/s
input_data.command_type = int32(1);       % 指令类型 0-7
input_data.command_active = true;         % 指令激活状态
input_data.manual_throttle_active = false;% 手动油门状态
input_data.control_error = 0.5;           % 两模式控制误差
input_data.control_mode_flag = int32(1);  % 1=距离模式, 2=速度模式
input_data.V_target_kmh = 50.0;           % 目标速度
input_data.V_min_kmh = 30.0;              % 最小速度
input_data.G2_s = 2.0;                    % 时距参数
input_data.timestamp = now();             % 时间戳
```

### 输出接口 (DecisionSPPVTOutput)

```matlab
% 输出示例
output_data.target_accel = 0.5;           % SPPVT目标加速度 m/s²
output_data.control_enabled = true;       % 控制使能状态
output_data.current_state = int32(1);     % 当前状态 (S0-S3)
output_data.current_decision = int32(2);  % 当前决策 (R0-R8)
output_data.torque_arbitration_active = false; % 扭矩仲裁状态
output_data.updated_V_target_kmh = 50.0;  % 更新的目标速度
output_data.updated_G2_s = 2.0;           % 更新的时距参数
output_data.sppvt_stage = int32(1);       % SPPVT阶段
output_data.sppvt_upgrade_count = int32(0); % SPPVT升级次数
output_data.debug_message = int32(1001);  % 调试信息代码
```

## 🔍 调试和监控

### 1. 调试信息解读

**决策调试代码格式**: `1000 + 状态*10 + 决策`
- 1011: 状态S1，决策R1
- 1023: 状态S2，决策R3

**SPPVT调试代码格式**: `2000 + 阶段*10 + (升级次数 mod 10)`
- 2010: 阶段1，升级次数0
- 2021: 阶段2，升级次数1

### 2. 状态监控

使用State_Monitor示波器观察：
- **信号1**: 决策状态和决策输出
- **信号2**: SPPVT控制输出和阶段
- **信号3**: 最终输出信号

### 3. 性能监控

```matlab
% 检查模型性能
[perf_data] = sldiagnostics('ACC_Decision_SPPVT_Integrated', 'PerformanceAdvisor');
```

## ⚙️ 高级配置

### 1. 自定义参数

```matlab
% 创建自定义参数结构
custom_params = acc_sppvt_parameters();
custom_params.sppvt.kp = 1.0;  % 修改比例增益
custom_params.decision.V_target_default_kmh = 60.0; % 修改目标速度

% 应用到模型
assignin('base', 'ModelParams', custom_params);
```

### 2. 添加自定义功能

在Simulink模型中添加自定义MATLAB Function块：

```matlab
function output = custom_logic(input, decision, sppvt)
%CUSTOM_LOGIC 自定义控制逻辑
    output = input;
    
    % 添加自定义逻辑
    if decision.current_state == 1 && sppvt.target_accel > 1.5
        output.safety_override = true;
    else
        output.safety_override = false;
    end
end
```

### 3. 代码生成配置

```matlab
% 配置代码生成
set_param('ACC_Decision_SPPVT_Integrated', 'RTWSystemTargetFile', 'grt.tlc');
set_param('ACC_Decision_SPPVT_Integrated', 'GenerateReport', 'on');

% 生成C代码
rtwbuild('ACC_Decision_SPPVT_Integrated');
```

## 🐛 常见问题解决

### 问题1: 总线定义错误

**症状**: "Bus object 'DecisionSPPVTInput' not found"

**解决方案**:
```matlab
clear all;
create_decision_sppvt_bus();
bdclose('all');  % 关闭所有模型后重新打开
```

### 问题2: MATLAB Function编译错误

**症状**: "Undefined function or variable"

**解决方案**:
1. 检查persistent变量初始化
2. 确保所有变量类型一致
3. 添加`%#codegen`指令

### 问题3: 仿真运行缓慢

**症状**: 仿真速度过慢

**解决方案**:
```matlab
% 优化求解器设置
set_param('ACC_Decision_SPPVT_Integrated', 'SolverType', 'Fixed-step');
set_param('ACC_Decision_SPPVT_Integrated', 'FixedStep', '0.05');
set_param('ACC_Decision_SPPVT_Integrated', 'OptimizeBlockIOStorage', 'on');
```

### 问题4: 与Python接口通信失败

**症状**: 数据传输错误或格式不匹配

**解决方案**:
1. 验证数据类型匹配（int32 vs double）
2. 检查数组维度
3. 确保时间戳格式一致

## 🧪 测试验证

### 完整测试流程

```matlab
% 运行完整测试套件
test_acc_decision_sppvt_model();
```

### 单元测试

```matlab
% 仅测试决策逻辑
test_results = test_decision_logic();

% 仅测试SPPVT控制
test_results = test_sppvt_control();
```

### 性能基准测试

```matlab
% 性能测试
performance_results = test_performance();
fprintf('平均执行时间: %.2f ms\n', performance_results.avg_execution_time_ms);
```

## 📚 参考资料

### 决策状态机
- **S0**: 待命状态 - 等待启控指令
- **S1**: 主动控制状态 - 正常ACC控制
- **S2**: 自适应历史待命状态 - 保持历史参数
- **S3**: 扭矩仲裁状态 - 驾驶员与ACC协调

### 指令映射
- **I0 (1)**: 降速/当速启控
- **I1 (2)**: 增速/继承启控
- **I2 (3)**: 减距控制
- **I3 (4)**: 增距控制
- **I4 (5)**: 油门指令
- **I5 (6)**: 刹车指令
- **I6 (7)**: 取消ACC

### SPPVT阶段
- **阶段1**: 基础PID控制
- **阶段2**: 增强预设定控制
- **阶段3**: 高级自适应控制

## 🔄 版本更新

**当前版本**: 2.0
- 支持MATLAB 2024b
- 完整的决策+SPPVT集成
- 增强的调试功能
- 自动化测试套件

**历史版本**:
- v1.0: 基础Python实现
- v1.5: Simulink集成开始


生成详细错误报告：
```matlab
generate_error_report();
```