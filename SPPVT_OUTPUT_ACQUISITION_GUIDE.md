# ACC决策+SPPVT集成模型输出获取完整指南

## 概述

本文档说明了如何从ACC决策+SPPVT集成Simulink模型中正确获取输出结果。经过详细测试验证，已确认正确的SPPVT输出获取方法。

## 目录
1. [已确认工作方法：SPPVT输出获取](#已确认工作方法sppvt输出获取)
2. [输入数据设计](#输入数据设计)
3. [模型配置要求](#模型配置要求)
4. [决策部分结果获取](#决策部分结果获取)
5. [总线数据处理](#总线数据处理)
6. [故障排除指南](#故障排除指南)

---

## 输入数据设计

### 基础输入数据结构

```matlab
function input_data = create_test_input_data(ego_speed_kmh, control_error, command_type, command_active)
% 创建标准测试输入数据
% 参数:
%   ego_speed_kmh - 自车速度 (km/h)
%   control_error - 控制误差 (m/s^2)
%   command_type - 指令类型 (1=I0, 2=I1, ..., 7=I6)
%   command_active - 指令是否激活 (boolean)

if nargin < 4
    % 默认参数
    ego_speed_kmh = 50.0;
    control_error = 1.5;
    command_type = 1;  % I0指令
    command_active = true;
end

ego_speed_ms = ego_speed_kmh / 3.6;
V_target_kmh = 50.0;
V_min_kmh = 30.0;
G2_s = 2.0;

% 时间序列设置
time_points = [0, 0.05];  % 50ms仿真时间

input_data = struct();
input_data.ego_speed_kmh = timeseries([ego_speed_kmh, ego_speed_kmh], time_points, 'Name', 'ego_speed_kmh');
input_data.ego_speed_ms = timeseries([ego_speed_ms, ego_speed_ms], time_points, 'Name', 'ego_speed_ms');
input_data.command_type = timeseries(int32([command_type, command_type]), time_points, 'Name', 'command_type');
input_data.command_active = timeseries(logical([command_active, command_active]), time_points, 'Name', 'command_active');
input_data.manual_throttle_active = timeseries(logical([false, false]), time_points, 'Name', 'manual_throttle_active');
input_data.control_error = timeseries([control_error, control_error], time_points, 'Name', 'control_error');
input_data.control_mode_flag = timeseries(int32([1, 1]), time_points, 'Name', 'control_mode_flag');
input_data.V_target_kmh = timeseries([V_target_kmh, V_target_kmh], time_points, 'Name', 'V_target_kmh');
input_data.V_min_kmh = timeseries([V_min_kmh, V_min_kmh], time_points, 'Name', 'V_min_kmh');
input_data.G2_s = timeseries([G2_s, G2_s], time_points, 'Name', 'G2_s');
input_data.timestamp = timeseries(time_points, time_points, 'Name', 'timestamp');

% 设置时间单位
field_names = fieldnames(input_data);
for i = 1:length(field_names)
    input_data.(field_names{i}).TimeInfo.Units = 'seconds';
end

end
```

### 特殊测试场景输入设计

#### 1. 决策持续性测试
```matlab
% NONE维持控制测试：先I0进入控制，然后测试无指令的持续性
command_sequence = [1, 1]; % 两步都是I0，但第二步inactive
active_sequence = [true, false]; % 第一步激活进入控制，第二步不激活测试持续性
sim_time = 0.15;

% 创建多步时间序列
time_points = 0:0.05:sim_time;
steps_per_cmd = max(1, floor(length(time_points) / length(command_sequence)));
cmd_data = int32(ones(1, length(time_points)));
act_data = false(1, length(time_points));

for j = 1:length(command_sequence)
    start_idx = (j-1) * steps_per_cmd + 1;
    end_idx = min(j * steps_per_cmd, length(time_points));
    cmd_data(start_idx:end_idx) = int32(command_sequence(j));
    act_data(start_idx:end_idx) = logical(active_sequence(j));
end
```

#### 2. SPPVT控制算法验证
```matlab
% 不同控制误差的SPPVT响应测试
error_values = [3.0, 1.5, 0.0, -1.5, -3.0];

for i = 1:length(error_values)
    test_input = base_input;
    test_input.control_error = timeseries([error_values(i), error_values(i)], [0, 0.05], 'Name', 'control_error');
    test_input.command_type = timeseries(int32([1, 1]), [0, 0.05], 'Name', 'command_type');  % I0指令
    test_input.command_active = timeseries(logical([true, true]), [0, 0.05], 'Name', 'command_active');
    % ... 运行测试
end
```

---

## 模型配置要求

### 必要的预加载步骤

```matlab
% 1. 加载总线定义（必须在模型加载前）
load('DecisionSPPVTBusDefinitions.mat');

% 2. 加载模型
model_name = 'ACC_Decision_SPPVT_Integrated';
if ~bdIsLoaded(model_name)
    load_system(model_name);
end
```

### 仿真参数配置

```matlab
% 基础仿真配置
set_param(model_name, 'StopTime', '0.05');
set_param(model_name, 'LoadExternalInput', 'on');
set_param(model_name, 'SaveOutput', 'on');

% 输出保存配置（多种方法）
method1_dataset = {
    'OutputSaveName', 'yout',
    'SaveFormat', 'Dataset'
};

method2_array = {
    'OutputSaveName', 'yout_array',
    'SaveFormat', 'Array'
};

method3_structure = {
    'OutputSaveName', 'yout_struct',
    'SaveFormat', 'Structure'
};
```

---

## 决策部分结果获取

### 标准访问方法

```matlab
function [state, decision, control_enabled] = extract_decision_results(output_data)
% 从输出数据中提取决策相关结果
% 参数: output_data - Dataset, Array或Structure格式的输出

state = [];
decision = [];
control_enabled = [];

if isa(output_data, 'Simulink.SimulationData.Dataset')
    % Dataset格式访问
    state_element = find(output_data, 'Name', 'current_state');
    if ~isempty(state_element)
        state = double(state_element.Values.Data(end));
    end

    decision_element = find(output_data, 'Name', 'current_decision');
    if ~isempty(decision_element)
        decision = double(decision_element.Values.Data(end));
    end

    control_element = find(output_data, 'Name', 'control_enabled');
    if ~isempty(control_element)
        control_enabled = logical(control_element.Values.Data(end));
    end

elseif isstruct(output_data) && isfield(output_data, 'signals')
    % Structure格式访问
    if length(output_data.signals) >= 3
        state = output_data.signals(2).values(end);  % current_state
        decision = output_data.signals(3).values(end);  % current_decision
        control_enabled = logical(output_data.signals(1).values(end));  % control_enabled
    end

elseif isnumeric(output_data)
    % Array格式访问
    if size(output_data, 2) >= 3
        control_enabled = logical(output_data(end, 1));
        state = output_data(end, 2);
        decision = output_data(end, 3);
    end
end

end
```

### 决策状态映射

```matlab
function state_name = decode_decision_state(state_code)
% 决策状态代码解码
state_map = containers.Map(...
    {0, 1, 2, 3}, ...
    {'S0-在控状态', 'S1-适速有史待命', 'S2-适速无史待命', 'S3-低速状态'});

if isKey(state_map, state_code)
    state_name = state_map(state_code);
else
    state_name = sprintf('未知状态(%d)', state_code);
end
end

function decision_name = decode_decision_type(decision_code)
% 决策类型代码解码
decision_map = containers.Map(...
    {1, 2, 3, 4, 5, 6, 7, 8}, ...
    {'R1-速度降低', 'R2-速度增加', 'R3-时距降低', 'R4-时距增加', ...
     'R5-无继控制', 'R6-继承控制', 'R7-扭矩仲裁', 'R8-系统待命'});

if isKey(decision_map, decision_code)
    decision_name = decision_map(decision_code);
else
    decision_name = sprintf('未知决策(%d)', decision_code);
end
end
```

---

## **已确认工作方法：SPPVT输出获取**

### ✅ 经过验证的正确方法

经过详细测试和调试，已确认从Simulink仿真结果中获取SPPVT输出的正确方法：

```matlab
function sppvt_value = get_sppvt_output_confirmed_method(sim_out)
% 已验证的SPPVT输出获取方法
% 参数: sim_out - sim()函数返回的仿真结果
% 返回: sppvt_value - SPPVT控制输出值

sppvt_value = 0.0;  % 默认值

try
    % 检查仿真结果结构
    if isstruct(sim_out) && isfield(sim_out, 'yout') && ~isempty(sim_out.yout)
        output_data = sim_out.yout;

        % 确认是Simulink Dataset格式
        if isa(output_data, 'Simulink.SimulationData.Dataset') && length(output_data) >= 1
            output_element = output_data{1};

            % 关键发现：Values是包含timeseries对象的结构体
            if isstruct(output_element.Values) && isfield(output_element.Values, 'sppvt_control_output')
                sppvt_ts = output_element.Values.sppvt_control_output;

                % 确认是timeseries对象并提取最终值
                if isa(sppvt_ts, 'timeseries') && ~isempty(sppvt_ts.Data)
                    sppvt_value = double(sppvt_ts.Data(end));
                end
            end
        end
    end

catch ME
    fprintf('警告：SPPVT输出获取失败: %s\n', ME.message);
    sppvt_value = 0.0;
end

end
```

### 🔍 关键技术发现

通过深入调试过程，发现了Simulink Dataset的实际数据结构：

```matlab
% 数据结构层次：
sim_out.yout                          % Simulink.SimulationData.Dataset
    {1}                               % Dataset第一个元素
        .Values                       % 结构体（不是Dataset元素）
            .sppvt_control_output     % timeseries对象
                .Data(end)            % 最终输出值
```

**重要发现**：
- `Values`字段是一个**结构体**，包含12个字段对应`DecisionSPPVTOutput`总线
- 每个字段都是独立的`timeseries`对象
- 不能使用`find()`、`get()`等Dataset方法访问`Values`内的字段
- 必须使用结构体字段访问：`output_element.Values.fieldname`

### 📋 完整的SPPVT输出提取函数

```matlab
function sppvt_results = extract_all_sppvt_outputs(sim_out)
% 提取所有SPPVT相关输出（已验证方法）

sppvt_results = struct();

% 定义所有SPPVT输出字段
sppvt_fields = {
    'sppvt_control_output',      % 主要控制输出
    'sppvt_velocity_output',     % 速度输出
    'sppvt_acceleration_output', % 加速度输出
    'sppvt_stage_output',        % 阶段输出
    'sppvt_status_output'        % 状态输出
};

try
    if isstruct(sim_out) && isfield(sim_out, 'yout') && ~isempty(sim_out.yout)
        output_data = sim_out.yout;

        if isa(output_data, 'Simulink.SimulationData.Dataset') && length(output_data) >= 1
            output_element = output_data{1};

            % 提取所有SPPVT字段
            for i = 1:length(sppvt_fields)
                field_name = sppvt_fields{i};

                if isstruct(output_element.Values) && isfield(output_element.Values, field_name)
                    ts_obj = output_element.Values.(field_name);

                    if isa(ts_obj, 'timeseries') && ~isempty(ts_obj.Data)
                        sppvt_results.(field_name) = double(ts_obj.Data(end));
                    else
                        sppvt_results.(field_name) = NaN;
                    end
                else
                    sppvt_results.(field_name) = NaN;
                end
            end
        end
    end

catch ME
    fprintf('错误：SPPVT输出提取失败: %s\n', ME.message);
    for i = 1:length(sppvt_fields)
        sppvt_results.(sppvt_fields{i}) = NaN;
    end
end

end
```

### 🧪 验证和测试代码

```matlab
function verify_sppvt_output_extraction()
% 验证SPPVT输出提取方法的正确性

fprintf('=== SPPVT输出提取方法验证 ===\n');

model_name = 'ACC_Decision_SPPVT_Integrated';

% 准备测试环境
load('DecisionSPPVTBusDefinitions.mat');
if ~bdIsLoaded(model_name)
    load_system(model_name);
end

% 创建测试输入（确保SPPVT模块激活）
test_input = create_test_input_data(50, 1.5, 1, true);  % I0指令，有控制误差

% 配置并运行仿真
set_param(model_name, 'StopTime', '0.05');
assignin('base', 'test_input', test_input);
set_param(model_name, 'ExternalInput', 'test_input');

fprintf('运行仿真...\n');
sim_out = sim(model_name);

% 使用已验证的方法提取SPPVT输出
sppvt_value = get_sppvt_output_confirmed_method(sim_out);
all_sppvt_outputs = extract_all_sppvt_outputs(sim_out);

% 显示结果
fprintf('\n提取结果:\n');
fprintf('主要SPPVT控制输出: %.6f\n', sppvt_value);
fprintf('\n所有SPPVT输出:\n');
field_names = fieldnames(all_sppvt_outputs);
for i = 1:length(field_names)
    field = field_names{i};
    value = all_sppvt_outputs.(field);
    if isnan(value)
        fprintf('  %s: 未获取到\n', field);
    else
        fprintf('  %s: %.6f\n', field, value);
    end
end

% 验证数据结构
fprintf('\n数据结构验证:\n');
if isstruct(sim_out) && isfield(sim_out, 'yout')
    fprintf('✅ sim_out.yout 存在\n');
    if isa(sim_out.yout, 'Simulink.SimulationData.Dataset')
        fprintf('✅ yout 是 Dataset 类型\n');
        if length(sim_out.yout) >= 1
            fprintf('✅ Dataset 包含至少1个元素\n');
            element = sim_out.yout{1};
            if isstruct(element.Values)
                fprintf('✅ Values 是结构体\n');
                sppvt_fields = {'sppvt_control_output', 'sppvt_velocity_output', 'sppvt_acceleration_output'};
                for i = 1:length(sppvt_fields)
                    if isfield(element.Values, sppvt_fields{i})
                        fprintf('✅ 找到字段: %s\n', sppvt_fields{i});
                    else
                        fprintf('❌ 缺少字段: %s\n', sppvt_fields{i});
                    end
                end
            else
                fprintf('❌ Values 不是结构体\n');
            end
        else
            fprintf('❌ Dataset 为空\n');
        end
    else
        fprintf('❌ yout 不是 Dataset 类型\n');
    end
else
    fprintf('❌ sim_out.yout 不存在\n');
end

fprintf('\n=== 验证完成 ===\n');

end
```

### 📝 调试过程记录

本方法是通过以下调试步骤确认的：

1. **问题发现**：所有传统方法（`find()`、`get()`、数组索引）都无法获取SPPVT输出
2. **数据结构分析**：创建诊断脚本分析`sim_out`的完整结构
3. **关键突破**：发现`Values`是结构体而非Dataset元素
4. **方法验证**：通过多次测试确认`sim_out.yout{1}.Values.fieldname.Data(end)`的正确性
5. **最终确认**：用户确认此方法可以获取实际的SPPVT输出值

### ⚠️ 使用注意事项

1. **总线定义**：确保在仿真前加载`DecisionSPPVTBusDefinitions.mat`
2. **模型配置**：确保输出保存配置正确：`SaveOutput='on'`, `SaveFormat='Dataset'`
3. **错误处理**：始终使用try-catch块处理可能的数据结构变化
4. **数据类型**：使用`double()`确保数值类型正确
5. **索引安全**：使用`Data(end)`获取最终时刻的值


---

## 总线数据处理

### 总线数据结构验证

```matlab
function verify_bus_structure()
% 验证总线定义是否正确加载

fprintf('=== 总线结构验证 ===\n');

% 检查输入总线
if exist('DecisionSPPVTInput', 'var')
    input_elements = DecisionSPPVTInput.Elements;
    fprintf('✅ DecisionSPPVTInput总线已加载 (%d个字段):\n', length(input_elements));
    for i = 1:length(input_elements)
        fprintf('  %d. %s (%s)\n', i, input_elements(i).Name, input_elements(i).DataType);
    end
else
    fprintf('❌ DecisionSPPVTInput总线未加载\n');
end

% 检查输出总线
if exist('DecisionSPPVTOutput', 'var')
    output_elements = DecisionSPPVTOutput.Elements;
    fprintf('✅ DecisionSPPVTOutput总线已加载 (%d个字段):\n', length(output_elements));
    for i = 1:length(output_elements)
        fprintf('  %d. %s (%s)\n', i, output_elements(i).Name, output_elements(i).DataType);
    end
else
    fprintf('❌ DecisionSPPVTOutput总线未加载\n');
end

end
```

### 总线数据类型转换

```matlab
function converted_data = convert_bus_data_types(raw_data)
% 确保总线数据类型正确

converted_data = raw_data;

% 确保整型字段为int32
int32_fields = {'command_type', 'control_mode_flag', 'current_state', 'current_decision', 'debug_message'};
for i = 1:length(int32_fields)
    field = int32_fields{i};
    if isfield(converted_data, field) && isa(converted_data.(field), 'timeseries')
        converted_data.(field).Data = int32(converted_data.(field).Data);
    end
end

% 确保逻辑字段为logical
logical_fields = {'command_active', 'manual_throttle_active', 'control_enabled', 'torque_arbitration_active'};
for i = 1:length(logical_fields)
    field = logical_fields{i};
    if isfield(converted_data, field) && isa(converted_data.(field), 'timeseries')
        converted_data.(field).Data = logical(converted_data.(field).Data);
    end
end

% 确保双精度字段为double
double_fields = {'ego_speed_kmh', 'ego_speed_ms', 'control_error', 'V_target_kmh', 'V_min_kmh', 'G2_s', 'timestamp', ...
                 'updated_V_target_kmh', 'updated_G2_s', 'sppvt_control_output', 'sppvt_velocity_output', ...
                 'sppvt_acceleration_output', 'sppvt_stage_output', 'sppvt_status_output'};
for i = 1:length(double_fields)
    field = double_fields{i};
    if isfield(converted_data, field) && isa(converted_data.(field), 'timeseries')
        converted_data.(field).Data = double(converted_data.(field).Data);
    end
end

end
```

---

## 故障排除指南

### 常见问题诊断

#### 1. 总线定义问题
```matlab
% 症状：模型加载失败，提示无法找到DecisionSPPVTInput
% 解决方案：
load('DecisionSPPVTBusDefinitions.mat');
% 确认：
whos DecisionSPPVTInput DecisionSPPVTOutput
```

#### 2. 输出数据为空
```matlab
% 症状：sim_result只有SimulationMetadata字段
% 诊断步骤：
function diagnose_empty_output(model_name)
    % 1. 检查输出端口
    outports = find_system(model_name, 'SearchDepth', 1, 'BlockType', 'Outport');
    fprintf('输出端口数量: %d\n', length(outports));

    % 2. 检查保存配置
    save_output = get_param(model_name, 'SaveOutput');
    save_format = get_param(model_name, 'SaveFormat');
    fprintf('SaveOutput: %s, SaveFormat: %s\n', save_output, save_format);

    % 3. 检查信号连接
    % （需要进一步实现）
end
```

#### 3. SPPVT输出为0
```matlab
% 症状：SPPVT适配器有输出，但获取到的值为0
% 可能原因：
% 1. 控制使能状态未激活
% 2. 总线字段映射错误
% 3. 数据类型转换问题

% 调试方法：
% 在decision_function.m中添加调试输出
fprintf('SPPVT Debug: Control=%d, Error=%.3f, Output=%.3f\n', ...
        control_enabled, control_error, sppvt_control_output);
```

### 性能优化建议

1. **使用固定步长仿真**：确保时间同步
2. **最小化仿真时间**：使用0.05-0.15秒的短时间窗口
3. **预分配数据结构**：避免动态内存分配
4. **批量处理**：对多个测试用例使用相同的模型实例

---

---

## 结论

经过详细测试验证，已确认正确的SPPVT输出获取方法：`sim_out.yout{1}.Values.sppvt_control_output.Data(end)`。

### 关键发现
- `Values`字段是结构体，包含12个timeseries对象对应`DecisionSPPVTOutput`总线
- 必须使用结构体字段访问，不能使用Dataset的`find()`或`get()`方法
- 通过5步调试过程确认此方法的正确性

### 使用要点
1. 加载总线定义：`load('DecisionSPPVTBusDefinitions.mat')`
2. 配置输出保存：`SaveOutput='on'`, `SaveFormat='Dataset'`
3. 使用正确的数据访问路径：`sim_out.yout{1}.Values.fieldname.Data(end)`
4. 添加错误处理确保程序稳定性