%% 测试 Simulink 模型输出端口
% 专门检查 new_adapter_states 的维度问题

clear all;
clc;

%% 1. 加载总线定义
fprintf('=== 步骤1: 加载总线定义 ===\n');
create_decision_sppvt_bus();

%% 2. 加载模型
fprintf('\n=== 步骤2: 加载模型 ===\n');
model_name = 'ACC_Decision_SPPVT_Integrated';
load_system(model_name);

%% 3. 准备测试输入数据
fprintf('\n=== 步骤3: 准备测试输入数据 ===\n');

% 创建测试输入
ego_speed_kmh = 50.0;
ego_speed_ms = ego_speed_kmh / 3.6;
command_type = int32(0);  % NONE
command_active = false;
manual_throttle_active = false;
control_error = -1.0;
control_mode_flag = int32(1);  % distance mode
V_target_kmh = 50.0;
V_min_kmh = 30.0;
G2_s = 2.0;
timestamp = 0.0;
current_state = int32(2);  % S2
has_history = false;
last_active_decision = int32(8);  % R8
external_stage_offset = 0.0;
external_stage_manager_states = [1.0; 0.0; 0.0];
external_adapter_states = [0.0; 0.0; 0.0];

% 创建 timeseries
simulink_input = struct();
simulink_input.ego_speed_kmh = timeseries(ego_speed_kmh, 0);
simulink_input.ego_speed_ms = timeseries(ego_speed_ms, 0);
simulink_input.command_type = timeseries(command_type, 0);
simulink_input.command_active = timeseries(logical(command_active), 0);
simulink_input.manual_throttle_active = timeseries(logical(manual_throttle_active), 0);
simulink_input.control_error = timeseries(control_error, 0);
simulink_input.control_mode_flag = timeseries(control_mode_flag, 0);
simulink_input.V_target_kmh = timeseries(V_target_kmh, 0);
simulink_input.V_min_kmh = timeseries(V_min_kmh, 0);
simulink_input.G2_s = timeseries(G2_s, 0);
simulink_input.timestamp = timeseries(timestamp, 0);
simulink_input.current_state = timeseries(current_state, 0);
simulink_input.has_history = timeseries(logical(has_history), 0);
simulink_input.last_active_decision = timeseries(last_active_decision, 0);
simulink_input.external_stage_offset = timeseries(external_stage_offset, 0);
simulink_input.external_stage_manager_states = timeseries(external_stage_manager_states, 0);
simulink_input.external_adapter_states = timeseries(external_adapter_states, 0);

% 设置时间单位
field_names = fieldnames(simulink_input);
for i = 1:length(field_names)
    simulink_input.(field_names{i}).TimeInfo.Units = 'seconds';
end

% 赋值给模型期望的变量名
input_data = simulink_input;

fprintf('测试输入数据已准备\n');

%% 4. 配置模型
fprintf('\n=== 步骤4: 配置模型 ===\n');
set_param(model_name, 'StopTime', '0.05');
set_param(model_name, 'FixedStep', '0.05');

%% 5. 运行仿真
fprintf('\n=== 步骤5: 运行仿真 ===\n');
simOut = sim(model_name);

%% 6. 检查输出
fprintf('\n=== 步骤6: 检查输出数据 ===\n');

output_data = simOut.yout;

% 检查输出结构
fprintf('输出数据类型: %s\n', class(output_data));
fprintf('输出数据长度: %d\n', length(output_data));

% 提取所有字段
fprintf('\n所有输出字段:\n');
fields = output_data{1}.Values;
field_names = fieldnames(fields);
for i = 1:length(field_names)
    fprintf('  字段 %d: %s\n', i, field_names{i});
end

%% 7. 专门检查 new_adapter_states
fprintf('\n=== 步骤7: 检查 new_adapter_states ===\n');

% 提取原始数据
raw_data = output_data{1}.Values.new_adapter_states.Data;
fprintf('原始 Data 维度: %s\n', mat2str(size(raw_data)));
fprintf('原始 Data 内容:\n');
disp(raw_data);

% 提取最后一个时间点的数据
last_data = output_data{1}.Values.new_adapter_states.Data(end, :, :);
fprintf('\nData(end,:,:) 维度: %s\n', mat2str(size(last_data)));
fprintf('Data(end,:,:) 内容:\n');
disp(last_data);

% Squeeze 后的数据
squeezed_data = squeeze(output_data{1}.Values.new_adapter_states.Data(end, :, :));
fprintf('\nsqueeze(Data(end,:,:)) 维度: %s\n', mat2str(size(squeezed_data)));
fprintf('squeeze(Data(end,:,:)) 内容:\n');
disp(squeezed_data);
fprintf('squeeze 后的长度: %d\n', length(squeezed_data));

% 逐个元素提取
fprintf('\n逐个元素提取:\n');
for i = 1:length(squeezed_data)
    fprintf('  元素 %d: %.6f\n', i, squeezed_data(i));
end

%% 8. 检查总线定义
fprintf('\n=== 步骤8: 检查总线对象定义 ===\n');
fprintf('\n查找 new_adapter_states 元素定义:\n');
elements = DecisionSPPVTOutputExtended.Elements;
for i = 1:length(elements)
    if strcmp(elements(i).Name, 'new_adapter_states')
        fprintf('  名称: %s\n', elements(i).Name);
        fprintf('  维度: %s\n', mat2str(elements(i).Dimensions));
        fprintf('  类型: %s\n', elements(i).DataType);
    end
end

%% 9. 总结
fprintf('\n=== 诊断总结 ===\n');
if length(squeezed_data) == 3
    fprintf('✅ new_adapter_states 有正确的3个元素\n');
    fprintf('   值: [%.6f, %.6f, %.6f]\n', squeezed_data(1), squeezed_data(2), squeezed_data(3));
else
    fprintf('❌ 问题: new_adapter_states 只有 %d 个元素（应该是3个）\n', length(squeezed_data));
    fprintf('   这说明 Simulink 输出端口配置有问题\n');
end

fprintf('\n测试完成！\n');
