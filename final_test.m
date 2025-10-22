%% 最终完整测试 - 检查 new_adapter_states 维度问题

clear all;
clc;

fprintf('=== 步骤1: 加载总线定义 ===\n');
create_decision_sppvt_bus();

fprintf('\n=== 步骤2: 验证总线定义 ===\n');
elements = DecisionSPPVTOutputExtended.Elements;
for i = 1:length(elements)
    if strcmp(elements(i).Name, 'new_adapter_states')
        fprintf('总线定义: new_adapter_states 维度 = %s\n', mat2str(elements(i).Dimensions));
    end
end

fprintf('\n=== 步骤3: 加载模型 ===\n');
model_name = 'ACC_Decision_SPPVT_Integrated';
load_system(model_name);

fprintf('\n=== 步骤4: 验证 SPPVT_Adapter 配置 ===\n');
adapter_blk = [model_name '/SPPVT_Adapter'];
chartData = get_param(adapter_blk, 'Object');
outputs = find(chartData, '-isa', 'Stateflow.Data', 'Scope', 'Output');
for i = 1:length(outputs)
    if strcmp(outputs(i).Name, 'adapter_states_out')
        fprintf('adapter_states_out 维度: %s\n', outputs(i).Props.Array.Size);
    end
end

fprintf('\n=== 步骤5: 验证输出端口配置 ===\n');
outport = [model_name '/integrated_output'];
fprintf('BusObject: %s\n', get_param(outport, 'BusObject'));
fprintf('BusOutputAsStruct: %s\n', get_param(outport, 'BusOutputAsStruct'));

fprintf('\n=== 步骤6: 准备测试输入 ===\n');
ego_speed_kmh = 50.0;
ego_speed_ms = ego_speed_kmh / 3.6;
control_error = -1.0;

simulink_input = struct();
simulink_input.ego_speed_kmh = timeseries(ego_speed_kmh, 0);
simulink_input.ego_speed_ms = timeseries(ego_speed_ms, 0);
simulink_input.command_type = timeseries(int32(0), 0);
simulink_input.command_active = timeseries(false, 0);
simulink_input.manual_throttle_active = timeseries(false, 0);
simulink_input.control_error = timeseries(control_error, 0);
simulink_input.control_mode_flag = timeseries(int32(1), 0);
simulink_input.V_target_kmh = timeseries(50.0, 0);
simulink_input.V_min_kmh = timeseries(30.0, 0);
simulink_input.G2_s = timeseries(2.0, 0);
simulink_input.timestamp = timeseries(0.0, 0);
simulink_input.current_state = timeseries(int32(2), 0);
simulink_input.has_history = timeseries(false, 0);
simulink_input.last_active_decision = timeseries(int32(8), 0);
simulink_input.external_stage_offset = timeseries(0.0, 0);
simulink_input.external_stage_manager_states = timeseries([1.0; 0.0; 0.0], 0);
simulink_input.external_adapter_states = timeseries([0.0; 0.0; 0.0], 0);

input_data = simulink_input;

fprintf('\n=== 步骤7: 配置并运行仿真 ===\n');
set_param(model_name, 'StopTime', '0.05');
set_param(model_name, 'FixedStep', '0.05');

simOut = sim(model_name);

fprintf('\n=== 步骤8: 分析输出数据 ===\n');
output_data = simOut.yout;

% 检查 new_adapter_states
raw_data = output_data{1}.Values.new_adapter_states.Data;
fprintf('原始 Data 维度: %s\n', mat2str(size(raw_data)));

last_frame = output_data{1}.Values.new_adapter_states.Data(end, :, :);
squeezed = squeeze(last_frame);
fprintf('squeeze 后维度: %s\n', mat2str(size(squeezed)));
fprintf('squeeze 后长度: %d\n', length(squeezed));

% 关键：检查中间步骤的数据
fprintf('\n=== 步骤9: 检查中间数据流 ===\n');
fprintf('从 simOut 提取完整数据...\n');

% 检查 Bus 信号
try
    % 获取 new_adapter_states 的完整时间序列
    ts_data = output_data{1}.Values.new_adapter_states;
    fprintf('TimeSeries 数据类型: %s\n', class(ts_data.Data));
    fprintf('TimeSeries 数据维度: %s\n', mat2str(size(ts_data.Data)));

    % 显示所有时间点的数据
    fprintf('\n所有时间点的 new_adapter_states:\n');
    for t = 1:size(ts_data.Data, 1)
        frame_data = squeeze(ts_data.Data(t, :, :));
        fprintf('  时间点 %d: 维度=%s, 数据=', t, mat2str(size(frame_data)));
        disp(frame_data');
    end
catch e
    fprintf('错误: %s\n', e.message);
end

fprintf('\n=== 最终结论 ===\n');
if length(squeezed) == 3
    fprintf('✅ 成功！new_adapter_states 有 3 个元素\n');
    fprintf('   值: [%.6f, %.6f, %.6f]\n', squeezed(1), squeezed(2), squeezed(3));
else
    fprintf('❌ 问题：new_adapter_states 只有 %d 个元素\n', length(squeezed));
    fprintf('   原始维度 %s 的第3维只有 %d 个元素\n', mat2str(size(raw_data)), size(raw_data, 3));
    fprintf('\n可能的原因：\n');
    fprintf('   1. Simulink 信号记录时截断了数组\n');
    fprintf('   2. 总线对象在运行时未正确应用\n');
    fprintf('   3. 模型中有其他模块覆盖了维度\n');
end

fprintf('\n测试完成！\n');