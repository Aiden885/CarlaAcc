%% 诊断 new_adapter_states 维度问题

clear all;
clc;

% 加载总线定义
create_decision_sppvt_bus();

% 验证总线定义
fprintf('=== 总线定义检查 ===\n');
elements = DecisionSPPVTOutputExtended.Elements;
for i = 1:length(elements)
    if strcmp(elements(i).Name, 'new_adapter_states')
        fprintf('总线定义: new_adapter_states 维度 = %s\n\n', mat2str(elements(i).Dimensions));
    end
end

% 加载并配置模型
model_name = 'ACC_Decision_SPPVT_Integrated';
load_system(model_name);

% 检查输出端口配置
outport = [model_name '/integrated_output'];
fprintf('=== 输出端口配置 ===\n');
fprintf('BusObject: %s\n', get_param(outport, 'BusObject'));
fprintf('BusOutputAsStruct: %s\n', get_param(outport, 'BusOutputAsStruct'));
fprintf('\n');

% 准备简单的测试输入
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

% 配置并运行
set_param(model_name, 'StopTime', '0.05');
set_param(model_name, 'FixedStep', '0.05');

fprintf('=== 运行仿真 ===\n');
simOut = sim(model_name);

% 检查输出
output_data = simOut.yout;
raw_data = output_data{1}.Values.new_adapter_states.Data;

fprintf('\n=== 关键诊断 ===\n');
fprintf('原始 Data 完整维度: %s\n', mat2str(size(raw_data)));
fprintf('原始 Data 内容:\n');
disp(raw_data);

% 提取最后一帧
last_frame = output_data{1}.Values.new_adapter_states.Data(end, :, :);
fprintf('\nData(end,:,:) 维度: %s\n', mat2str(size(last_frame)));

% Squeeze
squeezed = squeeze(last_frame);
fprintf('squeeze(Data(end,:,:)) 维度: %s\n', mat2str(size(squeezed)));
fprintf('squeeze 后的长度: %d\n', length(squeezed));

fprintf('\n=== 结论 ===\n');
if length(squeezed) == 3
    fprintf('✅ 成功！new_adapter_states 有 3 个元素\n');
else
    fprintf('❌ 问题：new_adapter_states 只有 %d 个元素\n', length(squeezed));
    fprintf('   原始维度 %s 表明第3维只有 %d 个元素\n', mat2str(size(raw_data)), size(raw_data, 3));
end

fprintf('\n诊断完成！\n');