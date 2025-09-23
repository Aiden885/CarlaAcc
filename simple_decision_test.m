function simple_decision_test()
%% 简化的决策持续性测试

fprintf('🧪 测试决策持续性逻辑...\n');

model_name = 'ACC_Decision_SPPVT_Integrated';

try
    load_system(model_name);
    create_decision_sppvt_bus();

    % 简单测试: S2→S0(I0激活)→保持(无指令)
    time_points = [0, 0.05, 0.10];

    input_data = struct();
    input_data.ego_speed_kmh = timeseries([50.0, 50.0, 50.0], time_points, 'Name', 'ego_speed_kmh');
    input_data.ego_speed_ms = timeseries([13.89, 13.89, 13.89], time_points, 'Name', 'ego_speed_ms');
    input_data.manual_throttle_active = timeseries([false, false, false], time_points, 'Name', 'manual_throttle_active');
    input_data.control_error = timeseries([1.5, 1.5, 1.5], time_points, 'Name', 'control_error');
    input_data.control_mode_flag = timeseries(int32([1, 1, 1]), time_points, 'Name', 'control_mode_flag');
    input_data.V_target_kmh = timeseries([50.0, 50.0, 50.0], time_points, 'Name', 'V_target_kmh');
    input_data.V_min_kmh = timeseries([30.0, 30.0, 30.0], time_points, 'Name', 'V_min_kmh');
    input_data.G2_s = timeseries([2.0, 2.0, 2.0], time_points, 'Name', 'G2_s');
    input_data.timestamp = timeseries(time_points, time_points, 'Name', 'timestamp');

    % 关键: 第一步激活I0，第二步保持但不激活
    input_data.command_type = timeseries(int32([1, 1, 1]), time_points, 'Name', 'command_type');
    input_data.command_active = timeseries([true, false, false], time_points, 'Name', 'command_active');

    % 设置时间单位
    field_names = fieldnames(input_data);
    for i = 1:length(field_names)
        input_data.(field_names{i}).TimeInfo.Units = 'seconds';
    end

    % 配置仿真
    set_param(model_name, 'StopTime', '0.10');
    set_param(model_name, 'FixedStep', '0.05');
    set_param(model_name, 'SaveOutput', 'on');
    set_param(model_name, 'OutputSaveName', 'yout');
    set_param(model_name, 'SaveFormat', 'Dataset');
    set_param(model_name, 'LoadExternalInput', 'on');
    set_param(model_name, 'ExternalInput', 'input_data');

    assignin('base', 'input_data', input_data);

    fprintf('🚀 运行简化测试...\n');
    fprintf('   时间0.00: I0 active=true (期望: S2→S0, R1或R5)\n');
    fprintf('   时间0.05: I0 active=false (期望: S0保持, 决策保持R1或R5)\n');
    fprintf('   时间0.10: I0 active=false (期望: S0保持, 决策保持R1或R5)\n');

    sim_out = sim(model_name);

    % 解析结果
    if isa(sim_out, 'Simulink.SimulationOutput') && isprop(sim_out, 'yout')
        output_data = sim_out.yout;
        if isa(output_data, 'Simulink.SimulationData.Dataset') && output_data.numElements >= 1
            element = output_data{1};
            if isa(element, 'Simulink.SimulationData.Signal') && isstruct(element.Values)
                values_struct = element.Values;

                state_data = double(values_struct.current_state.Data);
                decision_data = double(values_struct.current_decision.Data);
                control_data = logical(values_struct.control_enabled.Data);

                fprintf('\n📊 结果分析:\n');
                fprintf('时间     状态  决策  控制  解释\n');
                fprintf('--------------------------------\n');
                for i = 1:length(time_points)
                    fprintf('%.2f     S%d    R%d   %d     ', time_points(i), state_data(i), decision_data(i), control_data(i));
                    if i == 1
                        fprintf('初始激活\n');
                    else
                        fprintf('决策保持测试\n');
                    end
                end

                % 验证决策持续性
                if length(decision_data) >= 2
                    if decision_data(2) == decision_data(3) && decision_data(2) ~= 8
                        fprintf('\n✅ 决策持续性测试通过! 决策从R%d保持到R%d\n', decision_data(2), decision_data(3));
                    else
                        fprintf('\n❌ 决策持续性测试失败! 决策从R%d变为R%d\n', decision_data(2), decision_data(3));
                        fprintf('   问题可能在: 修复逻辑没有正确生效\n');
                    end
                end
            end
        end
    end

catch ME
    fprintf('❌ 测试失败: %s\n', ME.message);
end

fprintf('\n🏁 简化测试完成\n');
end