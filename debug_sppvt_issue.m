function debug_sppvt_issue()
%DEBUG_SPPVT_ISSUE 诊断SPPVT输出为0的问题

fprintf('🔍 诊断SPPVT输出为0的问题...\n');

model_name = 'ACC_Decision_SPPVT_Integrated';

try
    % 加载模型和总线定义
    if ~evalin('base', 'exist(''DecisionSPPVTInput'', ''var'')')
        create_decision_sppvt_bus();
    end
    if ~bdIsLoaded(model_name)
        load_system(model_name);
    end

    % 创建简单的测试输入（只有2个时间点）
    fprintf('📋 创建简单测试输入...\n');
    time_points = [0, 0.05];

    simple_input = struct();
    simple_input.ego_speed_kmh = timeseries([50.0, 50.0], time_points, 'Name', 'ego_speed_kmh');
    simple_input.ego_speed_ms = timeseries([13.89, 13.89], time_points, 'Name', 'ego_speed_ms');
    simple_input.command_type = timeseries(int32([1, 1]), time_points, 'Name', 'command_type');
    simple_input.command_active = timeseries(logical([true, true]), time_points, 'Name', 'command_active');
    simple_input.manual_throttle_active = timeseries(logical([false, false]), time_points, 'Name', 'manual_throttle_active');
    simple_input.control_error = timeseries([1.5, 1.5], time_points, 'Name', 'control_error');
    simple_input.control_mode_flag = timeseries(int32([1, 1]), time_points, 'Name', 'control_mode_flag');
    simple_input.V_target_kmh = timeseries([50.0, 50.0], time_points, 'Name', 'V_target_kmh');
    simple_input.V_min_kmh = timeseries([30.0, 30.0], time_points, 'Name', 'V_min_kmh');
    simple_input.G2_s = timeseries([2.0, 2.0], time_points, 'Name', 'G2_s');
    simple_input.timestamp = timeseries([0.0, 0.05], time_points, 'Name', 'timestamp');

    % 设置时间单位
    field_names = fieldnames(simple_input);
    for i = 1:length(field_names)
        simple_input.(field_names{i}).TimeInfo.Units = 'seconds';
    end

    fprintf('✅ 简单输入创建完成，所有字段时间长度一致\n');

    % 配置仿真
    set_param(model_name, 'StopTime', '0.05');
    set_param(model_name, 'FixedStep', '0.05');
    set_param(model_name, 'LoadExternalInput', 'on');
    set_param(model_name, 'SaveOutput', 'on');
    set_param(model_name, 'OutputSaveName', 'yout');
    set_param(model_name, 'SaveFormat', 'Dataset');

    assignin('base', 'simple_input', simple_input);
    set_param(model_name, 'ExternalInput', 'simple_input');

    fprintf('▶️ 运行简单测试仿真...\n');
    sim_out = sim(model_name);

    % 使用正确的方法提取SPPVT输出
    fprintf('📊 提取SPPVT输出...\n');

    if isa(sim_out, 'Simulink.SimulationOutput') && isprop(sim_out, 'yout')
        output_data = sim_out.yout;

        if isa(output_data, 'Simulink.SimulationData.Dataset') && output_data.numElements >= 1
            element = output_data{1};

            if isa(element, 'Simulink.SimulationData.Signal') && isstruct(element.Values)
                values_struct = element.Values;

                if isfield(values_struct, 'sppvt_control_output')
                    sppvt_ts = values_struct.sppvt_control_output;

                    fprintf('🔍 SPPVT数据结构检查:\n');
                    fprintf('   字段类型: %s\n', class(sppvt_ts));

                    if isa(sppvt_ts, 'timeseries')
                        fprintf('   数据长度: %d\n', length(sppvt_ts.Data));
                        fprintf('   时间长度: %d\n', length(sppvt_ts.Time));
                        fprintf('   数据类型: %s\n', class(sppvt_ts.Data));
                        fprintf('   数据值: [');
                        for i = 1:length(sppvt_ts.Data)
                            fprintf('%.6f', sppvt_ts.Data(i));
                            if i < length(sppvt_ts.Data)
                                fprintf(', ');
                            end
                        end
                        fprintf(']\n');
                        fprintf('   最终值: %.6f\n', double(sppvt_ts.Data(end)));

                        % 检查其他相关字段
                        if isfield(values_struct, 'control_enabled')
                            control_ts = values_struct.control_enabled;
                            fprintf('   控制使能: %d (最终值)\n', logical(control_ts.Data(end)));
                        end
                        if isfield(values_struct, 'current_state')
                            state_ts = values_struct.current_state;
                            fprintf('   当前状态: S%d (最终值)\n', double(state_ts.Data(end)));
                        end
                        if isfield(values_struct, 'current_decision')
                            decision_ts = values_struct.current_decision;
                            fprintf('   当前决策: R%d (最终值)\n', double(decision_ts.Data(end)));
                        end

                        % 检查SPPVT输入参数
                        if isfield(values_struct, 'updated_V_target_kmh')
                            v_target_ts = values_struct.updated_V_target_kmh;
                            fprintf('   更新目标速度: %.1f (最终值)\n', double(v_target_ts.Data(end)));
                        end
                        if isfield(values_struct, 'updated_G2_s')
                            g2_ts = values_struct.updated_G2_s;
                            fprintf('   更新时距: %.1f (最终值)\n', double(g2_ts.Data(end)));
                        end

                    else
                        fprintf('   ❌ sppvt_control_output不是timeseries类型\n');
                    end
                else
                    fprintf('   ❌ 未找到sppvt_control_output字段\n');
                    fprintf('   可用字段: %s\n', strjoin(fieldnames(values_struct), ', '));
                end
            else
                fprintf('   ❌ element.Values不是struct类型\n');
            end
        else
            fprintf('   ❌ 输出不是Dataset格式或为空\n');
        end
    else
        fprintf('   ❌ sim_out格式异常\n');
    end

    % 测试不同控制误差
    fprintf('\n🔄 测试不同控制误差...\n');
    error_values = [3.0, 1.5, 0.0, -1.5, -3.0];

    for i = 1:length(error_values)
        fprintf('   误差值 %.1f: ', error_values(i));

        % 更新控制误差
        test_input = simple_input;
        test_input.control_error = timeseries([error_values(i), error_values(i)], time_points, 'Name', 'control_error');

        assignin('base', 'test_input', test_input);
        set_param(model_name, 'ExternalInput', 'test_input');

        sim_out = sim(model_name);

        % 提取SPPVT输出
        sppvt_value = 0;
        if isa(sim_out, 'Simulink.SimulationOutput') && isprop(sim_out, 'yout')
            output_data = sim_out.yout;
            if isa(output_data, 'Simulink.SimulationData.Dataset') && output_data.numElements >= 1
                element = output_data{1};
                if isa(element, 'Simulink.SimulationData.Signal') && isstruct(element.Values)
                    values_struct = element.Values;
                    if isfield(values_struct, 'sppvt_control_output')
                        sppvt_ts = values_struct.sppvt_control_output;
                        if isa(sppvt_ts, 'timeseries') && ~isempty(sppvt_ts.Data)
                            sppvt_value = double(sppvt_ts.Data(end));
                        end
                    end
                end
            end
        end

        fprintf('SPPVT输出 = %.6f\n', sppvt_value);
    end

catch ME
    fprintf('❌ 诊断失败: %s\n', ME.message);
    fprintf('错误详情: %s\n', getReport(ME, 'basic'));
end

fprintf('\n📋 诊断完成！\n');

end