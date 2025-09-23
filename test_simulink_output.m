function test_simulink_output()
    %% 测试新版本Simulink.SimulationOutput格式

    fprintf('🧪 测试Simulink.SimulationOutput格式解析...\n');

    model_name = 'ACC_Decision_SPPVT_Integrated';

    try
        % 加载模型
        load_system(model_name);
        create_decision_sppvt_bus();

        % 创建简单输入
        time_points = [0, 0.05];
        input_data = struct();
        input_data.ego_speed_kmh = timeseries([50.0, 50.0], time_points, 'Name', 'ego_speed_kmh');
        input_data.ego_speed_ms = timeseries([13.89, 13.89], time_points, 'Name', 'ego_speed_ms');
        input_data.command_type = timeseries(int32([1, 1]), time_points, 'Name', 'command_type');
        input_data.command_active = timeseries([true, true], time_points, 'Name', 'command_active');
        input_data.manual_throttle_active = timeseries([false, false], time_points, 'Name', 'manual_throttle_active');
        input_data.control_error = timeseries([1.5, -0.8], time_points, 'Name', 'control_error');
        input_data.control_mode_flag = timeseries(int32([1, 1]), time_points, 'Name', 'control_mode_flag');
        input_data.V_target_kmh = timeseries([50.0, 50.0], time_points, 'Name', 'V_target_kmh');
        input_data.V_min_kmh = timeseries([30.0, 30.0], time_points, 'Name', 'V_min_kmh');
        input_data.G2_s = timeseries([2.0, 2.0], time_points, 'Name', 'G2_s');
        input_data.timestamp = timeseries(time_points, time_points, 'Name', 'timestamp');

        % 设置时间单位
        field_names = fieldnames(input_data);
        for i = 1:length(field_names)
            input_data.(field_names{i}).TimeInfo.Units = 'seconds';
        end

        % 配置仿真
        set_param(model_name, 'StopTime', '0.05');
        set_param(model_name, 'FixedStep', '0.05');
        set_param(model_name, 'SaveOutput', 'on');
        set_param(model_name, 'OutputSaveName', 'yout');
        set_param(model_name, 'SaveFormat', 'Dataset');
        set_param(model_name, 'LoadExternalInput', 'on');
        set_param(model_name, 'ExternalInput', 'input_data');

        assignin('base', 'input_data', input_data);

        fprintf('📊 运行仿真...\n');
        sim_out = sim(model_name);

        fprintf('🔍 解析Simulink.SimulationOutput格式...\n');
        fprintf('sim_out类型: %s\n', class(sim_out));

        % 检查所有可用属性
        if isa(sim_out, 'Simulink.SimulationOutput')
            props = properties(sim_out);
            fprintf('可用属性: %s\n', strjoin(props, ', '));

            % 尝试获取yout
            if isprop(sim_out, 'yout')
                yout_data = sim_out.yout;
                fprintf('✅ 找到yout属性！\n');
                fprintf('yout类型: %s\n', class(yout_data));

                if isa(yout_data, 'Simulink.SimulationData.Dataset')
                    fprintf('📊 Dataset解析:\n');
                    fprintf('元素数量: %d\n', yout_data.numElements);

                    % 提取实际数据
                    for i = 1:yout_data.numElements
                        element = yout_data{i};
                        if isa(element, 'Simulink.SimulationData.Signal')
                            signal_name = element.Name;
                            signal_data = element.Values.Data;

                            fprintf('信号%d: %s\n', i, signal_name);
                            fprintf('  数据维度: %s\n', mat2str(size(signal_data)));
                            if ~isempty(signal_data)
                                fprintf('  最终值: %.6f\n', signal_data(end));

                                % 根据位置解析关键信号
                                if i == 1
                                    control_enabled = logical(signal_data(end));
                                    fprintf('  -> control_enabled: %d\n', control_enabled);
                                elseif i == 2
                                    current_state = double(signal_data(end));
                                    fprintf('  -> current_state: %d\n', current_state);
                                elseif i == 3
                                    current_decision = double(signal_data(end));
                                    fprintf('  -> current_decision: %d\n', current_decision);
                                elseif i == 8
                                    sppvt_output = double(signal_data(end));
                                    fprintf('  -> sppvt_control_output: %.6f\n', sppvt_output);
                                elseif i == 7
                                    debug_code = double(signal_data(end));
                                    fprintf('  -> debug_message: %d\n', debug_code);
                                end
                            end
                        end
                    end

                    % 验证结果
                    fprintf('\n🎯 验证结果:\n');
                    if exist('control_enabled', 'var') && exist('current_state', 'var') && exist('current_decision', 'var')
                        fprintf('✅ 成功解析输出!\n');
                        fprintf('控制使能: %d\n', control_enabled);
                        fprintf('当前状态: S%d\n', current_state);
                        fprintf('当前决策: R%d\n', current_decision);
                        if exist('sppvt_output', 'var')
                            fprintf('SPPVT输出: %.6f\n', sppvt_output);
                        end
                        if exist('debug_code', 'var')
                            fprintf('调试码: %d\n', debug_code);
                        end
                    else
                        fprintf('❌ 解析失败\n');
                    end
                else
                    fprintf('⚠️ yout不是Dataset格式: %s\n', class(yout_data));
                end
            else
                fprintf('❌ 没有找到yout属性\n');
            end

            % 检查其他可能的输出属性
            for i = 1:length(props)
                prop_name = props{i};
                if ~strcmp(prop_name, 'yout')
                    try
                        prop_value = sim_out.(prop_name);
                        fprintf('%s: %s\n', prop_name, class(prop_value));
                    catch
                        fprintf('%s: (无法访问)\n', prop_name);
                    end
                end
            end
        end

    catch ME
        fprintf('❌ 测试失败: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('错误位置: %s:%d\n', ME.stack(1).file, ME.stack(1).line);
        end
    end
end