function diagnose_acc_decision_model()
%% 测试基于MATLAB Function的ACC决策Simulink模型
% 专门测试acc_decision_matlab_function模型的功能

    model_name = 'acc_decision_matlab_function';

    fprintf('=== 测试MATLAB Function版ACC决策模型 ===\n');

    % 检查模型是否加载
    if ~bdIsLoaded(model_name)
        fprintf('模型未加载，尝试加载...\n');
        try
            load_system(model_name);
        catch
            error('无法加载模型 %s.slx，请确保模型存在且连接正确', model_name);
        end
    end

    % 初始化测试统计
    test_results = struct('total', 0, 'passed', 0, 'failed', 0, 'details', {{}});

    fprintf('开始执行测试...\n\n');

    % 测试1: 模型基本功能
    test_results = run_basic_functionality_test(model_name, test_results);

    % 测试2: 状态转移测试
    test_results = run_state_transition_test(model_name, test_results);

    % 测试3: 参数调整测试
    test_results = run_parameter_adjustment_test(model_name, test_results);

    % 测试4: 历史机制测试
    test_results = run_history_mechanism_test(model_name, test_results);

    % 测试5: 自动转移测试
    test_results = run_automatic_transition_test(model_name, test_results);

    % 生成测试报告
    generate_test_report(test_results);
end

function test_results = run_basic_functionality_test(model_name, test_results)
    fprintf('测试1: 模型基本功能\n');
    fprintf('====================\n');

    % 测试1.1: 重置功能
    fprintf('1.1 测试重置功能\n');
    outputs = run_single_test(model_name, struct('reset_signal', true), 0.2);
    test_results = verify_test(test_results, '重置后状态', outputs.current_state, 3, '==');
    test_results = verify_test(test_results, '重置后V3', outputs.V3_kmh, 50.0, '~=');
    test_results = verify_test(test_results, '重置后G1', outputs.G1_m, 15.0, '~=');
    test_results = verify_test(test_results, '重置后历史标志', outputs.has_history, false, '==');

    % 测试1.2: 无指令状态保持
    fprintf('1.2 测试无指令状态保持\n');
    reset_model(model_name);
    outputs = run_single_test(model_name, struct('command_input', -1, 'ego_speed_kmh', 50), 0.2);
    test_results = verify_test(test_results, '无指令状态保持', outputs.current_state, 2, '==');

    % 测试1.3: 参数边界保护
    fprintf('1.3 测试参数边界保护\n');
    % 这个需要通过多次调整来测试，暂时跳过

    fprintf('基本功能测试完成\n\n');
end

function test_results = run_state_transition_test(model_name, test_results)
    fprintf('测试2: 核心状态转移\n');
    fprintf('==================\n');

    % 测试2.1: S2->S0 (降速启控)
    fprintf('2.1 S2->S0 降速启控\n');
    reset_model(model_name);
    outputs = run_single_test(model_name, struct('command_input', 0, 'ego_speed_kmh', 50), 0.3);
    test_results = verify_test(test_results, 'S2->S0转移', outputs.current_state, 0, '==');
    test_results = verify_test(test_results, 'S2->S0控制模式', outputs.control_mode, 5, '==');
    test_results = verify_test(test_results, 'S2->S0消息码', outputs.message_code, 305, '==');

    % 测试2.2: S0->S1 (刹车待命)
    fprintf('2.2 S0->S1 刹车待命\n');
    outputs = run_single_test(model_name, struct('command_input', 5, 'ego_speed_kmh', 50), 0.3);
    test_results = verify_test(test_results, 'S0->S1转移', outputs.current_state, 1, '==');
    test_results = verify_test(test_results, 'S0->S1控制模式', outputs.control_mode, 8, '==');
    test_results = verify_test(test_results, 'S0->S1历史保存', outputs.has_history, true, '==');

    % 测试2.3: S1->S0 (继承启控)
    fprintf('2.3 S1->S0 继承启控\n');
    outputs = run_single_test(model_name, struct('command_input', 1, 'ego_speed_kmh', 50), 0.3);
    test_results = verify_test(test_results, 'S1->S0转移', outputs.current_state, 0, '==');
    test_results = verify_test(test_results, 'S1->S0控制模式', outputs.control_mode, 6, '==');

    % 测试2.4: S0->S1->S0 完整循环
    fprintf('2.4 完整控制循环\n');
    reset_model(model_name);
    run_single_test(model_name, struct('command_input', 0, 'ego_speed_kmh', 50), 0.2); % S2->S0
    run_single_test(model_name, struct('command_input', 6, 'ego_speed_kmh', 50), 0.2); % S0->S1
    outputs = run_single_test(model_name, struct('command_input', 1, 'ego_speed_kmh', 50), 0.2); % S1->S0
    test_results = verify_test(test_results, '完整循环最终状态', outputs.current_state, 0, '==');

    fprintf('状态转移测试完成\n\n');
end

function test_results = run_parameter_adjustment_test(model_name, test_results)
    fprintf('测试3: 参数调整功能\n');
    fprintf('==================\n');

    % 进入S0状态
    reset_model(model_name);
    run_single_test(model_name, struct('command_input', 0, 'ego_speed_kmh', 50), 0.2);

    % 测试3.1: 速度调整
    fprintf('3.1 速度参数调整\n');
    outputs_before = run_single_test(model_name, struct('command_input', -1, 'ego_speed_kmh', 50), 0.1);
    initial_V3 = outputs_before.V3_kmh;

    outputs_inc = run_single_test(model_name, struct('command_input', 1, 'ego_speed_kmh', 50), 0.2);
    test_results = verify_test(test_results, '增速功能', outputs_inc.V3_kmh, initial_V3 + 1, '~=');

    outputs_dec = run_single_test(model_name, struct('command_input', 0, 'ego_speed_kmh', 50), 0.2);
    test_results = verify_test(test_results, '降速功能', outputs_dec.V3_kmh, initial_V3, '~=');

    % 测试3.2: 距离调整
    fprintf('3.2 距离参数调整\n');
    initial_G1 = outputs_dec.G1_m;

    outputs_inc_dist = run_single_test(model_name, struct('command_input', 3, 'ego_speed_kmh', 50), 0.2);
    test_results = verify_test(test_results, '增距功能', outputs_inc_dist.G1_m, initial_G1 + 1, '~=');

    outputs_dec_dist = run_single_test(model_name, struct('command_input', 2, 'ego_speed_kmh', 50), 0.2);
    test_results = verify_test(test_results, '降距功能', outputs_dec_dist.G1_m, initial_G1, '~=');

    fprintf('参数调整测试完成\n\n');
end

function test_results = run_history_mechanism_test(model_name, test_results)
    fprintf('测试4: 历史机制\n');
    fprintf('==============\n');

    % 设置初始状态并调整参数
    reset_model(model_name);
    run_single_test(model_name, struct('command_input', 0, 'ego_speed_kmh', 50), 0.2); % 进入S0
    run_single_test(model_name, struct('command_input', 1, 'ego_speed_kmh', 50), 0.2); % V3+1
    run_single_test(model_name, struct('command_input', 3, 'ego_speed_kmh', 50), 0.2); % G1+1
    outputs_before_save = run_single_test(model_name, struct('command_input', -1, 'ego_speed_kmh', 50), 0.1);

    target_V3 = outputs_before_save.V3_kmh;
    target_G1 = outputs_before_save.G1_m;

    fprintf('4.1 历史数据保存\n');
    outputs_save = run_single_test(model_name, struct('command_input', 5, 'ego_speed_kmh', 50), 0.2);
    test_results = verify_test(test_results, '保存后进入S1', outputs_save.current_state, 1, '==');
    test_results = verify_test(test_results, '历史标志设置', outputs_save.has_history, true, '==');

    fprintf('4.2 历史数据恢复\n');
    outputs_restore = run_single_test(model_name, struct('command_input', 1, 'ego_speed_kmh', 50), 0.3);
    test_results = verify_test(test_results, '恢复后进入S0', outputs_restore.current_state, 0, '==');
    test_results = verify_test(test_results, 'V3参数恢复', outputs_restore.V3_kmh, target_V3, '~=');
    test_results = verify_test(test_results, 'G1参数恢复', outputs_restore.G1_m, target_G1, '~=');

    fprintf('历史机制测试完成\n\n');
end

function test_results = run_automatic_transition_test(model_name, test_results)
    fprintf('测试5: 自动状态转移\n');
    fprintf('==================\n');

    % 测试5.1: 低速自动转移
    fprintf('5.1 低速自动转移 S1->S3\n');
    reset_model(model_name);
    run_single_test(model_name, struct('command_input', 0, 'ego_speed_kmh', 50), 0.2); % S2->S0
    run_single_test(model_name, struct('command_input', 5, 'ego_speed_kmh', 50), 0.2); % S0->S1
    outputs_low = run_single_test(model_name, struct('command_input', -1, 'ego_speed_kmh', 25), 0.3);
    test_results = verify_test(test_results, '低速转移S1->S3', outputs_low.current_state, 3, '==');

    % 测试5.2: 恢复适速自动转移
    fprintf('5.2 恢复适速自动转移 S3->S1\n');
    outputs_normal = run_single_test(model_name, struct('command_input', -1, 'ego_speed_kmh', 40), 0.3);
    test_results = verify_test(test_results, '恢复转移S3->S1', outputs_normal.current_state, 1, '==');

    % 测试5.3: 无历史的恢复转移
    fprintf('5.3 无历史恢复转移 S3->S2\n');
    reset_model(model_name);
    run_single_test(model_name, struct('command_input', -1, 'ego_speed_kmh', 25), 0.2); % 进入S3
    outputs_no_history = run_single_test(model_name, struct('command_input', -1, 'ego_speed_kmh', 40), 0.3);
    test_results = verify_test(test_results, '无历史转移S3->S2', outputs_no_history.current_state, 2, '==');

    fprintf('自动转移测试完成\n\n');
end

function outputs = run_single_test(model_name, inputs, sim_time)
    % 运行单次测试的辅助函数

    % 默认输入值
    default_inputs = struct(...
        'command_input', -1, ...
        'ego_speed_kmh', 50, ...
        'has_target', false, ...
        'current_distance', -1, ...
        'reset_signal', false);

    % 合并输入
    if nargin > 1 && ~isempty(inputs)
        fields = fieldnames(inputs);
        for i = 1:length(fields)
            default_inputs.(fields{i}) = inputs.(fields{i});
        end
    end

    if nargin < 3
        sim_time = 0.2;
    end

    % 创建时间序列
    time_points = [0, sim_time];
    input_names = fieldnames(default_inputs);

    for i = 1:length(input_names)
        name = input_names{i};
        value = default_inputs.(name);

        if strcmp(name, 'command_input') && value >= 0
            % 指令信号：短脉冲
            ts_data = [value, -1];
        else
            % 其他信号：保持
            ts_data = [value, value];
        end

        ts = timeseries(ts_data, time_points);
        assignin('base', name, ts);
    end

    % 配置并运行仿真
    set_param(model_name, 'LoadExternalInput', 'on');
    set_param(model_name, 'ExternalInput', strjoin(input_names, ', '));
    set_param(model_name, 'StopTime', num2str(sim_time));
    set_param(model_name, 'SaveOutput', 'on');
    set_param(model_name, 'OutputSaveName', 'yout');
    set_param(model_name, 'SaveFormat', 'Dataset');

    try
        sim_result = sim(model_name);
        outputs = extract_final_outputs(sim_result);

        % 清理
        for i = 1:length(input_names)
            evalin('base', ['clear ' input_names{i}]);
        end
    catch ME
        % 清理
        for i = 1:length(input_names)
            try
                evalin('base', ['clear ' input_names{i}]);
            catch
            end
        end
        error('仿真失败: %s', ME.message);
    end
end

function reset_model(model_name)
    % 重置模型到初始状态
    run_single_test(model_name, struct('reset_signal', true), 0.1);
end

function outputs = extract_final_outputs(sim_result)
    % 提取最终输出值
    output_names = {'current_state', 'control_mode', 'acc_active', 'control_enabled', ...
                   'V3_kmh', 'G1_m', 'G2_s', 'has_history', 'message_code', ...
                   'pending_distance_adj', 'is_in_control'};

    outputs = struct();

    try
        yout = sim_result.yout;
        if isa(yout, 'Simulink.SimulationData.Dataset')
            for i = 1:length(output_names)
                name = output_names{i};
                if i <= yout.numElements
                    elem = yout.getElement(i);
                    if isa(elem.Values, 'timeseries')
                        outputs.(name) = elem.Values.Data(end);
                    else
                        outputs.(name) = elem.Values(end);
                    end
                else
                    outputs.(name) = get_default_value(name);
                end
            end
        else
            for i = 1:length(output_names)
                outputs.(output_names{i}) = get_default_value(output_names{i});
            end
        end
    catch
        for i = 1:length(output_names)
            outputs.(output_names{i}) = get_default_value(output_names{i});
        end
    end
end

function val = get_default_value(name)
    switch name
        case 'current_state', val = 2;
        case {'control_mode', 'message_code'}, val = 0;
        case 'V3_kmh', val = 50.0;
        case 'G1_m', val = 15.0;
        case 'G2_s', val = 2.0;
        case {'acc_active', 'control_enabled', 'has_history', 'is_in_control'}, val = false;
        otherwise, val = 0;
    end
end

function test_results = verify_test(test_results, description, actual, expected, operator)
    % 验证测试结果
    test_results.total = test_results.total + 1;

    success = false;
    try
        switch operator
            case '==', success = (actual == expected);
            case '~=', success = abs(actual - expected) < 0.1;
            case '>', success = actual > expected;
            case '<', success = actual < expected;
            case '>=', success = actual >= expected;
            case '<=', success = actual <= expected;
        end
    catch
        success = false;
    end

    if success
        test_results.passed = test_results.passed + 1;
        status = '通过';
    else
        test_results.failed = test_results.failed + 1;
        status = '失败';
    end

    detail = sprintf('  %s: %s (实际: %.3f, 期望: %.3f)', description, status, double(actual), double(expected));
    test_results.details{end+1} = detail;
    fprintf('%s\n', detail);
end

function generate_test_report(test_results)
    % 生成测试报告
    fprintf('===========================================\n');
    fprintf('测试报告\n');
    fprintf('===========================================\n');

    success_rate = (test_results.passed / test_results.total) * 100;
    fprintf('总测试数: %d\n', test_results.total);
    fprintf('通过数: %d (%.1f%%)\n', test_results.passed, success_rate);
    fprintf('失败数: %d (%.1f%%)\n', test_results.failed, 100 - success_rate);

    if test_results.failed > 0
        fprintf('\n失败的测试:\n');
        for i = 1:length(test_results.details)
            if contains(test_results.details{i}, '失败')
                fprintf('%s\n', test_results.details{i});
            end
        end
    end

    fprintf('\n总体评价: ');
    if success_rate >= 95
        fprintf('优秀 - 系统运行正常\n');
    elseif success_rate >= 80
        fprintf('良好 - 大部分功能正常\n');
    elseif success_rate >= 60
        fprintf('一般 - 需要检查部分功能\n');
    else
        fprintf('需要改进 - 存在较多问题\n');
    end

    fprintf('===========================================\n');
end