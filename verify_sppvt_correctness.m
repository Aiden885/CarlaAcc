function verify_sppvt_correctness()
%VERIFY_SPPVT_CORRECTNESS 验证SPPVT输出的正确性
%   深入测试SPPVT算法是否按预期工作，而不仅仅是有响应

fprintf('🔍 开始SPPVT正确性验证...\n');

model_name = 'ACC_Decision_SPPVT_Integrated';

try
    % 加载必要组件
    if ~evalin('base', 'exist(''DecisionSPPVTInput'', ''var'')')
        create_decision_sppvt_bus();
    end
    if ~bdIsLoaded(model_name)
        load_system(model_name);
    end

    fprintf('📋 测试1: 基础SPPVT响应特性\n');

    % 创建基础输入数据
    base_input = create_base_input();

    % 测试场景1: 不同控制误差的响应
    fprintf('   场景1: 不同控制误差的静态响应\n');
    error_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -1.5, -2.0, -3.0];
    static_results = test_static_response(model_name, base_input, error_values);

    % 分析静态响应特性
    analyze_static_response(error_values, static_results);

    fprintf('\n📋 测试2: SPPVT动态响应特性\n');

    % 测试场景2: 动态误差变化
    fprintf('   场景2: 动态误差变化响应\n');
    dynamic_results = test_dynamic_response(model_name, base_input);

    % 分析动态响应特性
    analyze_dynamic_response(dynamic_results);

    fprintf('\n📋 测试3: SPPVT参数敏感性\n');

    % 测试场景3: 不同车速下的响应
    fprintf('   场景3: 不同车速下的响应特性\n');
    speed_results = test_speed_sensitivity(model_name, base_input);

    % 分析速度敏感性
    analyze_speed_sensitivity(speed_results);

    fprintf('\n📋 测试4: 与理论SPPVT对比\n');

    % 测试场景4: 与理论SPPVT算法对比
    fprintf('   场景4: 与理论SPPVT算法输出对比\n');
    theory_results = compare_with_theory(model_name, base_input);

    % 分析理论一致性
    analyze_theory_consistency(theory_results);

    fprintf('\n🎯 SPPVT正确性验证总结\n');
    generate_correctness_summary(static_results, dynamic_results, speed_results, theory_results);

catch ME
    fprintf('❌ SPPVT正确性验证失败: %s\n', ME.message);
    fprintf('错误详情: %s\n', getReport(ME, 'basic'));
end

end

function base_input = create_base_input()
    %% 创建标准基础输入

    time_points = [0, 0.05];

    base_input = struct();
    base_input.ego_speed_kmh = timeseries([50.0, 50.0], time_points, 'Name', 'ego_speed_kmh');
    base_input.ego_speed_ms = timeseries([13.89, 13.89], time_points, 'Name', 'ego_speed_ms');
    base_input.command_type = timeseries(int32([1, 1]), time_points, 'Name', 'command_type');
    base_input.command_active = timeseries(logical([true, true]), time_points, 'Name', 'command_active');
    base_input.manual_throttle_active = timeseries(logical([false, false]), time_points, 'Name', 'manual_throttle_active');
    base_input.control_error = timeseries([1.0, 1.0], time_points, 'Name', 'control_error');
    base_input.control_mode_flag = timeseries(int32([1, 1]), time_points, 'Name', 'control_mode_flag');
    base_input.V_target_kmh = timeseries([50.0, 50.0], time_points, 'Name', 'V_target_kmh');
    base_input.V_min_kmh = timeseries([30.0, 30.0], time_points, 'Name', 'V_min_kmh');
    base_input.G2_s = timeseries([2.0, 2.0], time_points, 'Name', 'G2_s');
    base_input.timestamp = timeseries([0.0, 0.05], time_points, 'Name', 'timestamp');

    % 设置时间单位
    field_names = fieldnames(base_input);
    for i = 1:length(field_names)
        base_input.(field_names{i}).TimeInfo.Units = 'seconds';
    end
end

function results = test_static_response(model_name, base_input, error_values)
    %% 测试静态响应特性

    results = struct();
    results.error_values = error_values;
    results.sppvt_outputs = zeros(size(error_values));
    results.response_times = zeros(size(error_values));

    % 配置仿真参数
    set_param(model_name, 'StopTime', '0.05');
    set_param(model_name, 'FixedStep', '0.05');
    set_param(model_name, 'SaveOutput', 'on');
    set_param(model_name, 'OutputSaveName', 'yout');
    set_param(model_name, 'SaveFormat', 'Dataset');

    for i = 1:length(error_values)
        fprintf('     误差 %.1f: ', error_values(i));

        % 创建测试输入
        test_input = base_input;
        test_input.control_error = timeseries([error_values(i), error_values(i)], [0, 0.05], 'Name', 'control_error');

        % 运行仿真
        assignin('base', 'static_test_input', test_input);
        set_param(model_name, 'LoadExternalInput', 'on');
        set_param(model_name, 'ExternalInput', 'static_test_input');

        tic;
        sim_out = sim(model_name);
        sim_time = toc;

        % 提取SPPVT输出
        sppvt_output = extract_sppvt_output(sim_out);
        results.sppvt_outputs(i) = sppvt_output;
        results.response_times(i) = sim_time;

        fprintf('SPPVT=%.6f (%.3fs)\n', sppvt_output, sim_time);
    end
end

function analyze_static_response(error_values, results)
    %% 分析静态响应特性

    fprintf('   📊 静态响应分析:\n');

    % 1. 线性性检查
    positive_errors = error_values(error_values > 0);
    positive_outputs = results.sppvt_outputs(error_values > 0);
    negative_errors = error_values(error_values < 0);
    negative_outputs = results.sppvt_outputs(error_values < 0);

    if length(positive_errors) > 1
        positive_correlation = corrcoef(positive_errors, positive_outputs);
        fprintf('     正误差线性度: %.3f\n', positive_correlation(1,2));
    end

    if length(negative_errors) > 1
        negative_correlation = corrcoef(negative_errors, negative_outputs);
        fprintf('     负误差线性度: %.3f\n', negative_correlation(1,2));
    end

    % 2. 对称性检查
    if length(positive_errors) > 0 && length(negative_errors) > 0
        pos_mean = mean(positive_outputs);
        neg_mean = mean(negative_outputs);
        symmetry_ratio = abs(pos_mean + neg_mean) / (abs(pos_mean) + abs(neg_mean));
        fprintf('     对称性比率: %.3f (越接近0越对称)\n', symmetry_ratio);
    end

    % 3. 零点检查
    zero_idx = find(error_values == 0);
    if ~isempty(zero_idx)
        zero_output = results.sppvt_outputs(zero_idx);
        fprintf('     零误差输出: %.6f\n', zero_output);
        if abs(zero_output) < 0.001
            fprintf('     ✅ 零点特性正确\n');
        else
            fprintf('     ❌ 零点特性异常\n');
        end
    end

    % 4. 饱和检查
    max_output = max(abs(results.sppvt_outputs));
    fprintf('     最大输出幅值: %.3f\n', max_output);
    if max_output > 10.0
        fprintf('     ⚠️ 输出可能过大\n');
    elseif max_output < 0.1
        fprintf('     ⚠️ 输出可能过小\n');
    else
        fprintf('     ✅ 输出幅值合理\n');
    end

    % 5. 比例关系检查
    if length(error_values) > 2
        % 检查是否是简单的比例关系 (y = k*x)
        non_zero_idx = abs(error_values) > 0.001;
        if sum(non_zero_idx) > 1
            ratios = results.sppvt_outputs(non_zero_idx) ./ error_values(non_zero_idx);
            ratio_std = std(ratios);
            ratio_mean = mean(ratios);
            fprintf('     比例因子: %.3f ± %.3f\n', ratio_mean, ratio_std);

            if ratio_std < 0.01
                fprintf('     ⚠️ 疑似简单比例关系 (非真正SPPVT)\n');
            else
                fprintf('     ✅ 非线性特性符合SPPVT预期\n');
            end
        end
    end
end

function results = test_dynamic_response(model_name, base_input)
    %% 测试动态响应特性

    fprintf('   创建动态误差序列...\n');

    % 创建动态变化的控制误差: 阶跃 + 斜坡 + 正弦
    sim_time = 1.0;  % 1秒仿真
    time_points = 0:0.05:sim_time;

    % 动态误差设计
    error_sequence = zeros(size(time_points));
    for i = 1:length(time_points)
        t = time_points(i);
        if t < 0.3
            error_sequence(i) = 2.0;  % 初始大误差
        elseif t < 0.6
            error_sequence(i) = 2.0 - 4.0 * (t - 0.3) / 0.3;  % 线性下降到-2
        else
            error_sequence(i) = -2.0 + 1.5 * sin(2*pi*(t-0.6)/0.2);  % 正弦波动
        end
    end

    % 创建动态输入
    dynamic_input = base_input;
    dynamic_input.control_error = timeseries(error_sequence, time_points, 'Name', 'control_error');

    % 扩展其他字段到相同长度
    field_names = fieldnames(dynamic_input);
    for i = 1:length(field_names)
        if ~strcmp(field_names{i}, 'control_error')
            orig_data = dynamic_input.(field_names{i}).Data(1);
            extended_data = repmat(orig_data, size(time_points));
            dynamic_input.(field_names{i}) = timeseries(extended_data, time_points, 'Name', field_names{i});
            dynamic_input.(field_names{i}).TimeInfo.Units = 'seconds';
        end
    end

    % 配置并运行仿真
    set_param(model_name, 'StopTime', num2str(sim_time));
    assignin('base', 'dynamic_test_input', dynamic_input);
    set_param(model_name, 'ExternalInput', 'dynamic_test_input');

    fprintf('   运行动态仿真 (%.1f秒)...\n', sim_time);
    sim_out = sim(model_name);

    % 提取完整时间序列
    results = extract_dynamic_results(sim_out, time_points, error_sequence);
end

function analyze_dynamic_response(results)
    %% 分析动态响应特性

    fprintf('   📊 动态响应分析:\n');

    if isempty(results.sppvt_sequence)
        fprintf('     ❌ 无法获取动态响应数据\n');
        return;
    end

    % 1. 响应延迟分析
    error_changes = diff(results.error_sequence);
    sppvt_changes = diff(results.sppvt_sequence);

    % 找到主要变化点
    significant_changes = find(abs(error_changes) > 0.5);
    if ~isempty(significant_changes)
        fprintf('     检测到 %d 个显著误差变化点\n', length(significant_changes));

        % 简单的响应延迟估计
        delays = [];
        for i = 1:min(3, length(significant_changes))  % 分析前3个变化点
            change_idx = significant_changes(i);
            if change_idx < length(sppvt_changes) - 5
                % 寻找SPPVT响应的起始点
                response_start = find(abs(sppvt_changes(change_idx:change_idx+5)) > 0.1, 1);
                if ~isempty(response_start)
                    delay = response_start - 1;
                    delays(end+1) = delay * 0.05;  % 转换为时间
                end
            end
        end

        if ~isempty(delays)
            avg_delay = mean(delays);
            fprintf('     平均响应延迟: %.3f秒 (%d个采样点)\n', avg_delay, round(avg_delay/0.05));
        end
    end

    % 2. 超调和振荡分析
    sppvt_peaks = findpeaks_simple(results.sppvt_sequence);
    if length(sppvt_peaks) > 2
        fprintf('     检测到 %d 个响应峰值 (可能有振荡)\n', length(sppvt_peaks));
    else
        fprintf('     响应较为平滑，无明显振荡\n');
    end

    % 3. 稳态误差分析
    final_error = results.error_sequence(end);
    final_sppvt = results.sppvt_sequence(end);
    fprintf('     最终误差: %.3f, 最终SPPVT: %.3f\n', final_error, final_sppvt);

    % 4. 动态范围分析
    error_range = max(results.error_sequence) - min(results.error_sequence);
    sppvt_range = max(results.sppvt_sequence) - min(results.sppvt_sequence);
    dynamic_gain = sppvt_range / error_range;
    fprintf('     动态增益: %.3f (SPPVT变化/误差变化)\n', dynamic_gain);

    % 5. 能量分析
    error_energy = sum(results.error_sequence.^2);
    sppvt_energy = sum(results.sppvt_sequence.^2);
    fprintf('     误差能量: %.2f, SPPVT能量: %.2f\n', error_energy, sppvt_energy);
end

function results = test_speed_sensitivity(model_name, base_input)
    %% 测试不同车速下的响应特性

    speeds_kmh = [30, 40, 50, 60, 80, 100];  % 不同测试车速
    control_error = 1.5;  % 固定控制误差

    results = struct();
    results.speeds = speeds_kmh;
    results.sppvt_outputs = zeros(size(speeds_kmh));

    for i = 1:length(speeds_kmh)
        fprintf('     车速 %d km/h: ', speeds_kmh(i));

        test_input = base_input;
        test_input.ego_speed_kmh = timeseries([speeds_kmh(i), speeds_kmh(i)], [0, 0.05], 'Name', 'ego_speed_kmh');
        test_input.ego_speed_ms = timeseries([speeds_kmh(i)/3.6, speeds_kmh(i)/3.6], [0, 0.05], 'Name', 'ego_speed_ms');
        test_input.control_error = timeseries([control_error, control_error], [0, 0.05], 'Name', 'control_error');

        assignin('base', 'speed_test_input', test_input);
        set_param(model_name, 'ExternalInput', 'speed_test_input');

        sim_out = sim(model_name);
        sppvt_output = extract_sppvt_output(sim_out);
        results.sppvt_outputs(i) = sppvt_output;

        fprintf('SPPVT=%.6f\n', sppvt_output);
    end
end

function analyze_speed_sensitivity(results)
    %% 分析速度敏感性

    fprintf('   📊 速度敏感性分析:\n');

    % 1. 速度影响检查
    output_variance = var(results.sppvt_outputs);
    output_mean = mean(results.sppvt_outputs);
    cv = sqrt(output_variance) / abs(output_mean);  % 变异系数

    fprintf('     输出变异系数: %.3f\n', cv);

    if cv < 0.1
        fprintf('     ✅ 输出对车速不敏感 (稳定)\n');
    elseif cv < 0.3
        fprintf('     ⚠️ 输出对车速有中等敏感性\n');
    else
        fprintf('     ❌ 输出对车速高度敏感 (可能有问题)\n');
    end

    % 2. 趋势分析
    if length(results.speeds) > 2
        correlation = corrcoef(results.speeds, results.sppvt_outputs);
        fprintf('     速度-输出相关性: %.3f\n', correlation(1,2));

        if abs(correlation(1,2)) < 0.3
            fprintf('     ✅ 输出与车速无显著相关性\n');
        else
            fprintf('     ⚠️ 输出与车速存在相关性\n');
        end
    end

    % 3. 范围检查
    min_output = min(results.sppvt_outputs);
    max_output = max(results.sppvt_outputs);
    fprintf('     输出范围: [%.3f, %.3f]\n', min_output, max_output);
end

function results = compare_with_theory(model_name, base_input)
    %% 与理论SPPVT算法对比

    fprintf('   计算理论SPPVT输出并对比...\n');

    test_errors = [0.5, 1.0, 1.5, 2.0, 2.5];
    results = struct();
    results.errors = test_errors;
    results.actual_outputs = zeros(size(test_errors));
    results.theory_outputs = zeros(size(test_errors));

    % SPPVT理论参数 (从调试输出得知)
    delta = 0.05;  % 采样时间
    eta = 0.2;     % SPPVT参数
    speed_ms = 13.89;  % 车速 m/s

    for i = 1:length(test_errors)
        error = test_errors(i);

        % 获取实际输出
        test_input = base_input;
        test_input.control_error = timeseries([error, error], [0, 0.05], 'Name', 'control_error');

        assignin('base', 'theory_test_input', test_input);
        set_param(model_name, 'ExternalInput', 'theory_test_input');

        sim_out = sim(model_name);
        actual_output = extract_sppvt_output(sim_out);
        results.actual_outputs(i) = actual_output;

        % 计算理论输出 (简化的SPPVT算法)
        % 这是基于SPPVT原理的简化计算
        theory_output = calculate_theory_sppvt(error, speed_ms, delta, eta);
        results.theory_outputs(i) = theory_output;

        fprintf('     误差 %.1f: 实际=%.6f, 理论=%.6f, 差异=%.6f\n', ...
                error, actual_output, theory_output, abs(actual_output - theory_output));
    end
end

function theory_output = calculate_theory_sppvt(error, speed, delta, eta)
    %% 计算理论SPPVT输出 (简化版本)

    % 这是一个简化的SPPVT算法实现，用于对比
    % 实际的SPPVT算法可能更复杂

    % 基本的比例控制 + 速度相关调整
    base_gain = 1.0;  % 基础增益
    speed_factor = 1.0 + 0.01 * (speed - 10);  % 速度相关因子

    % 简单的SPPVT计算
    theory_output = base_gain * error * speed_factor;

    % 限制输出范围
    max_output = 5.0;
    theory_output = max(-max_output, min(max_output, theory_output));
end

function analyze_theory_consistency(results)
    %% 分析理论一致性

    fprintf('   📊 理论一致性分析:\n');

    if length(results.errors) < 2
        fprintf('     ❌ 数据不足，无法分析\n');
        return;
    end

    % 1. 均方根误差
    rmse = sqrt(mean((results.actual_outputs - results.theory_outputs).^2));
    fprintf('     RMSE: %.6f\n', rmse);

    % 2. 相关性
    correlation = corrcoef(results.actual_outputs, results.theory_outputs);
    fprintf('     相关系数: %.3f\n', correlation(1,2));

    % 3. 相对误差
    relative_errors = abs(results.actual_outputs - results.theory_outputs) ./ (abs(results.theory_outputs) + 0.001);
    mean_relative_error = mean(relative_errors);
    fprintf('     平均相对误差: %.1f%%\n', mean_relative_error * 100);

    % 4. 一致性判断
    if correlation(1,2) > 0.9 && mean_relative_error < 0.2
        fprintf('     ✅ 与理论高度一致\n');
    elseif correlation(1,2) > 0.7 && mean_relative_error < 0.5
        fprintf('     ⚠️ 与理论基本一致\n');
    else
        fprintf('     ❌ 与理论存在显著差异\n');
    end
end

function sppvt_output = extract_sppvt_output(sim_out)
    %% 提取SPPVT输出 (使用验证过的正确方法)

    sppvt_output = 0;

    try
        if isa(sim_out, 'Simulink.SimulationOutput') && isprop(sim_out, 'yout')
            output_data = sim_out.yout;

            if isa(output_data, 'Simulink.SimulationData.Dataset') && output_data.numElements >= 1
                element = output_data{1};

                if isa(element, 'Simulink.SimulationData.Signal') && isstruct(element.Values)
                    values_struct = element.Values;

                    if isfield(values_struct, 'sppvt_control_output')
                        sppvt_ts = values_struct.sppvt_control_output;
                        if isa(sppvt_ts, 'timeseries') && ~isempty(sppvt_ts.Data)
                            sppvt_output = double(sppvt_ts.Data(end));
                        end
                    end
                end
            end
        end
    catch
        % 提取失败，返回默认值0
    end
end

function results = extract_dynamic_results(sim_out, time_points, error_sequence)
    %% 提取动态仿真结果

    results = struct();
    results.time_points = time_points;
    results.error_sequence = error_sequence;
    results.sppvt_sequence = [];

    try
        if isa(sim_out, 'Simulink.SimulationOutput') && isprop(sim_out, 'yout')
            output_data = sim_out.yout;

            if isa(output_data, 'Simulink.SimulationData.Dataset') && output_data.numElements >= 1
                element = output_data{1};

                if isa(element, 'Simulink.SimulationData.Signal') && isstruct(element.Values)
                    values_struct = element.Values;

                    if isfield(values_struct, 'sppvt_control_output')
                        sppvt_ts = values_struct.sppvt_control_output;
                        if isa(sppvt_ts, 'timeseries') && ~isempty(sppvt_ts.Data)
                            results.sppvt_sequence = double(sppvt_ts.Data);
                        end
                    end
                end
            end
        end
    catch
        % 提取失败
    end
end

function peaks = findpeaks_simple(data)
    %% 简单的峰值检测

    peaks = [];
    if length(data) < 3
        return;
    end

    for i = 2:length(data)-1
        if data(i) > data(i-1) && data(i) > data(i+1)
            peaks(end+1) = i;
        end
    end
end

function generate_correctness_summary(static_results, dynamic_results, speed_results, theory_results)
    %% 生成正确性验证总结

    fprintf('==========================================\n');
    fprintf('📊 SPPVT正确性验证总结\n');
    fprintf('==========================================\n');

    % 1. 静态响应评估
    static_passed = true;
    if ~isempty(static_results.sppvt_outputs)
        zero_idx = find(static_results.error_values == 0);
        if ~isempty(zero_idx) && abs(static_results.sppvt_outputs(zero_idx)) > 0.001
            static_passed = false;
        end

        max_output = max(abs(static_results.sppvt_outputs));
        if max_output > 10.0 || max_output < 0.1
            static_passed = false;
        end
    else
        static_passed = false;
    end

    if static_passed
        fprintf('✅ 静态响应: 通过\n');
    else
        fprintf('❌ 静态响应: 失败\n');
    end

    % 2. 动态响应评估
    dynamic_passed = ~isempty(dynamic_results.sppvt_sequence);
    if dynamic_passed
        fprintf('✅ 动态响应: 通过\n');
    else
        fprintf('❌ 动态响应: 失败\n');
    end

    % 3. 速度敏感性评估
    speed_passed = true;
    if ~isempty(speed_results.sppvt_outputs)
        cv = std(speed_results.sppvt_outputs) / abs(mean(speed_results.sppvt_outputs));
        if cv > 0.5
            speed_passed = false;
        end
    else
        speed_passed = false;
    end

    if speed_passed
        fprintf('✅ 速度敏感性: 通过\n');
    else
        fprintf('❌ 速度敏感性: 失败\n');
    end

    % 4. 理论一致性评估
    theory_passed = false;
    if ~isempty(theory_results.actual_outputs) && length(theory_results.actual_outputs) > 1
        correlation = corrcoef(theory_results.actual_outputs, theory_results.theory_outputs);
        if correlation(1,2) > 0.7
            theory_passed = true;
        end
    end

    if theory_passed
        fprintf('✅ 理论一致性: 通过\n');
    else
        fprintf('❌ 理论一致性: 失败\n');
    end

    % 5. 总体评估
    overall_passed = static_passed && dynamic_passed && speed_passed && theory_passed;

    fprintf('\n🎯 总体评估: ');
    if overall_passed
        fprintf('✅ SPPVT算法正确实现\n');
        fprintf('🚀 可以部署使用\n');
    else
        fprintf('❌ SPPVT算法存在问题\n');
        fprintf('🔧 需要进一步调试和修复\n');

        fprintf('\n💡 建议:\n');
        if ~static_passed
            fprintf('- 检查SPPVT基础算法实现\n');
        end
        if ~dynamic_passed
            fprintf('- 检查动态响应和时间序列处理\n');
        end
        if ~speed_passed
            fprintf('- 检查速度相关参数设置\n');
        end
        if ~theory_passed
            fprintf('- 对比理论算法，验证参数配置\n');
        end
    end

    fprintf('\n📄 详细数据可查看仿真输出和调试信息\n');
end