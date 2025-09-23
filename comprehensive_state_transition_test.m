function comprehensive_state_transition_test()
%% 全面状态转移测试 - 覆盖所有可能的状态转移场景
%  根据decision_function.m源码分析设计的完整测试套件

fprintf('🧪 启动全面状态转移测试...\n');

model_name = 'ACC_Decision_SPPVT_Integrated';

try
    % 加载模型和总线定义
    load_system(model_name);
    create_decision_sppvt_bus();

    % 测试场景定义 - 包含所有可能的状态转移
    test_scenarios = {
        % 格式: {场景名, 初始状态设置, 测试输入, 期望输出, 描述}

        %% 1. S2初始状态测试 (系统启动时的默认状态)
        {'S2_I0_当速启控', 'default', [1, true], [0, 1, true], 'S2状态I0指令→S0+R5+Control=1'},
        {'S2_I1_系统待命', 'default', [2, true], [2, 8, false], 'S2状态I1指令→S2+R8+Control=0'},
        {'S2_I2_系统待命', 'default', [3, true], [2, 8, false], 'S2状态I2指令→S2+R8+Control=0'},
        {'S2_I3_系统待命', 'default', [4, true], [2, 8, false], 'S2状态I3指令→S2+R8+Control=0'},
        {'S2_I4_系统待命', 'default', [5, true], [2, 8, false], 'S2状态I4指令→S2+R8+Control=0'},
        {'S2_I5_系统待命', 'default', [6, true], [2, 8, false], 'S2状态I5指令→S2+R8+Control=0'},
        {'S2_I6_系统待命', 'default', [7, true], [2, 8, false], 'S2状态I6指令→S2+R8+Control=0'},
        {'S2_NONE_无指令', 'default', [1, false], [2, 8, false], 'S2状态无有效指令→S2+R8+Control=0'},

        %% 2. S0在控状态测试 (从S2+I0进入S0后的各种指令)
        {'S0_I0_速度降低', 'in_control', [1, true], [0, 1, true], 'S0状态I0指令→S0+R1+Control=1'},
        {'S0_I1_速度增加', 'in_control', [2, true], [0, 2, true], 'S0状态I1指令→S0+R2+Control=1'},
        {'S0_I2_时距降低', 'in_control', [3, true], [0, 3, true], 'S0状态I2指令→S0+R3+Control=1'},
        {'S0_I3_时距增加', 'in_control', [4, true], [0, 4, true], 'S0状态I3指令→S0+R4+Control=1'},
        {'S0_I4_扭矩仲裁', 'in_control', [5, true], [0, 7, true], 'S0状态I4指令→S0+R7+Control=1'},
        {'S0_I5_刹车退出', 'in_control', [6, true], [1, 8, false], 'S0状态I5指令→S1+R8+Control=0'},
        {'S0_I6_取消退出', 'in_control', [7, true], [1, 8, false], 'S0状态I6指令→S1+R8+Control=0'},
        {'S0_NONE_维持控制', 'in_control', [1, false], [0, 1, true], 'S0状态无有效指令→S0+R1(保持之前决策)+Control=1'},

        %% 3. S1适速有史待命状态测试 (从S0退出后的状态)
        {'S1_I0_无继控制', 'history_standby', [1, true], [0, 5, true], 'S1状态I0指令→S0+R5+Control=1'},
        {'S1_I1_继承控制', 'history_standby', [2, true], [0, 6, true], 'S1状态I1指令→S0+R6+Control=1'},
        {'S1_I2_系统待命', 'history_standby', [3, true], [1, 8, false], 'S1状态I2指令→S1+R8+Control=0'},
        {'S1_I3_系统待命', 'history_standby', [4, true], [1, 8, false], 'S1状态I3指令→S1+R8+Control=0'},
        {'S1_I4_系统待命', 'history_standby', [5, true], [1, 8, false], 'S1状态I4指令→S1+R8+Control=0'},
        {'S1_I5_系统待命', 'history_standby', [6, true], [1, 8, false], 'S1状态I5指令→S1+R8+Control=0'},
        {'S1_I6_系统待命', 'history_standby', [7, true], [1, 8, false], 'S1状态I6指令→S1+R8+Control=0'},
        {'S1_NONE_维持待命', 'history_standby', [1, false], [1, 8, false], 'S1状态无有效指令→S1+R8+Control=0'},

        %% 4. S3低速状态测试 (速度低于V_min_kmh时的强制状态)
        {'S3_LOW_I0_低速待命', 'low_speed', [1, true], [3, 8, false], 'S3低速状态I0指令→S3+R8+Control=0'},
        {'S3_LOW_I1_低速待命', 'low_speed', [2, true], [3, 8, false], 'S3低速状态I1指令→S3+R8+Control=0'},
        {'S3_LOW_I6_低速待命', 'low_speed', [7, true], [3, 8, false], 'S3低速状态I6指令→S3+R8+Control=0'},
        {'S3_LOW_NONE_低速待命', 'low_speed', [1, false], [3, 8, false], 'S3低速状态无指令→S3+R8+Control=0'},

        %% 5. 速度自动转换测试
        {'S3_TO_S2_速度恢复', 'low_to_normal', [1, false], [2, 8, false], 'S3→S2速度恢复(无历史)'},
        {'S3_TO_S1_速度恢复', 'low_to_normal_with_history', [1, false], [1, 8, false], 'S3→S1速度恢复(有历史)'},

        %% 6. 边界条件测试
        {'NORMAL_TO_S3_速度过低', 'normal_to_low', [1, true], [3, 8, false], '正常状态→S3速度过低'},
        {'S0_TO_S3_控制中速度过低', 'control_to_low', [2, true], [3, 8, false], 'S0控制中→S3速度过低'},
    };

    % 执行所有测试场景
    passed_count = 0;
    total_count = length(test_scenarios);

    fprintf('\n📋 开始执行 %d 个测试场景:\n', total_count);
    fprintf('场景名称                        期望输出                 实际输出            结果\n');
    fprintf('--------------------------------------------------------------------------------\n');

    for i = 1:length(test_scenarios)
        scenario = test_scenarios{i};
        scenario_name = scenario{1};
        initial_state = scenario{2};
        test_input = scenario{3};  % [command_type, command_active]
        expected_output = scenario{4}; % [expected_state, expected_decision, expected_control]
        description = scenario{5};

        % 运行单个测试场景
        [actual_state, actual_decision, actual_control, sppvt_output] = ...
            run_single_scenario(model_name, initial_state, test_input);

        % 验证结果
        state_match = (actual_state == expected_output(1));
        decision_match = (actual_decision == expected_output(2));
        control_match = (actual_control == expected_output(3));
        test_passed = state_match && decision_match && control_match;

        if test_passed
            passed_count = passed_count + 1;
            result_icon = '✅通过';
        else
            result_icon = '❌失败';
        end

        % 格式化输出
        expected_str = sprintf('S%d+R%d+C%d', expected_output(1), expected_output(2), expected_output(3));
        actual_str = sprintf('S%d+R%d+C%d', actual_state, actual_decision, actual_control);

        fprintf('%-30s  %-20s  %-15s %s\n', ...
                scenario_name, expected_str, actual_str, result_icon);
    end

    %% 测试结果汇总
    fprintf('\n📊 测试结果汇总:\n');
    fprintf('总测试数: %d\n', total_count);
    fprintf('通过数: %d\n', passed_count);
    fprintf('失败数: %d\n', total_count - passed_count);
    fprintf('通过率: %.1f%%\n', (passed_count / total_count) * 100);

    if passed_count == total_count
        fprintf('\n🎉 所有状态转移测试通过! 决策逻辑完全正确!\n');
    else
        fprintf('\n⚠️  发现 %d 个状态转移问题，需要进一步分析:\n', total_count - passed_count);

        % 分析失败原因
        analyze_failure_patterns(test_scenarios, model_name);
    end

catch ME
    fprintf('❌ 全面状态转移测试失败: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('错误位置: %s:%d\n', ME.stack(1).file, ME.stack(1).line);
    end
end

fprintf('\n🏁 全面状态转移测试完成\n');
end

function [actual_state, actual_decision, actual_control, sppvt_output] = ...
    run_single_scenario(model_name, initial_state_type, test_input)
%% 运行单个测试场景

command_type = test_input(1);
command_active = test_input(2);

% 根据初始状态类型设置不同的初始条件
switch initial_state_type
    case 'default' % S2初始状态
        ego_speed = 50.0; % 正常速度
        V_min = 30.0;
        simulation_time = 0.1; % 短时间仿真

    case 'in_control' % S0在控状态 (需要先通过S2+I0进入)
        ego_speed = 50.0;
        V_min = 30.0;
        simulation_time = 0.15; % 需要两步: S2→S0, 然后测试S0状态

    case 'history_standby' % S1有史待命状态 (需要先进入S0再退出)
        ego_speed = 50.0;
        V_min = 30.0;
        simulation_time = 0.2; % 需要三步: S2→S0→S1, 然后测试S1状态

    case 'low_speed' % S3低速状态
        ego_speed = 25.0; % 低于V_min_kmh
        V_min = 30.0;
        simulation_time = 0.1;

    case 'low_to_normal' % 从低速恢复到正常速度(无历史)
        ego_speed = 35.0; % 高于V_min_kmh
        V_min = 30.0;
        simulation_time = 0.1;

    case 'low_to_normal_with_history' % 从低速恢复到正常速度(有历史)
        ego_speed = 35.0;
        V_min = 30.0;
        simulation_time = 0.25; % 需要先建立历史: S2→S0→S3→恢复

    case 'normal_to_low' % 从正常速度降到低速
        ego_speed = 25.0; % 在仿真过程中降速
        V_min = 30.0;
        simulation_time = 0.1;

    case 'control_to_low' % 从控制状态降到低速
        ego_speed = 25.0;
        V_min = 30.0;
        simulation_time = 0.15;

    otherwise
        ego_speed = 50.0;
        V_min = 30.0;
        simulation_time = 0.1;
end

% 创建时间序列
time_points = 0:0.05:simulation_time;
num_points = length(time_points);

% 创建输入数据结构
input_data = struct();

% 根据测试类型创建速度变化曲线
if strcmp(initial_state_type, 'low_to_normal_with_history')
    % S3→S1测试需要模拟完整的速度变化过程
    % 阶段1(0-0.10): 50km/h建立历史 → S2→S0
    % 阶段2(0.10-0.15): 25km/h强制进入S3
    % 阶段3(0.15-0.25): 35km/h恢复正常 → S3→S1
    stage1_points = round(num_points * 0.4);  % 建立历史阶段
    stage2_points = round(num_points * 0.2);  % 低速阶段
    stage3_points = num_points - stage1_points - stage2_points; % 恢复阶段

    speed_profile = [repmat(50.0, 1, stage1_points), ...
                     repmat(25.0, 1, stage2_points), ...
                     repmat(35.0, 1, stage3_points)];
else
    % 其他测试保持恒定速度
    speed_profile = repmat(ego_speed, 1, num_points);
end

% 基础输入数据
input_data.ego_speed_kmh = timeseries(speed_profile, time_points, 'Name', 'ego_speed_kmh');
input_data.ego_speed_ms = timeseries(speed_profile/3.6, time_points, 'Name', 'ego_speed_ms');
input_data.manual_throttle_active = timeseries(false(1, num_points), time_points, 'Name', 'manual_throttle_active');
input_data.control_error = timeseries(repmat(1.5, 1, num_points), time_points, 'Name', 'control_error');
input_data.control_mode_flag = timeseries(int32(ones(1, num_points)), time_points, 'Name', 'control_mode_flag');
input_data.V_target_kmh = timeseries(repmat(50.0, 1, num_points), time_points, 'Name', 'V_target_kmh');
input_data.V_min_kmh = timeseries(repmat(V_min, 1, num_points), time_points, 'Name', 'V_min_kmh');
input_data.G2_s = timeseries(repmat(2.0, 1, num_points), time_points, 'Name', 'G2_s');
input_data.timestamp = timeseries(time_points, time_points, 'Name', 'timestamp');

% 根据不同的初始状态类型设置命令序列
switch initial_state_type
    case 'in_control'
        % 第一步: S2→S0 (I0激活控制)
        % 第二步: 测试S0状态下的指令
        command_sequence = [1, command_type]; % 先I0进入控制，再测试目标指令
        active_sequence = [true, command_active];

    case 'history_standby'
        % S2→S0→S1→测试S1状态
        command_sequence = [1, 6, command_type]; % I0进控制→I6退出→测试指令
        active_sequence = [true, true, command_active];

    case 'low_to_normal_with_history'
        % 建立历史: S2→S0, 然后模拟低速恢复
        % 阶段1: I0建立历史，阶段2: 无指令进入低速，阶段3: 测试速度恢复过程
        command_sequence = [1, 1, command_type]; % 建立历史，保持，测试恢复
        active_sequence = [true, false, command_active]; % 第二阶段不激活指令

    case 'control_to_low'
        % S2→S0然后速度降低
        command_sequence = [1, command_type]; % 先进入控制再测试
        active_sequence = [true, command_active];

    otherwise
        % 单步测试
        command_sequence = command_type;
        active_sequence = command_active;
end

% 扩展命令序列以匹配时间点数量
command_data = int32(ones(1, num_points));
active_data = false(1, num_points);

if length(command_sequence) == 1
    command_data(:) = command_sequence;
    active_data(:) = active_sequence;
else
    % 多步序列：平均分配时间
    steps_per_command = max(1, floor(num_points / length(command_sequence)));
    for i = 1:length(command_sequence)
        start_idx = (i-1) * steps_per_command + 1;
        end_idx = min(i * steps_per_command, num_points);
        command_data(start_idx:end_idx) = command_sequence(i);
        active_data(start_idx:end_idx) = active_sequence(i);
    end
end

input_data.command_type = timeseries(command_data, time_points, 'Name', 'command_type');
input_data.command_active = timeseries(active_data, time_points, 'Name', 'command_active');

% 设置时间单位
field_names = fieldnames(input_data);
for i = 1:length(field_names)
    input_data.(field_names{i}).TimeInfo.Units = 'seconds';
end

% 配置仿真参数
set_param(model_name, 'StopTime', num2str(simulation_time));
set_param(model_name, 'FixedStep', '0.05');
set_param(model_name, 'SaveOutput', 'on');
set_param(model_name, 'OutputSaveName', 'yout');
set_param(model_name, 'SaveFormat', 'Dataset');
set_param(model_name, 'LoadExternalInput', 'on');
set_param(model_name, 'ExternalInput', 'input_data');

assignin('base', 'input_data', input_data);

% 运行仿真
sim_out = sim(model_name);

% 解析输出
actual_state = -1;
actual_decision = -1;
actual_control = -1;
sppvt_output = 0.0;

if isa(sim_out, 'Simulink.SimulationOutput') && isprop(sim_out, 'yout')
    output_data = sim_out.yout;

    if isa(output_data, 'Simulink.SimulationData.Dataset') && output_data.numElements >= 1
        element = output_data{1};

        if isa(element, 'Simulink.SimulationData.Signal') && isstruct(element.Values)
            values_struct = element.Values;

            % 获取最后时刻的输出值
            if isfield(values_struct, 'current_state') && isa(values_struct.current_state, 'timeseries')
                actual_state = double(values_struct.current_state.Data(end));
            end

            if isfield(values_struct, 'current_decision') && isa(values_struct.current_decision, 'timeseries')
                actual_decision = double(values_struct.current_decision.Data(end));
            end

            if isfield(values_struct, 'control_enabled') && isa(values_struct.control_enabled, 'timeseries')
                actual_control = logical(values_struct.control_enabled.Data(end));
            end

            if isfield(values_struct, 'sppvt_control_output') && isa(values_struct.sppvt_control_output, 'timeseries')
                sppvt_output = double(values_struct.sppvt_control_output.Data(end));
            end
        end
    end
end

end

function analyze_failure_patterns(test_scenarios, model_name)
%% 分析失败模式，找出根本原因

fprintf('\n🔍 分析失败模式...\n');

% 重新运行几个关键失败场景，详细分析输出
key_scenarios = {'S2_I1_系统待命', 'S0_NONE_维持控制', 'S1_I0_无继控制'};

for i = 1:length(key_scenarios)
    scenario_name = key_scenarios{i};

    % 找到对应场景
    for j = 1:length(test_scenarios)
        if strcmp(test_scenarios{j}{1}, scenario_name)
            fprintf('\n📝 详细分析场景: %s\n', scenario_name);

            initial_state = test_scenarios{j}{2};
            test_input = test_scenarios{j}{3};

            % 运行详细分析
            run_detailed_analysis(model_name, initial_state, test_input, scenario_name);
            break;
        end
    end
end

end

function run_detailed_analysis(model_name, initial_state_type, test_input, scenario_name)
%% 运行详细分析，输出状态转移的每一步

fprintf('  输入: command_type=%d, command_active=%d\n', test_input(1), test_input(2));
fprintf('  初始状态类型: %s\n', initial_state_type);

end