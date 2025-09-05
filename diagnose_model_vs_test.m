function diagnose_model_vs_test()
%% diagnose_model_vs_test - 诊断是模型问题还是测试脚本问题
% =========================================================================
% 通过三种不同的测试方法来定位问题：
% 1. 单次长仿真 - 所有操作在一次仿真中完成
% 2. 多次短仿真 - 每个操作独立仿真（原方法）
% 3. 持续状态仿真 - 使用初始状态继承
% =========================================================================

    fprintf('\n=== 诊断测试：模型 vs 测试脚本 ===\n\n');

    model_name = 'acc_decision_md';
    if ~bdIsLoaded(model_name)
        load_system(model_name);
    end

    % 测试1: 单次长仿真（所有操作在一次仿真中）
    test_single_long_simulation();

    % 测试2: 多次独立仿真（类似原测试方法）
    test_multiple_independent_simulations();

    % 测试3: 状态继承仿真
    test_state_inheritance_simulation();

    % 测试4: 直接状态检查
    test_direct_state_check();

    fprintf('\n=== 诊断完成 ===\n');
end

%% 测试1: 单次长仿真
function test_single_long_simulation()
    fprintf('【测试1】单次长仿真测试\n');
    fprintf('----------------------------------------\n');
    fprintf('说明：在一次仿真中连续执行多个操作\n\n');

    model_name = 'acc_decision_md';

    % 准备连续的输入序列
    time_points = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

    % 创建输入信号
    command_sequence = [-1, -1, 0, 1, 1, 5, 1];  % 等待->进入S0->增速x2->刹车->恢复
    speed_sequence = [80, 80, 80, 80, 80, 80, 80];

    command_input = timeseries(command_sequence', time_points');
    ego_speed_kmh = timeseries(speed_sequence', time_points');
    has_target = timeseries(false(size(time_points'))', time_points');
    current_distance = timeseries(-ones(size(time_points'))', time_points');
    reset_signal = timeseries(false(size(time_points'))', time_points');

    % 设置到工作空间
    assignin('base', 'command_input', command_input);
    assignin('base', 'ego_speed_kmh', ego_speed_kmh);
    assignin('base', 'has_target', has_target);
    assignin('base', 'current_distance', current_distance);
    assignin('base', 'reset_signal', reset_signal);

    % 配置仿真
    set_param(model_name, 'LoadExternalInput', 'on');
    set_param(model_name, 'ExternalInput', 'command_input, ego_speed_kmh, has_target, current_distance, reset_signal');
    set_param(model_name, 'StopTime', '3.0');
    set_param(model_name, 'SaveOutput', 'on');
    set_param(model_name, 'OutputSaveName', 'yout');
    set_param(model_name, 'SaveFormat', 'Dataset');

    % 运行仿真
    fprintf('运行仿真（0-3秒）...\n');
    sim_out = sim(model_name);

    % 提取输出
    yout = sim_out.yout;
    state_data = yout.getElement(1).Values;  % current_state
    mode_data = yout.getElement(2).Values;   % control_mode
    v3_data = yout.getElement(5).Values;     % V3_kmh
    history_data = yout.getElement(8).Values; % has_history

    % 显示关键时间点的状态
    fprintf('\n时间点分析：\n');
    key_times = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    for t = key_times
        idx = find(state_data.Time >= t, 1);
        if ~isempty(idx)
            fprintf('t=%.1fs: State=S%d, Mode=R%d, V3=%.1f, History=%d\n', ...
                    t, state_data.Data(idx), mode_data.Data(idx), ...
                    v3_data.Data(idx), history_data.Data(idx));
        end
    end

    % 分析结果
    fprintf('\n结果分析：\n');
    if state_data.Data(end) == 0 && history_data.Data(end) == 1
        fprintf('✅ 状态机工作正常：最终在S0，有历史记录\n');
    else
        fprintf('❌ 状态机行为异常：最终状态S%d，历史=%d\n', ...
                state_data.Data(end), history_data.Data(end));
    end

    % 清理工作空间
    evalin('base', 'clear command_input ego_speed_kmh has_target current_distance reset_signal');
    fprintf('\n');
end

%% 测试2: 多次独立仿真
function test_multiple_independent_simulations()
    fprintf('【测试2】多次独立仿真测试\n');
    fprintf('----------------------------------------\n');
    fprintf('说明：每个操作独立仿真（模拟原测试方法）\n\n');

    model_name = 'acc_decision_md';

    % 操作序列
    operations = {
        struct('desc', '初始状态', 'command', -1),
        struct('desc', 'I0进入S0', 'command', 0),
        struct('desc', 'I1增速', 'command', 1),
        struct('desc', 'I5刹车', 'command', 5),
        struct('desc', 'I1恢复', 'command', 1)
    };

    fprintf('执行操作序列：\n');
    for i = 1:length(operations)
        op = operations{i};

        % 准备输入
        command_input = timeseries(op.command, [0; 0.1]);
        ego_speed_kmh = timeseries(80, [0; 0.1]);
        has_target = timeseries(false, [0; 0.1]);
        current_distance = timeseries(-1, [0; 0.1]);
        reset_signal = timeseries(false, [0; 0.1]);

        assignin('base', 'command_input', command_input);
        assignin('base', 'ego_speed_kmh', ego_speed_kmh);
        assignin('base', 'has_target', has_target);
        assignin('base', 'current_distance', current_distance);
        assignin('base', 'reset_signal', reset_signal);

        % 配置和运行
        set_param(model_name, 'LoadExternalInput', 'on');
        set_param(model_name, 'ExternalInput', 'command_input, ego_speed_kmh, has_target, current_distance, reset_signal');
        set_param(model_name, 'StopTime', '0.1');

        sim_out = sim(model_name);

        % 提取结果
        yout = sim_out.yout;
        final_state = yout.getElement(1).Values.Data(end);
        final_mode = yout.getElement(2).Values.Data(end);
        final_v3 = yout.getElement(5).Values.Data(end);

        fprintf('%d. %s -> State=S%d, Mode=R%d, V3=%.1f\n', ...
                i, op.desc, final_state, final_mode, final_v3);
    end

    % 清理工作空间
    evalin('base', 'clear command_input ego_speed_kmh has_target current_distance reset_signal');

    fprintf('\n结果分析：\n');
    fprintf('❓ 每次独立仿真都从初始状态开始，状态无法保持\n\n');
end

%% 测试3: 状态继承仿真
function test_state_inheritance_simulation()
    fprintf('【测试3】状态继承仿真测试\n');
    fprintf('----------------------------------------\n');
    fprintf('说明：尝试在仿真间保持状态\n\n');

    model_name = 'acc_decision_md';

    % 先reset到干净状态
    fprintf('步骤1: Reset到S2...\n');
    reset_signal = timeseries([true; false], [0; 0.05]);
    command_input = timeseries(-1, [0; 0.1]);
    ego_speed_kmh = timeseries(80, [0; 0.1]);
    has_target = timeseries(false, [0; 0.1]);
    current_distance = timeseries(-1, [0; 0.1]);

    assignin('base', 'command_input', command_input);
    assignin('base', 'ego_speed_kmh', ego_speed_kmh);
    assignin('base', 'has_target', has_target);
    assignin('base', 'current_distance', current_distance);
    assignin('base', 'reset_signal', reset_signal);

    set_param(model_name, 'LoadExternalInput', 'on');
    set_param(model_name, 'ExternalInput', 'command_input, ego_speed_kmh, has_target, current_distance, reset_signal');
    set_param(model_name, 'StopTime', '0.1');
    set_param(model_name, 'SaveFinalState', 'on');
    set_param(model_name, 'FinalStateName', 'xFinal');

    sim_out1 = sim(model_name);
    state1 = sim_out1.yout.getElement(1).Values.Data(end);
    fprintf('  Reset后: State=S%d\n', state1);

    % 使用保存的状态继续
    fprintf('\n步骤2: 从S2执行I0进入S0...\n');
    command_input = timeseries(0, [0; 0.1]);
    reset_signal = timeseries(false, [0; 0.1]);

    assignin('base', 'command_input', command_input);
    assignin('base', 'reset_signal', reset_signal);
    assignin('base', 'xInitial', sim_out1.xFinal);

    set_param(model_name, 'LoadInitialState', 'on');
    set_param(model_name, 'InitialState', 'xInitial');

    sim_out2 = sim(model_name);
    state2 = sim_out2.yout.getElement(1).Values.Data(end);
    mode2 = sim_out2.yout.getElement(2).Values.Data(end);
    fprintf('  I0后: State=S%d, Mode=R%d\n', state2, mode2);

    % 继续测试
    fprintf('\n步骤3: 从S0执行I1增速...\n');
    command_input = timeseries(1, [0; 0.1]);
    assignin('base', 'command_input', command_input);
    assignin('base', 'xInitial', sim_out2.xFinal);

    sim_out3 = sim(model_name);
    state3 = sim_out3.yout.getElement(1).Values.Data(end);
    mode3 = sim_out3.yout.getElement(2).Values.Data(end);
    v3_3 = sim_out3.yout.getElement(5).Values.Data(end);
    fprintf('  I1后: State=S%d, Mode=R%d, V3=%.1f\n', state3, mode3, v3_3);

    % 清理
    evalin('base', 'clear command_input ego_speed_kmh has_target current_distance reset_signal xInitial xFinal');
    set_param(model_name, 'LoadInitialState', 'off');

    fprintf('\n结果分析：\n');
    if state2 == 0 && state3 == 0 && v3_3 > 50
        fprintf('✅ 状态继承工作正常\n');
    else
        fprintf('❌ 状态继承存在问题\n');
    end
    fprintf('\n');
end

%% 测试4: 直接状态检查
function test_direct_state_check()
    fprintf('【测试4】直接状态检查\n');
    fprintf('----------------------------------------\n');
    fprintf('说明：检查Stateflow内部配置\n\n');

    % 获取Stateflow对象
    rt = sfroot;
    chart = rt.find('-isa', 'Stateflow.Chart', 'Path', 'acc_decision_md/ACC_Decision_Chart');

    % 检查默认转移
    transitions = chart.find('-isa', 'Stateflow.Transition');
    default_trans = [];
    for i = 1:length(transitions)
        if isempty(transitions(i).Source)
            default_trans = transitions(i);
            break;
        end
    end

    if ~isempty(default_trans) && ~isempty(default_trans.Destination)
        fprintf('默认转移目标: %s\n', default_trans.Destination.Name);

        % 检查默认转移的动作
        if ~isempty(default_trans.LabelString)
            fprintf('默认转移动作: %s\n', default_trans.LabelString);
        end
    else
        fprintf('❌ 没有找到有效的默认转移\n');
    end

    % 检查S0的自转移
    fprintf('\nS0状态的自转移:\n');
    s0 = chart.find('-isa', 'Stateflow.State', 'Name', 'S0_IN_CONTROL');
    if ~isempty(s0)
        s0_transitions = transitions(arrayfun(@(t) ~isempty(t.Source) && strcmp(t.Source.Name, 'S0_IN_CONTROL') && ...
                                             ~isempty(t.Destination) && strcmp(t.Destination.Name, 'S0_IN_CONTROL'), transitions));
        fprintf('找到 %d 个S0自转移\n', length(s0_transitions));
        for i = 1:min(3, length(s0_transitions))
            label = s0_transitions(i).LabelString;
            % 提取条件
            cond_match = regexp(label, '\[(.*?)\]', 'tokens');
            if ~isempty(cond_match)
                fprintf('  转移%d条件: %s\n', i, cond_match{1}{1});
            end
        end
    end

    % 检查S2到S0的转移
    fprintf('\nS2到S0的转移:\n');
    s2_to_s0 = transitions(arrayfun(@(t) ~isempty(t.Source) && contains(t.Source.Name, 'S2') && ...
                                         ~isempty(t.Destination) && contains(t.Destination.Name, 'S0'), transitions));
    if ~isempty(s2_to_s0)
        for i = 1:length(s2_to_s0)
            label = s2_to_s0(i).LabelString;
            cond_match = regexp(label, '\[(.*?)\]', 'tokens');
            if ~isempty(cond_match)
                fprintf('  条件: %s\n', cond_match{1}{1});
            end
        end
    else
        fprintf('  ❌ 没有找到S2到S0的转移\n');
    end

    % 检查执行顺序
    fprintf('\n转移执行顺序:\n');
    exec_orders = arrayfun(@(t) t.ExecutionOrder, transitions);
    unique_orders = unique(exec_orders(exec_orders > 0));
    if ~isempty(unique_orders)
        fprintf('  定义的执行顺序: %s\n', mat2str(unique_orders));
    else
        fprintf('  ⚠️ 没有明确定义执行顺序\n');
    end

    fprintf('\n');
end