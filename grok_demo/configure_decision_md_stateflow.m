function configure_decision_md_stateflow()

%       实现4个核心状态(S0-S3)和7种指令(I0-I6)的完整转移逻辑。

    model_name = 'acc_decision_md';
    chart_path = [model_name '/ACC_Decision_Chart'];

    fprintf('=== 配置decision.md Stateflow状态机 (根状态注入法) ===\n');

    if ~bdIsLoaded(model_name)
        error('模型 %s 未加载，请先运行 create_decision_md_simulink.m', model_name);
    end

    fprintf('获取Stateflow Chart对象...\n');
    try
        block_handle = get_param(chart_path, 'Handle');
        chart_id = sfprivate('block2chart', block_handle);
        rt = sfroot;
        chart = rt.idToHandle(chart_id);
    catch ME
        error('获取Stateflow Chart对象失败: %s', ME.message);
    end

    if isempty(chart)
        error('无法找到Stateflow Chart对象，请检查模型结构');
    end
    fprintf('✅ Chart对象获取成功 (ID: %d)\n', chart_id);

    fprintf('清除现有内容...\n');
    % 安全地清除所有现有内容
    states_to_delete = chart.find('-isa', 'Stateflow.State');
    if ~isempty(states_to_delete), states_to_delete.delete; end

    transitions_to_delete = chart.find('-isa', 'Stateflow.Transition');
    if ~isempty(transitions_to_delete), transitions_to_delete.delete; end

    functions_to_delete = chart.find('-isa', 'Stateflow.EMFunction');
    if ~isempty(functions_to_delete), functions_to_delete.delete; end

    % --- 按顺序执行配置 ---
    %create_decision_md_states(chart);
    %create_decision_md_transitions(chart);
    add_chart_logic_and_functions(chart);

    save_system(model_name);

    fprintf('\n🎉 decision.md状态机配置完成 (根状态注入法)！\n');
end

function create_decision_md_states(chart)
    fprintf('创建4个核心状态...\n');

    state_s0 = Stateflow.State(chart);
    state_s0.Name = 'S0_IN_CONTROL';
    state_s0.Position = [100 100 200 150];
    state_s0.LabelString = sprintf('S0_IN_CONTROL\nentry: %s', ...
        'current_state=0; acc_active=true; control_enabled=true; is_in_control=true;');

    state_s1 = Stateflow.State(chart);
    state_s1.Name = 'S1_ADAPTIVE_HISTORY_STANDBY';
    state_s1.Position = [350 100 220 150];
    state_s1.LabelString = sprintf('S1_ADAPTIVE_HISTORY_STANDBY\nentry: %s', ...
        'current_state=1; acc_active=false; control_enabled=false; is_in_control=false;');

    state_s2 = Stateflow.State(chart);
    state_s2.Name = 'S2_ADAPTIVE_NO_HISTORY_STANDBY';
    state_s2.Position = [100 300 220 150];
    state_s2.LabelString = sprintf('S2_ADAPTIVE_NO_HISTORY_STANDBY\nentry: %s', ...
        'current_state=2; acc_active=false; control_enabled=false; is_in_control=false; has_history=false;');

    state_s3 = Stateflow.State(chart);
    state_s3.Name = 'S3_LOW_SPEED';
    state_s3.Position = [350 300 200 150];
    state_s3.LabelString = sprintf('S3_LOW_SPEED\nentry: %s', ...
        'current_state=3; acc_active=false; control_enabled=false; is_in_control=false;');

    fprintf('✅ 4个核心状态创建完成\n');
end

function create_decision_md_transitions(chart)
    fprintf('创建状态转移逻辑...\n');

    s0 = chart.find('-isa', 'Stateflow.State', 'Name', 'S0_IN_CONTROL');
    s1 = chart.find('-isa', 'Stateflow.State', 'Name', 'S1_ADAPTIVE_HISTORY_STANDBY');
    s2 = chart.find('-isa', 'Stateflow.State', 'Name', 'S2_ADAPTIVE_NO_HISTORY_STANDBY');
    s3 = chart.find('-isa', 'Stateflow.State', 'Name', 'S3_LOW_SPEED');

    % S0在控状态的转移
    create_transition(s0, s0, 'command_input == 0', 'control_mode = 1; V3_kmh = max(V2_KMH+1, V3_kmh - SPEED_STEP); message_code = 101;');
    create_transition(s0, s0, 'command_input == 1', 'control_mode = 2; V3_kmh = min(120, V3_kmh + SPEED_STEP); message_code = 102;');
    create_transition(s0, s0, 'command_input == 2', 'control_mode = 3; G1_m = max(5, G1_m - DISTANCE_STEP); message_code = 103;');
    create_transition(s0, s0, 'command_input == 3', 'control_mode = 4; G1_m = min(50, G1_m + DISTANCE_STEP); message_code = 104;');
    create_transition(s0, s0, 'command_input == 4', 'control_mode = 7; message_code = 107;');
    create_transition(s0, s1, 'command_input == 5', 'control_mode = 8; save_history(); message_code = 108;');
    create_transition(s0, s1, 'command_input == 6', 'control_mode = 8; save_history(); message_code = 109;');

    % S1适速有史待命状态的转移
    create_transition(s1, s0, 'command_input == 1', 'control_mode = 6; restore_history(); message_code = 206;');
    create_transition(s1, s1, 'command_input == 0', 'control_mode = 5; message_code = 205;');
    create_transition(s1, s1, 'command_input >= 2 && command_input <= 6', 'control_mode = 8; message_code = 200 + command_input;');

    % S2适速无史待命状态的转移
    create_transition(s2, s2, 'command_input == 0', 'control_mode = 5; message_code = 305;');
    create_transition(s2, s2, 'command_input >= 1 && command_input <= 6', 'control_mode = 8; message_code = 300 + command_input;');

    % S3低速状态的转移
    create_transition(s3, s3, 'command_input >= 0 && command_input <= 6', 'control_mode = 8; message_code = 400 + command_input;');

    % 自动速度状态转移
    create_transition(s1, s3, 'ego_speed_kmh < V1_KMH', 'message_code = 999;');
    create_transition(s2, s3, 'ego_speed_kmh < V1_KMH', 'message_code = 999;');
    create_transition(s3, s1, 'ego_speed_kmh >= V1_KMH && has_history', 'message_code = 901;');
    create_transition(s3, s2, 'ego_speed_kmh >= V1_KMH && ~has_history', 'message_code = 902;');

    % 创建默认初始转移
    default_trans = Stateflow.Transition(chart);
    default_trans.Destination = s2;
    fprintf('✅ 状态转移逻辑配置完成\n');
end

function create_transition(source, dest, condition, action)
    trans = Stateflow.Transition(source.Chart);
    trans.Source = source;
    trans.Destination = dest;
    trans.LabelString = sprintf('[%s] {%s}', condition, action);
end

function add_chart_logic_and_functions(chart)
    fprintf('添加Chart级别逻辑和函数 (使用 Enclosing Superstate)...\n');

    % 创建 Enclosing Superstate
    superstate = Stateflow.State(chart);
    superstate.Name = 'RootSuperstate';
    superstate.Position = [50 50 600 500];  % 大足够包含所有子状态
    superstate.Decomposition = 'EXCLUSIVE_OR';  % 使用独占 OR，因为核心状态是互斥的（根据 md 逻辑）
    superstate.LabelString = strjoin({ ...
        'during:', ...
        '  %% 在每个时间步执行的逻辑', ...
        '  if (reset_signal)', ...
        '    reset_all_parameters();', ...
        '  end;', ...
        '  pending_distance_adj = pending_adj;', ...
        '  if (V3_kmh <= 0), V3_kmh = 50.0; end;', ...
        '  if (G1_m <= 0), G1_m = 15.0; end;', ...
        '  if (G2_s <= 0), G2_s = 2.0; end;' ...
    }, '\n');
    fprintf('✅ Superstate 创建并注入 during 逻辑\n');

    % 在 Superstate 内创建 4 个核心状态（复用原 create_decision_md_states 逻辑，但传入 superstate）
    % 注意：原 create_decision_md_states 需修改为接受 parent 参数，或在这里手动创建
    state_s0 = Stateflow.State(superstate);
    state_s0.Name = 'S0_IN_CONTROL';
    state_s0.Position = [100 100 200 150];  % 相对 Superstate 的位置
    state_s0.LabelString = sprintf('S0_IN_CONTROL\nentry: %s', ...
        'current_state=0; acc_active=true; control_enabled=true; is_in_control=true;');

    state_s1 = Stateflow.State(superstate);
    state_s1.Name = 'S1_ADAPTIVE_HISTORY_STANDBY';
    state_s1.Position = [350 100 220 150];
    state_s1.LabelString = sprintf('S1_ADAPTIVE_HISTORY_STANDBY\nentry: %s', ...
        'current_state=1; acc_active=false; control_enabled=false; is_in_control=false;');

    state_s2 = Stateflow.State(superstate);
    state_s2.Name = 'S2_ADAPTIVE_NO_HISTORY_STANDBY';
    state_s2.Position = [100 300 220 150];
    state_s2.LabelString = sprintf('S2_ADAPTIVE_NO_HISTORY_STANDBY\nentry: %s', ...
        'current_state=2; acc_active=false; control_enabled=false; is_in_control=false; has_history=false;');

    state_s3 = Stateflow.State(superstate);
    state_s3.Name = 'S3_LOW_SPEED';
    state_s3.Position = [350 300 200 150];
    state_s3.LabelString = sprintf('S3_LOW_SPEED\nentry: %s', ...
        'current_state=3; acc_active=false; control_enabled=false; is_in_control=false;');

    fprintf('✅ 4个核心状态在 Superstate 内创建完成\n');

    % 重新创建状态转移逻辑（因为状态现在在 Superstate 内，传入新的状态对象）
    s0 = superstate.find('-isa', 'Stateflow.State', 'Name', 'S0_IN_CONTROL');
    s1 = superstate.find('-isa', 'Stateflow.State', 'Name', 'S1_ADAPTIVE_HISTORY_STANDBY');
    s2 = superstate.find('-isa', 'Stateflow.State', 'Name', 'S2_ADAPTIVE_NO_HISTORY_STANDBY');
    s3 = superstate.find('-isa', 'Stateflow.State', 'Name', 'S3_LOW_SPEED');

    % S0在控状态的转移
    create_transition(s0, s0, 'command_input == 0', 'control_mode = 1; V3_kmh = max(V2_KMH+1, V3_kmh - SPEED_STEP); message_code = 101;');
    create_transition(s0, s0, 'command_input == 1', 'control_mode = 2; V3_kmh = min(120, V3_kmh + SPEED_STEP); message_code = 102;');
    create_transition(s0, s0, 'command_input == 2', 'control_mode = 3; G1_m = max(5, G1_m - DISTANCE_STEP); message_code = 103;');
    create_transition(s0, s0, 'command_input == 3', 'control_mode = 4; G1_m = min(50, G1_m + DISTANCE_STEP); message_code = 104;');
    create_transition(s0, s0, 'command_input == 4', 'control_mode = 7; message_code = 107;');
    create_transition(s0, s1, 'command_input == 5', 'control_mode = 8; save_history(); message_code = 108;');
    create_transition(s0, s1, 'command_input == 6', 'control_mode = 8; save_history(); message_code = 109;');

    % S1适速有史待命状态的转移
    create_transition(s1, s0, 'command_input == 1', 'control_mode = 6; restore_history(); message_code = 206;');
    create_transition(s1, s1, 'command_input == 0', 'control_mode = 5; message_code = 205;');
    create_transition(s1, s1, 'command_input >= 2 && command_input <= 6', 'control_mode = 8; message_code = 200 + command_input;');

    % S2适速无史待命状态的转移
    create_transition(s2, s2, 'command_input == 0', 'control_mode = 5; message_code = 305;');
    create_transition(s2, s2, 'command_input >= 1 && command_input <= 6', 'control_mode = 8; message_code = 300 + command_input;');


    % S3低速状态的转移
    % 增加 ego_speed_kmh < V1_KMH 条件，确保只有在低速时才保持S3，避免遮蔽高速时向S1/S2的转移
    create_transition(s3, s3, '[command_input >= 0 && command_input <= 6] && [ego_speed_kmh < V1_KMH]', 'control_mode = 8; message_code = 400 + command_input;');

    % 自动速度状态转移
    create_transition(s1, s3, 'ego_speed_kmh < V1_KMH', 'message_code = 999;');
    create_transition(s2, s3, 'ego_speed_kmh < V1_KMH', 'message_code = 999;');
    create_transition(s3, s1, 'ego_speed_kmh >= V1_KMH && has_history', 'message_code = 901;');
    create_transition(s3, s2, 'ego_speed_kmh >= V1_KMH && ~has_history', 'message_code = 902;');

    % 创建默认初始转移（默认到 S2）
    default_trans = Stateflow.Transition(superstate);  % 默认转移在 Superstate 级别
    default_trans.Destination = s2;
    default_trans.SourceOClock = 0;  % 可选位置调整
    fprintf('✅ 状态转移逻辑在 Superstate 内配置完成\n');

    % 添加图形化函数 (EMFunction，原代码不变)
    fprintf('添加图形化函数...\n');

    functions_to_create = [
        struct('Name', 'save_history', 'Script', {{
            'function save_history()', ...
            '% 保存当前设定为历史', ...
            'has_history = true;', ...
            'history_V3_kmh = V3_kmh;', ...
            'history_G1_m = G1_m;', ...
            'history_G2_s = G2_s;'
        }}),
        struct('Name', 'restore_history', 'Script', {{
            'function restore_history()', ...
            '% 恢复历史设定', ...
            'if (has_history)', ...
            '  V3_kmh = history_V3_kmh;', ...
            '  G1_m = history_G1_m;', ...
            '  G2_s = history_G2_s;', ...
            '  if (pending_adj ~= 0)', ...
            '    G1_m = max(5.0, G1_m + pending_adj);', ...
            '    pending_adj = 0;', ...
            '  end;', ...
            'end;'
        }}),
        struct('Name', 'reset_all_parameters', 'Script', {{
            'function reset_all_parameters()', ...
            '% 重置所有参数到初始值', ...
            'has_history = false;', ...
            'history_V3_kmh = 50.0;', ...
            'history_G1_m = 15.0;', ...
            'history_G2_s = 2.0;', ...
            'pending_adj = 0;', ...
            'V3_kmh = 50.0;', ...
            'G1_m = 15.0;', ...
            'G2_s = 2.0;', ...
            'control_mode = 0;', ...
            'message_code = 0;'
        }})
    ];

    for i = 1:length(functions_to_create)
        func_def = functions_to_create(i);
        func_obj = Stateflow.EMFunction(chart);
        func_obj.Name = func_def.Name;
        func_obj.Script = strjoin(func_def.Script, '\n');
    end

    fprintf('✅ 图形化函数配置完成\n');
end