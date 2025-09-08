function configure_decision_md_stateflow_syntax_fix()
% =========================================================================
% configure_decision_md_stateflow_syntax_fix (版本 15.0 - 保留原功能语法修复版)
%
% 核心修复：
% 1. 修复Stateflow语法错误，但保留所有原有功能
% 2. 保持原有的参数验证、输入处理、历史机制等所有逻辑
% 3. 只修复语法问题，不删除任何功能特性
% 4. 使用Stateflow兼容的条件语法
% =========================================================================

    model_name = 'acc_decision_md';
    chart_path = [model_name '/ACC_Decision_Chart'];

    fprintf('=== 开始配置 decision.md Stateflow Chart (保留原功能语法修复版) ===\n');

    % 1. 获取 Stateflow Chart 对象
    if ~bdIsLoaded(model_name)
        error('模型 %s 未加载，请先运行 create_decision_md_simulink.m', model_name);
    end
    fprintf('获取 Stateflow Chart 对象...\n');
    try
        rt = sfroot;
        chart = rt.find('-isa', 'Stateflow.Chart', 'Path', chart_path);
        if isempty(chart)
            error('找不到Stateflow Chart');
        end
    catch ME
        error('获取 Stateflow Chart 对象失败: %s', ME.message);
    end
    fprintf('✅ Chart 对象获取成功 (ID: %d)\n', chart.Id);

    % 2. 清理 Chart 中的所有现有内容
    fprintf('清理现有内容...\n');
    items_to_delete = chart.find('-isa', 'Stateflow.State');
    if ~isempty(items_to_delete), items_to_delete.delete; end
    items_to_delete = chart.find('-isa', 'Stateflow.Transition');
    if ~isempty(items_to_delete), items_to_delete.delete; end
    items_to_delete = chart.find('-isa', 'Stateflow.EMFunction');
    if ~isempty(items_to_delete), items_to_delete.delete; end
    fprintf('✅ 现有内容已清除\n');

    % 3. 调用统一的构建函数来创建整个状态机
    build_complete_state_machine_preserve_all(chart);

    % 4. 保存模型
    save_system(model_name);
    fprintf('\n🎉 decision.md 状态机配置完成 (保留原功能语法修复版)！\n');
end

function build_complete_state_machine_preserve_all(chart)
    superstate = Stateflow.State(chart);
    superstate.Name = 'ACC_Logic';
    superstate.Position = [50 30 900 650];
    superstate.Decomposition = 'EXCLUSIVE_OR';

    % 保留原有during动作，但修复语法 - 使用函数调用代替内联if
    superstate.LabelString = strjoin({
        'ACC_Logic', ...
        'during:', ...
        '  pending_distance_adj = pending_adj;', ...
        '  process_reset_signal();', ...
        '  validate_parameters();', ...
        '  use_inputs_to_avoid_warnings();'
    }, '\n');

    fprintf('✅ 超状态创建完成（保留所有原功能）\n');

    % 创建四个主要状态 - 完全保持原有定义
    s0 = create_state_preserve(superstate, 'S0_IN_CONTROL', [100 100 250 180], {
        'current_state = 0;',
        'acc_active = true;',
        'control_enabled = true;',
        'is_in_control = true;'
    });

    s1 = create_state_preserve(superstate, 'S1_ADAPTIVE_HISTORY_STANDBY', [500 100 280 180], {
        'current_state = 1;',
        'acc_active = false;',
        'control_enabled = false;',
        'is_in_control = false;'
    });

    s2 = create_state_preserve(superstate, 'S2_ADAPTIVE_NO_HISTORY_STANDBY', [100 350 280 180], {
        'current_state = 2;',
        'acc_active = false;',
        'control_enabled = false;',
        'is_in_control = false;',
        'has_history = false;'
    });

    s3 = create_state_preserve(superstate, 'S3_LOW_SPEED', [500 350 250 180], {
        'current_state = 3;',
        'acc_active = false;',
        'control_enabled = false;',
        'is_in_control = false;'
    });

    % 创建基于车速的条件默认转移 - 保持原有逻辑
    create_conditional_default_transitions_preserve(superstate, s2, s3);

    % 创建内部转移 - 保持原有逻辑
    create_internal_transitions_preserve(superstate, s0, s1, s2, s3);

    % 创建图形函数 - 保持原有所有功能
    create_graphical_functions_preserve_all(chart);
end

function create_conditional_default_transitions_preserve(superstate, s2, s3)
    fprintf('创建默认转移到S3（保持原有逻辑）...\n');

    % 保持原有的默认转移逻辑，只修复语法
    default_trans = Stateflow.Transition(superstate);
    default_trans.Destination = s3;
    default_trans.LabelString = strjoin({
        '{', ...
        'current_state = 3;', ...
        'V3_kmh = 50.0;', ...
        'G1_m = 15.0;', ...
        'G2_s = 2.0;', ...
        'control_mode = 0;', ...
        'message_code = 0;', ...
        'has_history = false;', ...
        '}'
    }, '\n');
    default_trans.SourceEndPoint = [40, 440];
    default_trans.DestinationEndPoint = [500, 440];

    fprintf('✅ 默认转移到S3创建完成，保持原设计逻辑\n');
end

function state = create_state_preserve(parent, name, position, entry_actions)
    state = Stateflow.State(parent);
    state.Name = name;
    state.Position = position;

    entry_str = strjoin(entry_actions, '\n  ');
    state.LabelString = sprintf('%s\nentry:\n  %s', name, entry_str);
end

function create_transition_preserve(parent, source, dest, condition, action, varargin)
    trans = Stateflow.Transition(parent);
    trans.Source = source;
    trans.Destination = dest;

    if ~isempty(condition) && ~isempty(action)
        trans.LabelString = sprintf('[%s]\n{%s}', condition, action);
    elseif ~isempty(condition)
        trans.LabelString = sprintf('[%s]', condition);
    elseif ~isempty(action)
        trans.LabelString = sprintf('{%s}', action);
    end

    if nargin > 5 && ~isempty(varargin{1})
        if isnumeric(varargin{1}) && isscalar(varargin{1})
            trans.ExecutionOrder = varargin{1};
        end
    end
end

function create_internal_transitions_preserve(superstate, s0, s1, s2, s3)
    fprintf('创建状态转移（保持原有所有转移逻辑）...\n');

    exec_order = 10;  % 从10开始，避免与默认转移冲突

    % ===== S0 (在控) 状态下的转移 - 完全保持原有逻辑 =====
    create_transition_preserve(s0, s0, s0, ...
        'command_input == 0', ...
        'control_mode = 1; V3_kmh = max(V2_KMH+1, V3_kmh - SPEED_STEP); message_code = 101;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s0, s0, s0, ...
        'command_input == 1', ...
        'control_mode = 2; V3_kmh = min(120, V3_kmh + SPEED_STEP); message_code = 102;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s0, s0, s0, ...
        'command_input == 2', ...
        'control_mode = 3; G1_m = max(5, G1_m - DISTANCE_STEP); message_code = 103;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s0, s0, s0, ...
        'command_input == 3', ...
        'control_mode = 4; G1_m = min(50, G1_m + DISTANCE_STEP); message_code = 104;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s0, s0, s0, ...
        'command_input == 4', ...
        'control_mode = 7; message_code = 107;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s0, s0, s1, ...
        'command_input == 5', ...
        'control_mode = 8; save_history(); message_code = 108;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s0, s0, s1, ...
        'command_input == 6', ...
        'control_mode = 8; save_history(); message_code = 109;', ...
        exec_order);
    exec_order = exec_order + 1;

    % ===== S1 (有史待命) 状态下的转移 - 完全保持原有逻辑 =====
    create_transition_preserve(s1, s1, s0, ...
        'command_input == 1', ...
        'control_mode = 6; restore_history(); message_code = 206;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s1, s1, s0, ...
        'command_input == 0', ...
        'control_mode = 5; message_code = 205;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s1, s1, s1, ...
        'command_input >= 2 && command_input <= 6', ...
        'control_mode = 8; message_code = 200 + command_input;', ...
        exec_order);
    exec_order = exec_order + 1;

    % ===== S2 (无史待命) 状态下的转移 - 完全保持原有逻辑 =====
    create_transition_preserve(s2, s2, s0, ...
        'command_input == 0', ...
        'control_mode = 5; message_code = 305;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s2, s2, s2, ...
        'command_input >= 1 && command_input <= 6', ...
        'control_mode = 8; message_code = 300 + command_input;', ...
        exec_order);
    exec_order = exec_order + 1;

    % ===== S3 (低速) 状态下的转移 - 保持原有速度条件和设计初衷 =====
    create_transition_preserve(s3, s3, s3, ...
        'command_input >= 0 && command_input <= 6 && ego_speed_kmh < V1_KMH', ...
        'control_mode = 8; message_code = 400 + command_input;', ...
        exec_order);
    exec_order = exec_order + 1;

    % ===== 自动速度转移 - 完全保持原有逻辑 =====
    create_transition_preserve(s1, s1, s3, ...
        'ego_speed_kmh < V1_KMH', ...
        'message_code = 913;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s2, s2, s3, ...
        'ego_speed_kmh < V1_KMH && ego_speed_kmh > 0', ...
        'message_code = 923;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s3, s3, s1, ...
        'ego_speed_kmh >= V1_KMH && has_history == true', ...
        'message_code = 931;', ...
        exec_order);
    exec_order = exec_order + 1;

    create_transition_preserve(s3, s3, s2, ...
        'ego_speed_kmh >= V1_KMH && has_history == false', ...
        'message_code = 932;', ...
        exec_order);
    exec_order = exec_order + 1;

    fprintf('✅ 创建了 %d 个状态转移（保持所有原功能）\n', exec_order - 10);
end

function create_graphical_functions_preserve_all(chart)
    fprintf('创建图形函数（保持所有原功能）...\n');

    % 完全保持原有功能的函数定义
    functions_definitions(1).Name = 'save_history';
    functions_definitions(1).Position = [850 50 140 120];
    functions_definitions(1).Script = {
        'function save_history()', ...
        '  % Save current parameters to history', ...
        '  has_history = true;', ...
        '  history_V3_kmh = V3_kmh;', ...
        '  history_G1_m = G1_m;', ...
        '  history_G2_s = G2_s;'
    };

    functions_definitions(2).Name = 'restore_history';
    functions_definitions(2).Position = [850 200 140 150];
    functions_definitions(2).Script = {
        'function restore_history()', ...
        '  % Restore parameters from history', ...
        '  if (has_history == true)', ...
        '    V3_kmh = history_V3_kmh;', ...
        '    G1_m = history_G1_m;', ...
        '    G2_s = history_G2_s;', ...
        '    % Apply pending adjustments', ...
        '    if (pending_adj ~= 0)', ...
        '      G1_m = max(5.0, min(50.0, G1_m + pending_adj));', ...
        '      pending_adj = 0;', ...
        '    end', ...
        '  end'
    };

    functions_definitions(3).Name = 'reset_all_parameters';
    functions_definitions(3).Position = [850 380 140 200];
    functions_definitions(3).Script = {
        'function reset_all_parameters()', ...
        '  % Reset all parameters to default', ...
        '  has_history = false;', ...
        '  history_V3_kmh = 50.0;', ...
        '  history_G1_m = 15.0;', ...
        '  history_G2_s = 2.0;', ...
        '  pending_adj = 0;', ...
        '  V3_kmh = 50.0;', ...
        '  G1_m = 15.0;', ...
        '  G2_s = 2.0;', ...
        '  control_mode = 0;', ...
        '  message_code = 0;', ...
        '  % Always reset to S3, let automatic transitions handle speed-based state', ...
        '  current_state = 3;  % S3 低速状态，自动转移会根据车速调整'
    };

    % 添加新的辅助函数来处理during动作中的逻辑
    functions_definitions(4).Name = 'process_reset_signal';
    functions_definitions(4).Position = [1000 50 140 100];
    functions_definitions(4).Script = {
        'function process_reset_signal()', ...
        '  % Process reset signal with highest priority', ...
        '  if (reset_signal == true)', ...
        '    reset_all_parameters();', ...
        '  end'
    };

    functions_definitions(5).Name = 'validate_parameters';
    functions_definitions(5).Position = [1000 170 140 120];
    functions_definitions(5).Script = {
        'function validate_parameters()', ...
        '  % Validate parameters', ...
        '  if (V3_kmh <= 0)', ...
        '    V3_kmh = 50.0;', ...
        '  end', ...
        '  if (G1_m <= 0)', ...
        '    G1_m = 15.0;', ...
        '  end'
    };

    functions_definitions(6).Name = 'use_inputs_to_avoid_warnings';
    functions_definitions(6).Position = [1000 310 140 100];
    functions_definitions(6).Script = {
        'function use_inputs_to_avoid_warnings()', ...
        '  % Use inputs to avoid warnings', ...
        '  if (has_target || current_distance > 0)', ...
        '    % Dummy operation', ...
        '  end'
    };

    for i = 1:length(functions_definitions)
        func_def = functions_definitions(i);

        func_obj = Stateflow.EMFunction(chart);
        func_obj.Name = func_def.Name;
        func_obj.Position = func_def.Position;
        func_obj.Script = strjoin(func_def.Script, '\n');
    end

    fprintf('✅ 图形函数创建完成（保留所有原功能）\n');
end