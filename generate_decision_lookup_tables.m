function generate_decision_lookup_tables()
%GENERATE_DECISION_LOOKUP_TABLES 从state_transitions.json生成Simulink查找表数据
%   自动读取JSON并生成用于Simulink 2D Lookup Table的数据
%
%   输出文件:
%   - decision_lookup_data.mat (包含所有查找表数据)
%   - decision_lookup_tables.txt (可读格式，便于检查)

fprintf('🔄 开始生成ACC决策查找表...\n\n');

%% 1. 读取state_transitions.json
json_file = 'state_transitions.json';
if ~exist(json_file, 'file')
    error('❌ 未找到 state_transitions.json 文件');
end

fprintf('📂 读取 %s...\n', json_file);
json_text = fileread(json_file);
data = jsondecode(json_text);

fprintf('✅ JSON读取成功\n\n');

%% 2. 定义查找表的维度
% 状态: 0-3 (S0, S1, S2, S3)
% 指令: 0-7 (NONE, I0-I6)
states = 0:3;
commands = 0:7;

num_states = length(states);
num_commands = length(commands);

fprintf('📊 查找表维度: %d states × %d commands\n', num_states, num_commands);
fprintf('   States: %s\n', mat2str(states));
fprintf('   Commands: %s\n\n', mat2str(commands));

%% 3. 初始化查找表 (默认值: S2待命状态)
% 5个输出表
next_state_table = zeros(num_states, num_commands);
decision_table = 8 * ones(num_states, num_commands);  % 默认R8待命
control_enabled_table = zeros(num_states, num_commands);
side_effect_table = zeros(num_states, num_commands);  % 0=无, 1=set_has_history_true
decision_raw_table = 8 * ones(num_states, num_commands);  % 原始decision值(可能=-1)

%% 4. 填充查找表数据
fprintf('🔧 填充查找表数据...\n');

transitions = data.transitions;
state_keys = fieldnames(transitions);

for i = 1:length(state_keys)
    state_key = state_keys{i};

    % MATLAB jsondecode将数字键转换为'x0', 'x1'格式，需要去掉'x'前缀
    if startsWith(state_key, 'x')
        state_num = str2double(state_key(2:end));
    else
        state_num = str2double(state_key);
    end

    % 验证state_num有效
    if isnan(state_num) || state_num < 0 || state_num > 3
        continue;
    end

    state_transitions = transitions.(state_key);

    % 先处理default规则（如果存在）
    default_rule = [];
    if isfield(state_transitions, 'default')
        default_rule = state_transitions.default;
    end

    % 处理每个command
    for cmd = 0:7
        % MATLAB将JSON数字键转换为'x0', 'x1'等格式
        cmd_key = sprintf('x%d', cmd);

        % 检查是否有该command的转移规则
        if isfield(state_transitions, cmd_key)
            rule = state_transitions.(cmd_key);
        elseif ~isempty(default_rule)
            rule = default_rule;
        else
            % 没有规则，使用安全默认值
            rule = struct('next_state', 2, 'decision', 8, 'control_enabled', false);
        end

        % 提取转移信息
        next_state = rule.next_state;
        decision_raw = rule.decision;
        control_enabled = double(rule.control_enabled);

        % 检查side_effect
        side_effect = 0;
        if isfield(rule, 'side_effect')
            if strcmp(rule.side_effect, 'set_has_history_true')
                side_effect = 1;
            end
        end

        % 填充表格 (MATLAB索引从1开始)
        row_idx = state_num + 1;
        col_idx = cmd + 1;

        % 验证索引有效
        if row_idx >= 1 && row_idx <= num_states && col_idx >= 1 && col_idx <= num_commands
            next_state_table(row_idx, col_idx) = next_state;
            decision_raw_table(row_idx, col_idx) = decision_raw;
            control_enabled_table(row_idx, col_idx) = control_enabled;
            side_effect_table(row_idx, col_idx) = side_effect;

            % decision处理: -1表示使用last_active_decision，这里先记录-1
            % 实际使用时需要在Simulink中用Switch模块处理
            if decision_raw == -1
                decision_table(row_idx, col_idx) = -1;
            else
                decision_table(row_idx, col_idx) = decision_raw;
            end
        end

        if mod(i * num_commands + cmd, 8) == 0
            fprintf('.');
        end
    end
end

fprintf('\n✅ 查找表填充完成\n\n');

%% 5. 保存为.mat文件供Simulink使用
save_file = 'decision_lookup_data.mat';

% Breakpoint向量
state_bp = states;      % [0, 1, 2, 3]
command_bp = commands;  % [0, 1, 2, 3, 4, 5, 6, 7]

save(save_file, 'state_bp', 'command_bp', ...
     'next_state_table', 'decision_table', 'decision_raw_table', ...
     'control_enabled_table', 'side_effect_table');

fprintf('💾 查找表数据已保存到: %s\n', save_file);
fprintf('   包含变量:\n');
fprintf('   - state_bp: [%s]\n', mat2str(state_bp));
fprintf('   - command_bp: [%s]\n', mat2str(command_bp));
fprintf('   - next_state_table: %dx%d\n', size(next_state_table));
fprintf('   - decision_table: %dx%d\n', size(decision_table));
fprintf('   - control_enabled_table: %dx%d\n', size(control_enabled_table));
fprintf('   - side_effect_table: %dx%d\n\n', size(side_effect_table));

%% 6. 生成可读文本文件
txt_file = 'decision_lookup_tables.txt';
fid = fopen(txt_file, 'w');

fprintf(fid, '====================================================================\n');
fprintf(fid, 'ACC决策查找表 - 可读格式\n');
fprintf(fid, '====================================================================\n\n');

fprintf(fid, '查找表维度:\n');
fprintf(fid, '  States (行): %s  (S0=在控, S1=有史待命, S2=无史待命, S3=低速)\n', mat2str(state_bp));
fprintf(fid, '  Commands (列): %s\n', mat2str(command_bp));
fprintf(fid, '  (0=NONE, 1=E键, 2=Q键, 3=T键, 4=R键, 5=W键, 6=S键, 7=C键)\n\n');

% 打印next_state_table
fprintf(fid, '--------------------------------------------------------------------\n');
fprintf(fid, '表1: next_state_table (下一状态)\n');
fprintf(fid, '--------------------------------------------------------------------\n');
fprintf(fid, '       ');
for cmd = commands
    fprintf(fid, ' Cmd%d', cmd);
end
fprintf(fid, '\n');

for s = 1:num_states
    fprintf(fid, 'State%d:', states(s));
    for c = 1:num_commands
        fprintf(fid, '    %d', next_state_table(s, c));
    end
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

% 打印decision_table
fprintf(fid, '--------------------------------------------------------------------\n');
fprintf(fid, '表2: decision_table (决策输出, -1=使用last_active_decision)\n');
fprintf(fid, '--------------------------------------------------------------------\n');
fprintf(fid, '       ');
for cmd = commands
    fprintf(fid, ' Cmd%d', cmd);
end
fprintf(fid, '\n');

for s = 1:num_states
    fprintf(fid, 'State%d:', states(s));
    for c = 1:num_commands
        val = decision_table(s, c);
        if val == -1
            fprintf(fid, '   -1');
        else
            fprintf(fid, '   R%d', val);
        end
    end
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

% 打印control_enabled_table
fprintf(fid, '--------------------------------------------------------------------\n');
fprintf(fid, '表3: control_enabled_table (控制使能, 0=禁用, 1=启用)\n');
fprintf(fid, '--------------------------------------------------------------------\n');
fprintf(fid, '       ');
for cmd = commands
    fprintf(fid, ' Cmd%d', cmd);
end
fprintf(fid, '\n');

for s = 1:num_states
    fprintf(fid, 'State%d:', states(s));
    for c = 1:num_commands
        fprintf(fid, '    %d', control_enabled_table(s, c));
    end
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

% 打印side_effect_table
fprintf(fid, '--------------------------------------------------------------------\n');
fprintf(fid, '表4: side_effect_table (副作用, 1=set_has_history_true)\n');
fprintf(fid, '--------------------------------------------------------------------\n');
fprintf(fid, '       ');
for cmd = commands
    fprintf(fid, ' Cmd%d', cmd);
end
fprintf(fid, '\n');

for s = 1:num_states
    fprintf(fid, 'State%d:', states(s));
    for c = 1:num_commands
        fprintf(fid, '    %d', side_effect_table(s, c));
    end
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

% 打印详细转移规则
fprintf(fid, '====================================================================\n');
fprintf(fid, '详细转移规则 (便于验证)\n');
fprintf(fid, '====================================================================\n\n');

for s = 1:num_states
    state_num = states(s);
    fprintf(fid, '--- State %d ---\n', state_num);
    for c = 1:num_commands
        cmd_num = commands(c);
        fprintf(fid, '  S%d + Cmd%d → S%d, R%d, enable=%d', ...
            state_num, cmd_num, ...
            next_state_table(s, c), ...
            decision_table(s, c), ...
            control_enabled_table(s, c));
        if side_effect_table(s, c) == 1
            fprintf(fid, ' [set_history]');
        end
        fprintf(fid, '\n');
    end
    fprintf(fid, '\n');
end

fclose(fid);

fprintf('📄 可读文本已保存到: %s\n\n', txt_file);

%% 7. 验证关键转移
fprintf('✅ 验证关键转移规则:\n');
fprintf('   S0 + Cmd6(S键) → S%d, R%d, enable=%d (期望: S1, R8, 0)\n', ...
    next_state_table(1, 7), decision_table(1, 7), control_enabled_table(1, 7));
fprintf('   S1 + Cmd1(E键) → S%d, R%d, enable=%d (期望: S0, R5, 1)\n', ...
    next_state_table(2, 2), decision_table(2, 2), control_enabled_table(2, 2));
fprintf('   S1 + Cmd2(Q键) → S%d, R%d, enable=%d (期望: S0, R6, 1)\n', ...
    next_state_table(2, 3), decision_table(2, 3), control_enabled_table(2, 3));
fprintf('   S2 + Cmd1(E键) → S%d, R%d, enable=%d, side_effect=%d (期望: S0, R5, 1, 1)\n', ...
    next_state_table(3, 2), decision_table(3, 2), control_enabled_table(3, 2), side_effect_table(3, 2));

fprintf('\n🎉 查找表生成完成！\n');

end
