%% 检查acc_decision_core中Lookup Table的配置
fprintf('======================================================================\n');
fprintf('检查Lookup Table配置\n');
fprintf('======================================================================\n\n');

model_name = 'acc_decision_core';

% 加载模型
fprintf('📂 加载模型: %s\n', model_name);
load_system(model_name);
fprintf('✅ 模型加载成功\n\n');

% 加载查找表数据
fprintf('📂 加载decision_lookup_data.mat...\n');
load('decision_lookup_data.mat');
fprintf('✅ 查找表数据已加载\n\n');

% 查找所有Lookup Table
lookup_blocks = find_system(model_name, 'BlockType', 'Lookup_n-D');

fprintf('🔍 检查 %d 个Lookup Table模块:\n', length(lookup_blocks));
fprintf('======================================================================\n\n');

for i = 1:length(lookup_blocks)
    block = lookup_blocks{i};
    block_name = get_param(block, 'Name');

    fprintf('【%d】 %s\n', i, block_name);
    fprintf('----------------------------------------------------------------------\n');

    try
        % 获取Table数据参数
        table_data_param = get_param(block, 'Table');
        fprintf('Table Data: %s\n', table_data_param);

        % 获取Breakpoint参数
        num_dims = str2double(get_param(block, 'NumberOfTableDimensions'));
        fprintf('维度数: %d\n', num_dims);

        if num_dims >= 1
            bp1 = get_param(block, 'BreakpointsForDimension1');
            fprintf('Breakpoint 1: %s\n', bp1);
        end

        if num_dims >= 2
            bp2 = get_param(block, 'BreakpointsForDimension2');
            fprintf('Breakpoint 2: %s\n', bp2);
        end

        % 检查是否正确引用了工作区变量
        if contains(block_name, 'next_state')
            expected_table = 'next_state_table';
            fprintf('\n期望引用: %s\n', expected_table);
            if strcmp(table_data_param, expected_table)
                fprintf('✅ 正确引用\n');
            else
                fprintf('❌ 错误！应该引用 %s\n', expected_table);
            end

        elseif contains(block_name, 'decision')
            expected_table = 'decision_table';
            fprintf('\n期望引用: %s\n', expected_table);
            if strcmp(table_data_param, expected_table)
                fprintf('✅ 正确引用\n');
            else
                fprintf('❌ 错误！应该引用 %s\n', expected_table);
            end

        elseif contains(block_name, 'control_enabled')
            expected_table = 'control_enabled_table';
            fprintf('\n期望引用: %s\n', expected_table);
            if strcmp(table_data_param, expected_table)
                fprintf('✅ 正确引用\n');
            else
                fprintf('❌ 错误！应该引用 %s\n', expected_table);
            end

        elseif contains(block_name, 'side_effect')
            expected_table = 'side_effect_table';
            fprintf('\n期望引用: %s\n', expected_table);
            if strcmp(table_data_param, expected_table)
                fprintf('✅ 正确引用\n');
            else
                fprintf('❌ 错误！应该引用 %s\n', expected_table);
            end
        end

    catch ME
        fprintf('❌ 获取参数失败: %s\n', ME.message);
    end

    fprintf('\n');
end

%% 检查模型工作区
fprintf('======================================================================\n');
fprintf('检查模型工作区变量\n');
fprintf('======================================================================\n\n');

mdlWks = get_param(model_name, 'ModelWorkspace');

vars = {'state_bp', 'command_bp', 'next_state_table', 'decision_table', 'control_enabled_table'};
all_exist = true;

for i = 1:length(vars)
    var_name = vars{i};
    if mdlWks.hasVariable(var_name)
        fprintf('✅ %s 存在\n', var_name);

        % 验证数据
        var_value = mdlWks.getVariable(var_name);
        if strcmp(var_name, 'next_state_table')
            % 检查S2+Cmd1的值
            if var_value(3, 2) == 0
                fprintf('   ✅ S2+Cmd1 → next_state=0 (正确)\n');
            else
                fprintf('   ❌ S2+Cmd1 → next_state=%d (错误)\n', var_value(3, 2));
            end
        elseif strcmp(var_name, 'decision_table')
            if var_value(3, 2) == 5
                fprintf('   ✅ S2+Cmd1 → decision=R5 (正确)\n');
            else
                fprintf('   ❌ S2+Cmd1 → decision=R%d (错误)\n', var_value(3, 2));
            end
        elseif strcmp(var_name, 'control_enabled_table')
            if var_value(3, 2) == 1
                fprintf('   ✅ S2+Cmd1 → enabled=1 (正确)\n');
            else
                fprintf('   ❌ S2+Cmd1 → enabled=%d (错误)\n', var_value(3, 2));
            end
        end
    else
        fprintf('❌ %s 不存在\n', var_name);
        all_exist = false;
    end
end

fprintf('\n');

if ~all_exist
    fprintf('⚠️  警告：模型工作区缺少必需的变量！\n');
    fprintf('   需要重新加载decision_lookup_data.mat到模型工作区\n\n');
end

%% 测试Lookup Table（如果数据存在）
if all_exist
    fprintf('======================================================================\n');
    fprintf('测试Lookup Table查找\n');
    fprintf('======================================================================\n\n');

    % 测试S2 + Cmd1
    state = 2;
    cmd = 1;

    next_state_val = mdlWks.getVariable('next_state_table');
    decision_val = mdlWks.getVariable('decision_table');
    enabled_val = mdlWks.getVariable('control_enabled_table');

    fprintf('测试输入: state=%d, cmd=%d\n', state, cmd);
    fprintf('期望输出: next_state=0, decision=5, enabled=1\n');
    fprintf('实际查表: next_state=%d, decision=%d, enabled=%d\n\n', ...
        next_state_val(state+1, cmd+1), ...
        decision_val(state+1, cmd+1), ...
        enabled_val(state+1, cmd+1));
end

close_system(model_name, 0);
fprintf('✅ 检查完成\n');
