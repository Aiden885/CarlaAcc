%% 验证Simulink决策模型是否正确加载查找表数据
% 检查acc_decision_core.slx中的Lookup Table是否使用了正确的数据

fprintf('======================================================================\n');
fprintf('验证Simulink决策模型查找表配置\n');
fprintf('======================================================================\n\n');

model_name = 'acc_decision_core';

%% 1. 检查模型是否存在
if ~exist([model_name '.slx'], 'file')
    error('❌ 模型文件 %s.slx 不存在', model_name);
end

fprintf('✅ 模型文件存在: %s.slx\n\n', model_name);

%% 2. 检查decision_lookup_data.mat是否存在并加载
if ~exist('decision_lookup_data.mat', 'file')
    error('❌ decision_lookup_data.mat 不存在，请先运行 generate_decision_lookup_tables()');
end

fprintf('📂 加载 decision_lookup_data.mat...\n');
mat_data = load('decision_lookup_data.mat');

fprintf('✅ 查找表数据已加载:\n');
fprintf('   state_bp: %s\n', mat2str(mat_data.state_bp));
fprintf('   command_bp: %s\n', mat2str(mat_data.command_bp));
fprintf('   next_state_table: %dx%d\n', size(mat_data.next_state_table));
fprintf('   decision_table: %dx%d\n', size(mat_data.decision_table));
fprintf('   control_enabled_table: %dx%d\n\n', size(mat_data.control_enabled_table));

%% 3. 验证关键转移规则（从.mat文件）
fprintf('🔍 验证关键转移规则（从decision_lookup_data.mat）:\n');
fprintf('----------------------------------------------------------------------\n');

% MATLAB索引从1开始，所以state=2对应索引3，cmd=1对应索引2
s2_cmd1_next_state = mat_data.next_state_table(3, 2);
s2_cmd1_decision = mat_data.decision_table(3, 2);
s2_cmd1_enabled = mat_data.control_enabled_table(3, 2);

fprintf('S2 + Cmd1(E键) → next_state=%d, decision=R%d, enabled=%d\n', ...
    s2_cmd1_next_state, s2_cmd1_decision, s2_cmd1_enabled);

if s2_cmd1_next_state == 0 && s2_cmd1_decision == 5 && s2_cmd1_enabled == 1
    fprintf('   ✅ 正确！S2+E键应该启动控制\n\n');
else
    fprintf('   ❌ 错误！期望: next_state=0, decision=R5, enabled=1\n\n');
end

% 验证S1 + Cmd1
s1_cmd1_next_state = mat_data.next_state_table(2, 2);
s1_cmd1_decision = mat_data.decision_table(2, 2);
s1_cmd1_enabled = mat_data.control_enabled_table(2, 2);

fprintf('S1 + Cmd1(E键) → next_state=%d, decision=R%d, enabled=%d\n', ...
    s1_cmd1_next_state, s1_cmd1_decision, s1_cmd1_enabled);

if s1_cmd1_next_state == 0 && s1_cmd1_decision == 5 && s1_cmd1_enabled == 1
    fprintf('   ✅ 正确！S1+E键应该启动控制\n\n');
else
    fprintf('   ❌ 错误！期望: next_state=0, decision=R5, enabled=1\n\n');
end

%% 4. 加载Simulink模型并检查工作区变量
fprintf('======================================================================\n');
fprintf('检查Simulink模型配置\n');
fprintf('======================================================================\n\n');

fprintf('📂 加载模型: %s\n', model_name);
try
    load_system(model_name);
    fprintf('✅ 模型加载成功\n\n');
catch ME
    error('❌ 模型加载失败: %s', ME.message);
end

%% 5. 检查模型工作区中的变量
fprintf('🔍 检查模型工作区中的查找表变量:\n');
fprintf('----------------------------------------------------------------------\n');

try
    % 获取模型工作区
    mdlWks = get_param(model_name, 'ModelWorkspace');

    % 检查必需的变量是否存在
    required_vars = {'state_bp', 'command_bp', 'next_state_table', ...
                     'decision_table', 'control_enabled_table'};

    all_exist = true;
    for i = 1:length(required_vars)
        var_name = required_vars{i};
        if mdlWks.hasVariable(var_name)
            fprintf('   ✅ %s 存在\n', var_name);

            % 如果是查找表，验证关键值
            if strcmp(var_name, 'next_state_table')
                model_table = mdlWks.getVariable(var_name);
                model_s2_cmd1 = model_table(3, 2);
                fprintf('      模型中 S2+Cmd1 → next_state=%d ', model_s2_cmd1);
                if model_s2_cmd1 == s2_cmd1_next_state
                    fprintf('✅ 与.mat一致\n');
                else
                    fprintf('❌ 与.mat不一致（.mat中是%d）\n', s2_cmd1_next_state);
                end
            end
        else
            fprintf('   ❌ %s 不存在\n', var_name);
            all_exist = false;
        end
    end
    fprintf('\n');

    if ~all_exist
        fprintf('⚠️  警告：模型工作区缺少必需的变量！\n');
        fprintf('   解决方案：在模型初始化回调中加载decision_lookup_data.mat\n\n');
    end

catch ME
    fprintf('⚠️  无法访问模型工作区: %s\n\n', ME.message);
end

%% 6. 检查模型回调函数
fprintf('🔍 检查模型初始化回调（PreLoadFcn/InitFcn）:\n');
fprintf('----------------------------------------------------------------------\n');

try
    preload_fcn = get_param(model_name, 'PreLoadFcn');
    if ~isempty(preload_fcn)
        fprintf('PreLoadFcn:\n%s\n\n', preload_fcn);
    else
        fprintf('PreLoadFcn: (空)\n\n');
    end

    init_fcn = get_param(model_name, 'InitFcn');
    if ~isempty(init_fcn)
        fprintf('InitFcn:\n%s\n\n', init_fcn);
    else
        fprintf('InitFcn: (空)\n\n');
    end

    % 检查是否加载了decision_lookup_data.mat
    if contains(preload_fcn, 'decision_lookup_data.mat') || ...
       contains(init_fcn, 'decision_lookup_data.mat')
        fprintf('   ✅ 模型回调中包含加载decision_lookup_data.mat\n\n');
    else
        fprintf('   ⚠️  模型回调中未找到加载decision_lookup_data.mat的代码\n');
        fprintf('   建议：在InitFcn中添加 load(''decision_lookup_data.mat'')\n\n');
    end

catch ME
    fprintf('⚠️  无法检查回调函数: %s\n\n', ME.message);
end

%% 7. 总结
fprintf('======================================================================\n');
fprintf('诊断总结\n');
fprintf('======================================================================\n\n');

fprintf('问题分析：\n');
fprintf('  1. decision_lookup_data.mat中的数据是正确的\n');
fprintf('  2. 但UDP测试显示Simulink返回了错误的值\n');
fprintf('  3. 这说明Simulink模型没有使用正确的查找表数据\n\n');

fprintf('可能的原因：\n');
fprintf('  A. 模型工作区中的变量与.mat文件不一致（未重新加载）\n');
fprintf('  B. 模型使用了硬编码的查找表数据而不是从.mat加载\n');
fprintf('  C. 模型需要重新编译（Accelerator模式缓存了旧数据）\n\n');

fprintf('解决步骤：\n');
fprintf('  1. 确保模型InitFcn中有: load(''decision_lookup_data.mat'')\n');
fprintf('  2. 在MATLAB中运行: load(''decision_lookup_data.mat'')\n');
fprintf('  3. 关闭并重新打开模型\n');
fprintf('  4. 重新编译模型（如果使用Accelerator模式）\n');
fprintf('  5. 重新启动Simulink UDP服务器\n\n');

close_system(model_name, 0);
fprintf('✅ 验证完成\n');