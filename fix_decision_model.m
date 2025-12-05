%% 修复acc_decision_core模型的查找表配置
% 将decision_lookup_data.mat加载到模型工作区并设置InitFcn

fprintf('======================================================================\n');
fprintf('修复ACC决策模型配置\n');
fprintf('======================================================================\n\n');

model_name = 'acc_decision_core';

%% 1. 确保查找表数据存在
if ~exist('decision_lookup_data.mat', 'file')
    error('❌ decision_lookup_data.mat不存在，请先运行 generate_decision_lookup_tables()');
end

fprintf('📂 加载 decision_lookup_data.mat...\n');
mat_data = load('decision_lookup_data.mat');
fprintf('✅ 查找表数据已加载\n\n');

%% 2. 加载模型
fprintf('📂 加载模型: %s\n', model_name);
load_system(model_name);
fprintf('✅ 模型加载成功\n\n');

%% 3. 将数据加载到模型工作区
fprintf('🔧 将查找表数据写入模型工作区...\n');
try
    mdlWks = get_param(model_name, 'ModelWorkspace');

    % 将所有变量写入模型工作区
    vars_to_save = {'state_bp', 'command_bp', 'next_state_table', ...
                    'decision_table', 'decision_raw_table', ...
                    'control_enabled_table', 'side_effect_table'};

    for i = 1:length(vars_to_save)
        var_name = vars_to_save{i};
        if isfield(mat_data, var_name)
            mdlWks.assignin(var_name, mat_data.(var_name));
            fprintf('   ✅ %s 已写入模型工作区\n', var_name);
        end
    end
    fprintf('\n');

catch ME
    error('❌ 写入模型工作区失败: %s', ME.message);
end

%% 4. 设置InitFcn回调
fprintf('🔧 设置模型InitFcn回调...\n');
init_fcn = sprintf('load(''%s'');', 'decision_lookup_data.mat');
set_param(model_name, 'InitFcn', init_fcn);
fprintf('   InitFcn: %s\n', init_fcn);
fprintf('✅ InitFcn已设置\n\n');

%% 5. 验证写入的数据
fprintf('🔍 验证模型工作区中的数据:\n');
fprintf('----------------------------------------------------------------------\n');

next_state_table_model = mdlWks.getVariable('next_state_table');
decision_table_model = mdlWks.getVariable('decision_table');
control_enabled_table_model = mdlWks.getVariable('control_enabled_table');

% 检查S2 + Cmd1
s2_cmd1_next_state = next_state_table_model(3, 2);
s2_cmd1_decision = decision_table_model(3, 2);
s2_cmd1_enabled = control_enabled_table_model(3, 2);

fprintf('S2 + Cmd1(E键) → next_state=%d, decision=R%d, enabled=%d\n', ...
    s2_cmd1_next_state, s2_cmd1_decision, s2_cmd1_enabled);

if s2_cmd1_next_state == 0 && s2_cmd1_decision == 5 && s2_cmd1_enabled == 1
    fprintf('   ✅ 正确！\n\n');
else
    fprintf('   ❌ 数据写入有误！\n\n');
end

%% 6. 保存模型
fprintf('💾 保存模型...\n');
save_system(model_name);
fprintf('✅ 模型已保存\n\n');

%% 7. 清除Accelerator缓存
fprintf('🗑️  清除Accelerator缓存...\n');
slprj_path = './slprj/accel/acc_decision_core';
if exist(slprj_path, 'dir')
    try
        rmdir(slprj_path, 's');
        fprintf('✅ 已清除缓存: %s\n\n', slprj_path);
    catch ME
        fprintf('⚠️  清除缓存失败: %s\n\n', ME.message);
    end
else
    fprintf('   缓存目录不存在，跳过\n\n');
end

%% 8. 关闭模型
close_system(model_name);
fprintf('✅ 模型已关闭\n\n');

%% 9. 总结
fprintf('======================================================================\n');
fprintf('修复完成！\n');
fprintf('======================================================================\n\n');

fprintf('已完成的操作：\n');
fprintf('  1. ✅ 将decision_lookup_data.mat加载到模型工作区\n');
fprintf('  2. ✅ 设置InitFcn自动加载查找表数据\n');
fprintf('  3. ✅ 保存模型\n');
fprintf('  4. ✅ 清除Accelerator缓存\n\n');

fprintf('下一步操作：\n');
fprintf('  1. 重新启动Simulink UDP服务器: run start_simulink_servers.m\n');
fprintf('  2. 测试E键功能: python debug_acc_state.py\n\n');