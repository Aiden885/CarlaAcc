function build_acc_decision_model_v2()
%BUILD_ACC_DECISION_MODEL_V2 创建ACC决策模型 - 使用n-D Lookup Table
%   使用更兼容的n-D Lookup Table模块替代2-D Lookup Table

fprintf('🔨 开始构建 ACC 决策 Simulink 模型 (v2 - 使用n-D Lookup Table)...\n\n');

%% 1. 检查数据文件
if ~exist('decision_lookup_data.mat', 'file')
    error('❌ 未找到 decision_lookup_data.mat，请先运行 generate_decision_lookup_tables.m');
end

load('decision_lookup_data.mat');
fprintf('✅ 查找表数据已加载\n');

%% 2. 创建新模型
model_name = 'acc_decision_core';

if bdIsLoaded(model_name)
    close_system(model_name, 0);
end
if exist([model_name '.slx'], 'file')
    fprintf('⚠️  模型已存在，将被覆盖\n');
    delete([model_name '.slx']);
end

new_system(model_name);
open_system(model_name);
fprintf('✅ 新模型已创建: %s\n', model_name);

%% 3. 配置模型参数
set_param(model_name, 'SolverType', 'Fixed-step');
set_param(model_name, 'Solver', 'FixedStepDiscrete');
set_param(model_name, 'FixedStep', '0.05');
set_param(model_name, 'StopTime', '0.05');
fprintf('✅ 模型参数已配置\n');

%% 4. 添加输入端口
fprintf('\n📥 添加输入端口...\n');
input_names = {'current_state', 'command_type', 'has_history', 'last_active_decision'};
input_pos = [50 50; 50 150; 50 250; 50 350];

for i = 1:length(input_names)
    add_block('simulink/Sources/In1', [model_name '/' input_names{i}], ...
        'Position', [input_pos(i,1), input_pos(i,2), input_pos(i,1)+30, input_pos(i,2)+14], ...
        'Port', num2str(i));
    fprintf('  ✓ %s (In%d)\n', input_names{i}, i);
end

%% 5. 添加n-D Lookup Table模块（更兼容）
fprintf('\n📊 添加 n-D Lookup Table 模块...\n');

% 方法：使用n-D Lookup Table，配置为2维
lut_names = {'LUT_next_state', 'LUT_decision', 'LUT_control_enabled', 'LUT_side_effect'};
lut_tables = {'next_state_table', 'decision_table', 'control_enabled_table', 'side_effect_table'};
lut_pos_y = [40, 140, 240, 340];

for i = 1:length(lut_names)
    blk_path = [model_name '/' lut_names{i}];

    % 添加n-D Lookup Table
    add_block('simulink/Lookup Tables/n-D Lookup Table', blk_path, ...
        'Position', [200, lut_pos_y(i), 280, lut_pos_y(i)+40]);

    % 配置为2维，使用显式断点
    set_param(blk_path, 'NumberOfTableDimensions', '2');
    set_param(blk_path, 'BreakpointsSpecification', 'Explicit values');

    % 设置表格数据和断点
    try
        % 方法1: 直接设置
        set_param(blk_path, 'Table', lut_tables{i});
        set_param(blk_path, 'BreakpointsForDimension1', 'state_bp');
        set_param(blk_path, 'BreakpointsForDimension2', 'command_bp');
        fprintf('  ✓ %s (方法1: 变量名)\n', lut_names{i});
    catch
        % 方法2: 使用数值
        try
            table_data = eval(lut_tables{i});
            set_param(blk_path, 'Table', mat2str(table_data));
            set_param(blk_path, 'BreakpointsForDimension1', mat2str(state_bp));
            set_param(blk_path, 'BreakpointsForDimension2', mat2str(command_bp));
            fprintf('  ✓ %s (方法2: 数值字符串)\n', lut_names{i});
        catch ME
            fprintf('  ⚠️  %s 配置失败: %s\n', lut_names{i}, ME.message);
        end
    end
end

%% 6. 处理 decision=-1 逻辑
fprintf('\n🔧 添加 decision=-1 处理逻辑...\n');

add_block('simulink/Sources/Constant', [model_name '/Const_Minus1'], ...
    'Position', [320, 100, 350, 120], 'Value', '-1');

add_block('simulink/Logic and Bit Operations/Relational Operator', ...
    [model_name '/Check_Decision_Minus1'], ...
    'Position', [380, 140, 410, 170], 'Operator', '==');

add_block('simulink/Signal Routing/Switch', [model_name '/Switch_Decision'], ...
    'Position', [450, 135, 480, 175], 'Threshold', '0.5');

fprintf('  ✓ decision=-1 处理已添加\n');

%% 7. 处理 has_history 副作用
fprintf('\n🔧 添加 has_history 副作用逻辑...\n');

add_block('simulink/Sources/Constant', [model_name '/Const_One'], ...
    'Position', [320, 300, 350, 320], 'Value', '1');

add_block('simulink/Logic and Bit Operations/Relational Operator', ...
    [model_name '/Check_SideEffect'], ...
    'Position', [320, 340, 350, 370], 'Operator', '==');

add_block('simulink/Signal Routing/Switch', [model_name '/Switch_HasHistory'], ...
    'Position', [400, 285, 430, 325], 'Threshold', '0.5');

fprintf('  ✓ has_history 副作用已添加\n');

%% 8. 处理 next_last_decision 更新
fprintf('\n🔧 添加 next_last_decision 更新逻辑...\n');

add_block('simulink/Logic and Bit Operations/Relational Operator', ...
    [model_name '/Check_Decision_GE1'], ...
    'Position', [520, 140, 550, 170], 'Operator', '>=');
add_block('simulink/Sources/Constant', [model_name '/Const_One_Decision'], ...
    'Position', [490, 175, 510, 195], 'Value', '1');

add_block('simulink/Logic and Bit Operations/Relational Operator', ...
    [model_name '/Check_Decision_LE6'], ...
    'Position', [520, 200, 550, 230], 'Operator', '<=');
add_block('simulink/Sources/Constant', [model_name '/Const_Six'], ...
    'Position', [490, 235, 510, 255], 'Value', '6');

add_block('simulink/Logic and Bit Operations/Logical Operator', ...
    [model_name '/AND_Decision_Range'], ...
    'Position', [580, 170, 610, 200], 'Operator', 'AND', 'Inputs', '2');

add_block('simulink/Signal Routing/Switch', [model_name '/Switch_LastDecision'], ...
    'Position', [650, 165, 680, 205], 'Threshold', '0.5');

fprintf('  ✓ next_last_decision 更新已添加\n');

%% 9. 添加输出端口
fprintf('\n📤 添加输出端口...\n');

output_names = {'next_state', 'decision', 'control_enabled', ...
                'next_has_history', 'next_last_decision'};
output_pos = [750 50; 750 150; 750 250; 750 350; 750 450];

for i = 1:length(output_names)
    add_block('simulink/Sinks/Out1', [model_name '/' output_names{i}], ...
        'Position', [output_pos(i,1), output_pos(i,2), ...
                     output_pos(i,1)+30, output_pos(i,2)+14], ...
        'Port', num2str(i));
    fprintf('  ✓ %s (Out%d)\n', output_names{i}, i);
end

%% 10. 连接模块
fprintf('\n🔗 连接模块...\n');

try
    % 输入 → 查找表
    add_line(model_name, 'current_state/1', 'LUT_next_state/1', 'autorouting', 'on');
    add_line(model_name, 'command_type/1', 'LUT_next_state/2', 'autorouting', 'on');

    add_line(model_name, 'current_state/1', 'LUT_decision/1', 'autorouting', 'on');
    add_line(model_name, 'command_type/1', 'LUT_decision/2', 'autorouting', 'on');

    add_line(model_name, 'current_state/1', 'LUT_control_enabled/1', 'autorouting', 'on');
    add_line(model_name, 'command_type/1', 'LUT_control_enabled/2', 'autorouting', 'on');

    add_line(model_name, 'current_state/1', 'LUT_side_effect/1', 'autorouting', 'on');
    add_line(model_name, 'command_type/1', 'LUT_side_effect/2', 'autorouting', 'on');

    % decision=-1 处理
    add_line(model_name, 'LUT_decision/1', 'Check_Decision_Minus1/1', 'autorouting', 'on');
    add_line(model_name, 'Const_Minus1/1', 'Check_Decision_Minus1/2', 'autorouting', 'on');
    add_line(model_name, 'Check_Decision_Minus1/1', 'Switch_Decision/2', 'autorouting', 'on');
    add_line(model_name, 'last_active_decision/1', 'Switch_Decision/1', 'autorouting', 'on');
    add_line(model_name, 'LUT_decision/1', 'Switch_Decision/3', 'autorouting', 'on');

    % has_history 副作用
    add_line(model_name, 'LUT_side_effect/1', 'Check_SideEffect/1', 'autorouting', 'on');
    add_line(model_name, 'Const_One/1', 'Check_SideEffect/2', 'autorouting', 'on');
    add_line(model_name, 'Check_SideEffect/1', 'Switch_HasHistory/2', 'autorouting', 'on');
    add_line(model_name, 'Const_One/1', 'Switch_HasHistory/1', 'autorouting', 'on');
    add_line(model_name, 'has_history/1', 'Switch_HasHistory/3', 'autorouting', 'on');

    % next_last_decision 更新
    add_line(model_name, 'Switch_Decision/1', 'Check_Decision_GE1/1', 'autorouting', 'on');
    add_line(model_name, 'Const_One_Decision/1', 'Check_Decision_GE1/2', 'autorouting', 'on');
    add_line(model_name, 'Switch_Decision/1', 'Check_Decision_LE6/1', 'autorouting', 'on');
    add_line(model_name, 'Const_Six/1', 'Check_Decision_LE6/2', 'autorouting', 'on');
    add_line(model_name, 'Check_Decision_GE1/1', 'AND_Decision_Range/1', 'autorouting', 'on');
    add_line(model_name, 'Check_Decision_LE6/1', 'AND_Decision_Range/2', 'autorouting', 'on');
    add_line(model_name, 'AND_Decision_Range/1', 'Switch_LastDecision/2', 'autorouting', 'on');
    add_line(model_name, 'Switch_Decision/1', 'Switch_LastDecision/1', 'autorouting', 'on');
    add_line(model_name, 'last_active_decision/1', 'Switch_LastDecision/3', 'autorouting', 'on');

    % 输出
    add_line(model_name, 'LUT_next_state/1', 'next_state/1', 'autorouting', 'on');
    add_line(model_name, 'Switch_Decision/1', 'decision/1', 'autorouting', 'on');
    add_line(model_name, 'LUT_control_enabled/1', 'control_enabled/1', 'autorouting', 'on');
    add_line(model_name, 'Switch_HasHistory/1', 'next_has_history/1', 'autorouting', 'on');
    add_line(model_name, 'Switch_LastDecision/1', 'next_last_decision/1', 'autorouting', 'on');

    fprintf('✅ 所有模块已连接\n');
catch ME
    fprintf('⚠️  连线时出错: %s\n', ME.message);
end

%% 11. 保存模型
save_system(model_name);
fprintf('\n💾 模型已保存: %s.slx\n', model_name);

%% 12. 完成
fprintf('\n');
fprintf('====================================================================\n');
fprintf('🎉 模型创建完成！\n');
fprintf('====================================================================\n');
fprintf('模型: %s.slx\n', model_name);
fprintf('使用: n-D Lookup Table (配置为2维)\n');
fprintf('\n下一步:\n');
fprintf('1. 在模型中手动检查查找表配置是否正确\n');
fprintf('2. 如有参数未设置，请手动在模块属性中配置\n');
fprintf('3. 测试模型功能\n');
fprintf('====================================================================\n');

end
