%% 测试Simulink模型配置
% 检查两个模型的Sample Time和StopTime配置是否正确

fprintf('=== 检查Simulink模型配置 ===\n\n');

%% 1. 检查决策模型配置
model1 = 'acc_decision_core';
fprintf('1️⃣  检查决策模型: %s\n', model1);
fprintf('--------------------------------------------------\n');

% 加载模型（如果未加载）
if ~bdIsLoaded(model1)
    load_system(model1);
    fprintf('✅ 模型已加载\n');
else
    fprintf('✅ 模型已在工作区\n');
end

% 获取仿真状态
sim_status = get_param(model1, 'SimulationStatus');
fprintf('仿真状态: %s\n', sim_status);

% 获取关键配置参数
stop_time = get_param(model1, 'StopTime');
solver = get_param(model1, 'Solver');
fixed_step = get_param(model1, 'FixedStep');
sim_mode = get_param(model1, 'SimulationMode');

fprintf('StopTime: %s (建议: inf)\n', stop_time);
fprintf('Solver: %s (建议: FixedStepDiscrete)\n', solver);
fprintf('FixedStep: %s (建议: 0.05)\n', fixed_step);
fprintf('SimulationMode: %s\n', sim_mode);

% 检查是否符合要求
decision_ok = true;
if ~strcmp(stop_time, 'inf')
    fprintf('⚠️  警告: StopTime应该是inf，当前是%s\n', stop_time);
    decision_ok = false;
end
if ~strcmp(solver, 'FixedStepDiscrete')
    fprintf('⚠️  警告: Solver应该是FixedStepDiscrete，当前是%s\n', solver);
    decision_ok = false;
end
if ~strcmp(fixed_step, '0.05')
    fprintf('⚠️  警告: FixedStep应该是0.05，当前是%s\n', fixed_step);
    decision_ok = false;
end

if decision_ok
    fprintf('✅ 决策模型配置正确\n');
else
    fprintf('❌ 决策模型配置需要修改\n');
end

fprintf('\n');

%% 2. 检查SPPVT控制模型配置
model2 = 'sppvt_control_model';
fprintf('2️⃣  检查SPPVT控制模型: %s\n', model2);
fprintf('--------------------------------------------------\n');

% 加载模型（如果未加载）
if ~bdIsLoaded(model2)
    load_system(model2);
    fprintf('✅ 模型已加载\n');
else
    fprintf('✅ 模型已在工作区\n');
end

% 获取仿真状态
sim_status = get_param(model2, 'SimulationStatus');
fprintf('仿真状态: %s\n', sim_status);

% 获取关键配置参数
stop_time = get_param(model2, 'StopTime');
solver = get_param(model2, 'Solver');
fixed_step = get_param(model2, 'FixedStep');
sim_mode = get_param(model2, 'SimulationMode');

fprintf('StopTime: %s (建议: inf)\n', stop_time);
fprintf('Solver: %s (建议: FixedStepDiscrete)\n', solver);
fprintf('FixedStep: %s (建议: 0.05)\n', fixed_step);
fprintf('SimulationMode: %s\n', sim_mode);

% 检查SPPVT参数（Constant模块）
fprintf('\nSPPVT Constant模块参数:\n');
param_blocks = {'SPPVT_dt', 'SPPVT_kp', 'SPPVT_max_accel', ...
                'SPPVT_max_decel', 'SPPVT_delta', 'SPPVT_eta'};
expected_values = {0.05, 1.0, 2.0, -3.0, 0.05, 0.2};

for i = 1:length(param_blocks)
    block_path = [model2 '/' param_blocks{i}];
    try
        value = get_param(block_path, 'Value');
        fprintf('  %s = %s (期望: %.2f)\n', param_blocks{i}, value, expected_values{i});
    catch
        fprintf('  ⚠️  %s 不存在或无法读取\n', param_blocks{i});
    end
end

% 检查是否符合要求
sppvt_ok = true;
if ~strcmp(stop_time, 'inf')
    fprintf('⚠️  警告: StopTime应该是inf，当前是%s\n', stop_time);
    sppvt_ok = false;
end
if ~strcmp(solver, 'FixedStepDiscrete')
    fprintf('⚠️  警告: Solver应该是FixedStepDiscrete，当前是%s\n', solver);
    sppvt_ok = false;
end
if ~strcmp(fixed_step, '0.05')
    fprintf('⚠️  警告: FixedStep应该是0.05，当前是%s\n', fixed_step);
    sppvt_ok = false;
end

if sppvt_ok
    fprintf('✅ SPPVT模型配置正确\n');
else
    fprintf('❌ SPPVT模型配置需要修改\n');
end

fprintf('\n');

%% 3. 自动修复配置（可选）
fprintf('3️⃣  是否需要自动修复配置？\n');
fprintf('--------------------------------------------------\n');

if ~decision_ok || ~sppvt_ok
    fprintf('发现配置问题，可以运行以下命令自动修复：\n');
    fprintf('  run fix_simulink_config.m\n');

    % 创建修复脚本
    fid = fopen('fix_simulink_config.m', 'w');
    fprintf(fid, '%% 自动修复Simulink模型配置\n\n');

    if ~decision_ok
        fprintf(fid, '%% 修复决策模型\n');
        fprintf(fid, 'set_param(''%s'', ''StopTime'', ''inf'');\n', model1);
        fprintf(fid, 'set_param(''%s'', ''Solver'', ''FixedStepDiscrete'');\n', model1);
        fprintf(fid, 'set_param(''%s'', ''FixedStep'', ''0.05'');\n', model1);
        fprintf(fid, 'fprintf(''✅ 决策模型配置已修复\\n'');\n\n');
    end

    if ~sppvt_ok
        fprintf(fid, '%% 修复SPPVT模型\n');
        fprintf(fid, 'set_param(''%s'', ''StopTime'', ''inf'');\n', model2);
        fprintf(fid, 'set_param(''%s'', ''Solver'', ''FixedStepDiscrete'');\n', model2);
        fprintf(fid, 'set_param(''%s'', ''FixedStep'', ''0.05'');\n', model2);
        fprintf(fid, 'fprintf(''✅ SPPVT模型配置已修复\\n'');\n\n');

        % 添加参数设置
        fprintf(fid, '%% 设置SPPVT参数\n');
        for i = 1:length(param_blocks)
            fprintf(fid, 'set_param(''%s/%s'', ''Value'', ''%.2f'');\n', ...
                model2, param_blocks{i}, expected_values{i});
        end
        fprintf(fid, 'fprintf(''✅ SPPVT参数已设置\\n'');\n');
    end

    fclose(fid);
    fprintf('✅ 修复脚本已创建: fix_simulink_config.m\n');
else
    fprintf('✅ 所有配置都正确，无需修复\n');
end

fprintf('\n');

%% 4. 总结
fprintf('=== 配置检查总结 ===\n');
if decision_ok && sppvt_ok
    fprintf('✅ 所有模型配置正确，可以运行Python程序\n');
    fprintf('\n启动步骤:\n');
    fprintf('  1. 启动决策模型: set_param(''%s'', ''SimulationCommand'', ''start'');\n', model1);
    fprintf('  2. 启动SPPVT模型: set_param(''%s'', ''SimulationCommand'', ''start'');\n', model2);
    fprintf('  3. 运行Python程序: python acc_updated.py\n');
else
    fprintf('❌ 存在配置问题，请修复后再运行\n');
    if exist('fix_simulink_config.m', 'file')
        fprintf('   运行 fix_simulink_config.m 自动修复\n');
    end
end