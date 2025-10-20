% 检查ACC_Decision_SPPVT_Integrated模型的频率配置
fprintf('========================================\n');
fprintf('检查模型频率配置\n');
fprintf('========================================\n\n');

% 加载模型
model_name = 'ACC_Decision_SPPVT_Integrated';
load_system(model_name);

% 1. 模型整体配置
fprintf('1. 模型整体配置:\n');
solver_type = get_param(model_name, 'SolverType');
fixed_step = get_param(model_name, 'FixedStep');
start_time = get_param(model_name, 'StartTime');
stop_time = get_param(model_name, 'StopTime');

fprintf('   求解器类型: %s\n', solver_type);
fprintf('   固定步长: %s 秒\n', fixed_step);
fprintf('   采样频率: %.2f Hz\n', 1/str2double(fixed_step));
fprintf('   开始时间: %s\n', start_time);
fprintf('   停止时间: %s\n\n', stop_time);

% 2. 顶层模块
fprintf('2. 顶层模块采样时间:\n');
top_blocks = find_system(model_name, 'SearchDepth', 1, 'Type', 'block');
for i = 1:length(top_blocks)
    block_name = top_blocks{i};
    if strcmp(block_name, model_name)
        continue;
    end

    try
        block_type = get_param(block_name, 'BlockType');
        short_name = strrep(block_name, [model_name '/'], '');

        % 尝试获取采样时间
        try
            st = get_param(block_name, 'SampleTime');
            fprintf('   %-40s [%-15s] SampleTime: %s\n', short_name, block_type, st);
        catch
            fprintf('   %-40s [%-15s] (无SampleTime参数)\n', short_name, block_type);
        end
    catch
        % 跳过错误
    end
end

fprintf('\n3. 查找所有子系统和MATLAB Function块:\n');
% 查找所有子系统
subsystems = find_system(model_name, 'BlockType', 'SubSystem');
for i = 1:length(subsystems)
    if strcmp(subsystems{i}, model_name)
        continue;
    end

    short_name = strrep(subsystems{i}, [model_name '/'], '');
    mask_type = get_param(subsystems{i}, 'MaskType');

    if contains(mask_type, 'MATLAB')
        fprintf('   %s [MATLAB Function]\n', short_name);
    else
        fprintf('   %s [Subsystem]\n', short_name);
    end
end

% 关闭模型
close_system(model_name, 0);

fprintf('\n========================================\n');
fprintf('检查完成\n');
fprintf('========================================\n');