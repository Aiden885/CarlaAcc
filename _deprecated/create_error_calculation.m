%% create_error_calculation.m
% 生成 Error_Calculation 子系统（纯 Simulink 基础模块，无 MATLAB Function）
% 用法：
%   1. 在 MATLAB 命令窗口运行：create_error_calculation('acc_integrated_model')
%   2. 或者直接运行：create_error_calculation()
%   3. 将子系统接入总模型（接线说明见末尾输出）
%
% 逻辑（固定时距控制模式）：
%   safe_speed = max(ego_speed, 0.1)
%   actual_gap = dist / safe_speed
%   error      = actual_gap - G2
%
%   正值 = 时距过大(太远)需加速
%   负值 = 时距过小(太近)需减速
%
% Simulink 2024b

function create_error_calculation(modelName)
    if nargin < 1
        modelName = 'acc_integrated_model';
    end

    if ~bdIsLoaded(modelName)
        load_system(modelName);
    end

    subsysName = 'Error_Calculation';
    subsysPath = [modelName '/' subsysName];

    % 如果已存在，先删除
    if getSimulinkBlockHandle(subsysPath) ~= -1
        delete_block(subsysPath);
    end

    % 创建子系统
    add_block('simulink/Ports & Subsystems/Subsystem', subsysPath);

    % 删除默认内容
    delete_line(subsysPath, 'In1/1', 'Out1/1');
    delete_block([subsysPath '/In1']);
    delete_block([subsysPath '/Out1']);

    %% ========== 参数 ==========
    MIN_SPEED = 0.1;  % 避免除零的最小速度 (m/s)

    %% ==================== 第1部分：创建所有块 ====================

    %% 输入端口 (3个)
    add_block('simulink/Sources/In1', [subsysPath '/ego_speed_ms'], ...
        'Position', [30, 100, 60, 114], 'Port', '1');
    add_block('simulink/Sources/In1', [subsysPath '/vehicle_distance'], ...
        'Position', [30, 200, 60, 214], 'Port', '2');
    add_block('simulink/Sources/In1', [subsysPath '/G2_s'], ...
        'Position', [30, 300, 60, 314], 'Port', '3');

    %% 输出端口 (1个)
    add_block('simulink/Sinks/Out1', [subsysPath '/control_error_signed'], ...
        'Position', [600, 190, 630, 206], 'Port', '1');

    %% 常量块
    add_block('simulink/Sources/Constant', [subsysPath '/Const_min_speed'], ...
        'Position', [100, 150, 140, 170], 'Value', num2str(MIN_SPEED));

    %% safe_speed = max(ego_speed_ms, MIN_SPEED)
    add_block('simulink/Math Operations/MinMax', [subsysPath '/Max_safe_speed'], ...
        'Position', [200, 100, 240, 140], 'Function', 'max', 'Inputs', '2');

    %% actual_gap = vehicle_distance / safe_speed
    add_block('simulink/Math Operations/Divide', [subsysPath '/Div_actual_gap'], ...
        'Position', [300, 180, 340, 220]);

    %% error = actual_gap - G2_s
    add_block('simulink/Math Operations/Subtract', [subsysPath '/Sub_time_error'], ...
        'Position', [430, 180, 470, 210]);

    %% ==================== 第2部分：连接所有信号线 ====================

    % safe_speed = max(ego_speed, 0.1)
    add_line(subsysPath, 'ego_speed_ms/1', 'Max_safe_speed/1');
    add_line(subsysPath, 'Const_min_speed/1', 'Max_safe_speed/2');

    % actual_gap = dist / safe_speed
    add_line(subsysPath, 'vehicle_distance/1', 'Div_actual_gap/1');
    add_line(subsysPath, 'Max_safe_speed/1', 'Div_actual_gap/2');

    % error = actual_gap(+) - G2(-)
    add_line(subsysPath, 'Div_actual_gap/1', 'Sub_time_error/1');
    add_line(subsysPath, 'G2_s/1', 'Sub_time_error/2');

    % 输出
    add_line(subsysPath, 'Sub_time_error/1', 'control_error_signed/1');

    %% ========== 设置子系统外观 ==========
    set_param(subsysPath, 'Position', [200, 400, 380, 480]);

    save_system(modelName);

    fprintf('\n========================================\n');
    fprintf('Error_Calculation 子系统已创建成功！\n');
    fprintf('路径：%s\n', subsysPath);
    fprintf('========================================\n');
    fprintf('输入端口：\n');
    fprintf('  1. ego_speed_ms      ← Demux #5\n');
    fprintf('  2. vehicle_distance  ← Demux #6\n');
    fprintf('  3. G2_s              ← Demux #7\n');
    fprintf('输出端口：\n');
    fprintf('  1. control_error_signed → SPPVT 的误差输入\n');
    fprintf('========================================\n');
    fprintf('Demux #8 (target_speed_ms) 为前车速度，仅显示用，\n');
    fprintf('不连接 Error_Calculation，可接 Display/Scope。\n');
    fprintf('========================================\n');
end
