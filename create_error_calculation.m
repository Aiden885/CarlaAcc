%% create_error_calculation.m
% 生成 Error_Calculation 子系统（纯 Simulink 基础模块，无 MATLAB Function）
% 用法：
%   1. 在 MATLAB 命令窗口运行：create_error_calculation('acc_integrated_model')
%   2. 或者直接运行：create_error_calculation() （默认模型名）
%   3. Error_Calculation 子系统将被添加到模型中
%   4. 手动将子系统接入总模型（接线说明见末尾输出）
%
% 逻辑：
%   TIME mode (flag==1):
%     有目标: error = actual_gap - G2  (即 -(G2 - actual_gap))
%     无目标: error = V_target - ego_speed (降级为速度控制)
%   SPEED mode (flag==2):
%     error = V_target - ego_speed
%
% Simulink 2024b

function create_error_calculation(modelName)
    if nargin < 1
        modelName = 'acc_integrated_model';
    end

    % 确保模型已打开
    if ~bdIsLoaded(modelName)
        load_system(modelName);
    end

    % 子系统路径
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
    MIN_SPEED = 0.1;           % 避免除零的最小速度 (m/s)
    NO_TARGET_DIST = 9995;     % 无目标判定阈值 (m)
    TIME_MODE_FLAG = 1;        % TIME 模式标志值

    %% ==================== 第1部分：创建所有块 ====================

    %% 输入端口 (5个)
    add_block('simulink/Sources/In1', [subsysPath '/ego_speed_ms'], ...
        'Position', [30, 100, 60, 114], 'Port', '1');
    add_block('simulink/Sources/In1', [subsysPath '/vehicle_distance'], ...
        'Position', [30, 200, 60, 214], 'Port', '2');
    add_block('simulink/Sources/In1', [subsysPath '/G2_s'], ...
        'Position', [30, 300, 60, 314], 'Port', '3');
    add_block('simulink/Sources/In1', [subsysPath '/V_target_ms'], ...
        'Position', [30, 400, 60, 414], 'Port', '4');
    add_block('simulink/Sources/In1', [subsysPath '/control_mode_flag'], ...
        'Position', [30, 500, 60, 514], 'Port', '5');

    %% 输出端口 (1个)
    add_block('simulink/Sinks/Out1', [subsysPath '/control_error_signed'], ...
        'Position', [900, 297, 930, 313], 'Port', '1');

    %% 常量块
    add_block('simulink/Sources/Constant', [subsysPath '/Const_min_speed'], ...
        'Position', [100, 150, 140, 170], 'Value', num2str(MIN_SPEED));
    add_block('simulink/Sources/Constant', [subsysPath '/Const_no_target_dist'], ...
        'Position', [100, 250, 140, 270], 'Value', num2str(NO_TARGET_DIST));
    add_block('simulink/Sources/Constant', [subsysPath '/Const_time_mode_flag'], ...
        'Position', [100, 550, 140, 570], 'Value', num2str(TIME_MODE_FLAG));

    %% ===== 速度误差 (SPEED mode, 也是 TIME mode 无目标时的 fallback) =====
    % speed_error = V_target_ms - ego_speed_ms
    add_block('simulink/Math Operations/Subtract', [subsysPath '/Sub_speed_error'], ...
        'Position', [200, 390, 240, 420]);
    % 输入1(+) = V_target_ms, 输入2(-) = ego_speed_ms

    %% ===== 时间间距误差 (TIME mode 有目标时) =====

    % safe_speed = max(ego_speed_ms, MIN_SPEED)
    add_block('simulink/Math Operations/MinMax', [subsysPath '/Max_safe_speed'], ...
        'Position', [200, 100, 240, 140], 'Function', 'max', 'Inputs', '2');

    % actual_gap = vehicle_distance / safe_speed
    add_block('simulink/Math Operations/Divide', [subsysPath '/Div_actual_gap'], ...
        'Position', [300, 180, 340, 220]);

    % time_error = actual_gap - G2_s
    % 即 -(G2 - actual_gap)，反号后：正值=太远需加速，负值=太近需减速
    add_block('simulink/Math Operations/Subtract', [subsysPath '/Sub_time_error'], ...
        'Position', [400, 270, 440, 300]);
    % 输入1(+) = actual_gap, 输入2(-) = G2_s

    %% ===== 无目标检测 =====
    % is_no_target = (vehicle_distance >= NO_TARGET_DIST)
    add_block('simulink/Logic and Bit Operations/Relational Operator', [subsysPath '/Cmp_no_target'], ...
        'Position', [300, 240, 340, 270], 'Operator', '>=');

    %% ===== TIME mode 内部选择：有目标用 time_error，无目标用 speed_error =====
    % Switch: u2 ~= 0 时选 u1 (no_target → speed_error)，否则选 u3 (time_error)
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_no_target'], ...
        'Position', [550, 270, 590, 330], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');
    % u1 = speed_error (无目标 fallback)
    % u2 = is_no_target (条件)
    % u3 = time_error (有目标)

    %% ===== 模式选择：TIME mode 用 time_mode_error，SPEED mode 用 speed_error =====
    % is_time_mode = (control_mode_flag == TIME_MODE_FLAG)
    add_block('simulink/Logic and Bit Operations/Relational Operator', [subsysPath '/Cmp_time_mode'], ...
        'Position', [200, 520, 240, 550], 'Operator', '==');

    % Switch: u2 ~= 0 时选 u1 (time mode)，否则选 u3 (speed mode)
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_mode'], ...
        'Position', [750, 270, 790, 330], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');
    % u1 = time_mode_error (Switch_no_target 输出)
    % u2 = is_time_mode (条件)
    % u3 = speed_error

    %% ==================== 第2部分：连接所有信号线 ====================

    %% 速度误差
    % V_target_ms → Sub_speed_error/1 (+)
    add_line(subsysPath, 'V_target_ms/1', 'Sub_speed_error/1');
    % ego_speed_ms → Sub_speed_error/2 (-)
    add_line(subsysPath, 'ego_speed_ms/1', 'Sub_speed_error/2');

    %% 安全速度
    % ego_speed_ms → Max_safe_speed/1
    add_line(subsysPath, 'ego_speed_ms/1', 'Max_safe_speed/1');
    % Const_min_speed → Max_safe_speed/2
    add_line(subsysPath, 'Const_min_speed/1', 'Max_safe_speed/2');

    %% 实际时间间距
    % vehicle_distance → Div_actual_gap/1 (被除数)
    add_line(subsysPath, 'vehicle_distance/1', 'Div_actual_gap/1');
    % Max_safe_speed → Div_actual_gap/2 (除数)
    add_line(subsysPath, 'Max_safe_speed/1', 'Div_actual_gap/2');

    %% 时间误差
    % Div_actual_gap → Sub_time_error/1 (+) (actual_gap)
    add_line(subsysPath, 'Div_actual_gap/1', 'Sub_time_error/1');
    % G2_s → Sub_time_error/2 (-) (G2)
    add_line(subsysPath, 'G2_s/1', 'Sub_time_error/2');

    %% 无目标检测
    % vehicle_distance → Cmp_no_target/1
    add_line(subsysPath, 'vehicle_distance/1', 'Cmp_no_target/1');
    % Const_no_target_dist → Cmp_no_target/2
    add_line(subsysPath, 'Const_no_target_dist/1', 'Cmp_no_target/2');

    %% Switch_no_target：无目标时用 speed_error，有目标时用 time_error
    % u1 = speed_error (无目标 fallback)
    add_line(subsysPath, 'Sub_speed_error/1', 'Switch_no_target/1');
    % u2 = is_no_target (条件)
    add_line(subsysPath, 'Cmp_no_target/1', 'Switch_no_target/2');
    % u3 = time_error (有目标)
    add_line(subsysPath, 'Sub_time_error/1', 'Switch_no_target/3');

    %% 模式检测
    % control_mode_flag → Cmp_time_mode/1
    add_line(subsysPath, 'control_mode_flag/1', 'Cmp_time_mode/1');
    % Const_time_mode_flag → Cmp_time_mode/2
    add_line(subsysPath, 'Const_time_mode_flag/1', 'Cmp_time_mode/2');

    %% Switch_mode：TIME 模式选 time_mode_error，SPEED 模式选 speed_error
    % u1 = time_mode_error (Switch_no_target 输出)
    add_line(subsysPath, 'Switch_no_target/1', 'Switch_mode/1');
    % u2 = is_time_mode (条件)
    add_line(subsysPath, 'Cmp_time_mode/1', 'Switch_mode/2');
    % u3 = speed_error
    add_line(subsysPath, 'Sub_speed_error/1', 'Switch_mode/3');

    %% 输出
    add_line(subsysPath, 'Switch_mode/1', 'control_error_signed/1');

    %% ========== 设置子系统外观 ==========
    set_param(subsysPath, 'Position', [200, 400, 400, 520]);

    % 保存模型
    save_system(modelName);

    fprintf('\n========================================\n');
    fprintf('Error_Calculation 子系统已创建成功！\n');
    fprintf('路径：%s\n', subsysPath);
    fprintf('========================================\n');
    fprintf('输入端口：\n');
    fprintf('  1. ego_speed_ms        - 自车速度 (m/s)\n');
    fprintf('  2. vehicle_distance    - 前车距离 (m)，无目标时 9999\n');
    fprintf('  3. G2_s                - 期望时距 (s)\n');
    fprintf('  4. V_target_ms         - 目标速度 (m/s)\n');
    fprintf('  5. control_mode_flag   - 控制模式 (1=TIME, 2=SPEED)\n');
    fprintf('输出端口：\n');
    fprintf('  1. control_error_signed - 控制误差（符号已处理）\n');
    fprintf('========================================\n');
    fprintf('接线说明：\n');
    fprintf('  Demux output 5  (ego_speed_ms)      → Error_Calculation/1\n');
    fprintf('  Demux output 6  (vehicle_distance)   → Error_Calculation/2\n');
    fprintf('  Demux output 7  (G2_s)               → Error_Calculation/3\n');
    fprintf('  Demux output 8  (V_target_ms)        → Error_Calculation/4\n');
    fprintf('  Demux output 9  (control_mode_flag)  → Error_Calculation/5\n');
    fprintf('  Error_Calculation/1 → 原来 control_error_signed 连接的位置\n');
    fprintf('  Demux output 10 (Y0)        → 原来 Y0 连接的位置\n');
    fprintf('  Demux output 11 (reset_flag) → 原来 reset_flag 连接的位置\n');
    fprintf('========================================\n');
    fprintf('参数设置：\n');
    fprintf('  MIN_SPEED      = %.1f m/s (除零保护)\n', MIN_SPEED);
    fprintf('  NO_TARGET_DIST = %d m (无目标判定)\n', NO_TARGET_DIST);
    fprintf('  TIME_MODE_FLAG = %d\n', TIME_MODE_FLAG);
    fprintf('========================================\n');
end
