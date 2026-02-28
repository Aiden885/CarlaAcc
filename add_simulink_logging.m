%% add_simulink_logging.m
% 在 acc_integrated_model 中添加综合信号记录子系统
%
% 保存全部 18 路关键信号到 acc_simulink_log.mat
% 矩阵格式: acc_log (19 × N)，第1行为仿真时间，第2~19行为各信号值
%
% 信号列表（行索引对应 acc_log 的行号）:
%   行1  : 仿真时间 (s)          [Simulink 自动]
%   ---- 输入信号 (Python→Simulink) ----
%   行2  : command_type          ← Demux #1
%   行3  : ego_speed_ms          ← Demux #2
%   行4  : vehicle_distance      ← Demux #3
%   行5  : target_speed_ms       ← Demux #4
%   行6  : current_engine_torque ← Demux #5
%   ---- 核心计算信号 ----
%   行7  : G2_s                  ← G2_Manager/1
%   行8  : control_error_signed  ← Error_Calculation/1
%   行9  : reset_flag            ← Reset_Flag_Detector/1
%   行10 : Y0                    ← Y0_Latch/1
%   行11 : final_output (Nm)     ← Torque_Arbitration/1
%   ---- 决策状态 ----
%   行12 : control_enabled       ← Decision/control_enabled
%   行13 : current_state         ← Low_Speed_Detection/current_state
%   行14 : decision_out          ← Decision/decision
%   ---- SPPVT 内部状态 ----
%   行15 : stage_offset          ← StageManager/1
%   行16 : stage                 ← StageManager/2
%   行17 : error_sign            ← StageManager/3
%   行18 : cooldown              ← StageManager/4
%   行19 : sign_changed          ← StageManager/5
%
% 用法:
%   add_simulink_logging
%   add_simulink_logging('acc_integrated_model')

function add_simulink_logging(modelName)
    if nargin < 1
        modelName = 'acc_integrated_model';
    end

    if ~bdIsLoaded(modelName)
        load_system(modelName);
    end

    fprintf('=== 添加 Simulink 综合信号记录模块 ===\n\n');

    subsysPath = [modelName '/Logging'];

    if getSimulinkBlockHandle(subsysPath) ~= -1
        delete_block(subsysPath);
        fprintf('已删除旧版 Logging 子系统\n');
    end

    add_block('simulink/Ports & Subsystems/Subsystem', subsysPath, ...
        'Position', [1200, 300, 1420, 680]);
    delete_line(subsysPath, 'In1/1', 'Out1/1');
    delete_block([subsysPath '/In1']);
    delete_block([subsysPath '/Out1']);

    sys = subsysPath;

    %% 信号端口定义（顺序即对应 acc_log 的行2~19）
    port_labels = { ...
        'command_type', ...          % 行2
        'ego_speed_ms', ...          % 行3
        'vehicle_distance', ...      % 行4
        'target_speed_ms', ...       % 行5
        'current_engine_torque', ... % 行6
        'G2_s', ...                  % 行7
        'control_error_signed', ...  % 行8
        'reset_flag', ...            % 行9
        'Y0', ...                    % 行10
        'final_output', ...          % 行11
        'control_enabled', ...       % 行12
        'current_state', ...         % 行13
        'decision_out', ...          % 行14
        'stage_offset', ...          % 行15
        'stage', ...                 % 行16
        'error_sign', ...            % 行17
        'cooldown', ...              % 行18
        'sign_changed' ...           % 行19
    };
    N_SIG = numel(port_labels);  % = 18

    %% boolean 信号索引（需在进 Mux 前转 double）
    bool_ports = [8, 11, 18];   % reset_flag, control_enabled, sign_changed

    %% 创建输入端口
    for i = 1:N_SIG
        add_block('simulink/Sources/In1', [sys '/' port_labels{i}], ...
            'Position', [30, 20 + (i-1)*50, 70, 34 + (i-1)*50], ...
            'Port', num2str(i));
    end

    %% 为 boolean 信号添加 Data Type Conversion (boolean → double)
    for i = bool_ports
        conv_name = ['DTC_' port_labels{i}];
        add_block('simulink/Signal Attributes/Data Type Conversion', ...
            [sys '/' conv_name], ...
            'Position', [110, 20 + (i-1)*50, 150, 34 + (i-1)*50], ...
            'OutDataTypeStr', 'double');
        add_line(sys, [port_labels{i} '/1'], [conv_name '/1']);
    end

    %% Mux
    mux_top    = 20;
    mux_bottom = 20 + N_SIG * 50;
    add_block('simulink/Signal Routing/Mux', [sys '/Mux_log'], ...
        'Position', [200, mux_top, 230, mux_bottom], ...
        'Inputs', num2str(N_SIG), ...
        'DisplayOption', 'none');

    %% To File
    log_filename = fullfile(pwd, 'acc_simulink_log.mat');
    add_block('simulink/Sinks/To File', [sys '/ToFile_log'], ...
        'Position', [310, (mux_top+mux_bottom)/2 - 30, 480, (mux_top+mux_bottom)/2 + 30], ...
        'Filename',   log_filename, ...
        'MatrixName', 'acc_log', ...
        'Decimation', '1', ...
        'SampleTime', '-1');

    %% 连线：输入端口 → Mux（boolean 信号经 DTC 中转）
    for i = 1:N_SIG
        if ismember(i, bool_ports)
            src = ['DTC_' port_labels{i} '/1'];
        else
            src = [port_labels{i} '/1'];
        end
        add_line(sys, src, ['Mux_log/' num2str(i)]);
    end
    add_line(sys, 'Mux_log/1', 'ToFile_log/1');

    save_system(modelName);

    %% 输出接线说明
    fprintf('✅ Logging 子系统创建完成（18路信号 → acc_simulink_log.mat）\n\n');
    fprintf('========================================\n');
    fprintf('⚠️  请在 Simulink 中将以下信号连接到 Logging 子系统对应端口:\n\n');
    fprintf('  【输入信号 (Python→Simulink)】\n');
    fprintf('  端口/1  command_type          ← Demux #1\n');
    fprintf('  端口/2  ego_speed_ms          ← Demux #2\n');
    fprintf('  端口/3  vehicle_distance      ← Demux #3\n');
    fprintf('  端口/4  target_speed_ms       ← Demux #4\n');
    fprintf('  端口/5  current_engine_torque ← Demux #5\n\n');
    fprintf('  【核心计算信号】\n');
    fprintf('  端口/6  G2_s                  ← G2_Manager/1\n');
    fprintf('  端口/7  control_error_signed  ← Error_Calculation/1\n');
    fprintf('  端口/8  reset_flag            ← Reset_Flag_Detector/1\n');
    fprintf('  端口/9  Y0                    ← Y0_Latch/1\n');
    fprintf('  端口/10 final_output          ← Torque_Arbitration/1\n\n');
    fprintf('  【决策状态】\n');
    fprintf('  端口/11 control_enabled       ← Decision/control_enabled\n');
    fprintf('  端口/12 current_state         ← Low_Speed_Detection/current_state\n');
    fprintf('  端口/13 decision_out          ← Decision/decision\n\n');
    fprintf('  【SPPVT 内部状态】\n');
    fprintf('  端口/14 stage_offset          ← StageManager/1\n');
    fprintf('  端口/15 stage                 ← StageManager/2\n');
    fprintf('  端口/16 error_sign            ← StageManager/3\n');
    fprintf('  端口/17 cooldown              ← StageManager/4\n');
    fprintf('  端口/18 sign_changed          ← StageManager/5\n\n');
    fprintf('保存路径: %s\n', log_filename);
    fprintf('矩阵大小: 19 × N（第1行=时间，第2~19行=各信号）\n');
    fprintf('========================================\n');
    fprintf('仿真结束后画图:\n');
    fprintf('  plot_simulink_trace\n');
    fprintf('  plot_simulink_trace(''acc_simulink_log.mat'')\n');
    fprintf('========================================\n');
end
