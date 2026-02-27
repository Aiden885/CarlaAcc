function create_simplified_subsystems(modelName)
%% create_simplified_subsystems.m
% 为简化架构创建所有新子系统（全部使用 Unit Delay）
%
% 新架构:
%   Python → Simulink [5个double]:
%     [command_type, ego_speed_ms, vehicle_distance, target_speed_ms, current_engine_torque]
%   Simulink → Python [2个double]:
%     [control_enabled, final_output]
%
% 创建子系统:
%   1. Low_Speed_Detection  - 低速检测与状态覆盖
%   2. G2_Manager           - 时距参数管理 (初始值4, T/R键±0.2, 范围[1,5])
%   3. Reset_Flag_Detector  - 重置标志检测 (control_enabled下降沿 + G2突变)
%   4. Torque_Arbitration   - R7扭矩仲裁 max(control_output, engine_torque)
%   5. Y0_Latch             - Unit Delay版本 (替换原Memory版本)
%
% 另需手动添加:
%   3个 Unit Delay 用于状态自回环 (state, has_history, last_decision)
%
% 模型工作区需要的变量:
%   V_min_ms  - 最低速度阈值 (m/s)，默认 30/3.6 ≈ 8.33
%
% 用法:
%   create_simplified_subsystems()
%   create_simplified_subsystems('acc_integrated_model')
%
% Simulink 2024b

    if nargin < 1
        modelName = 'acc_integrated_model';
    end

    if ~bdIsLoaded(modelName)
        load_system(modelName);
    end

    fprintf('=== 创建简化架构子系统 ===\n\n');

    create_low_speed_detection(modelName);
    create_g2_manager(modelName);
    create_reset_flag_detector(modelName);
    create_torque_arbitration(modelName);
    create_y0_latch_v2(modelName);

    %% 设置模型工作区变量
    fprintf('6. 设置模型工作区变量...\n');
    mdlWks = get_param(modelName, 'ModelWorkspace');
    if ~mdlWks.hasVariable('V_min_ms')
        mdlWks.assignin('V_min_ms', 30.0 / 3.6);  % 30 km/h → 8.33 m/s
        fprintf('   ✅ V_min_ms = %.4f (30 km/h)\n', 30.0 / 3.6);
    else
        fprintf('   ⏭  V_min_ms 已存在，跳过\n');
    end

    save_system(modelName);

    fprintf('\n========================================\n');
    fprintf('所有子系统创建完成！\n');
    fprintf('========================================\n\n');
    fprintf('还需手动完成以下操作:\n\n');
    fprintf('1. UDP Receive: DataSize [10 1] → [5 1]\n');
    fprintf('2. Demux: 10路 → 5路\n');
    fprintf('   新 Demux 输出:\n');
    fprintf('     #1 command_type       → Decision, G2_Manager\n');
    fprintf('     #2 ego_speed_ms       → Error_Calculation, Low_Speed_Detection\n');
    fprintf('     #3 vehicle_distance   → Error_Calculation\n');
    fprintf('     #4 target_speed_ms    → (显示用，可接Scope)\n');
    fprintf('     #5 current_engine_torque → Y0_Latch, Torque_Arbitration\n\n');
    fprintf('3. 添加3个 Unit Delay (状态自回环):\n');
    fprintf('   UD_state:    IC=2 (S2)  next_state → Low_Speed_Detection/mem_state\n');
    fprintf('   UD_history:  IC=0       next_has_history → Low_Speed_Detection/has_history\n');
    fprintf('                                            → Decision/has_history\n');
    fprintf('   UD_last_dec: IC=8 (R8)  next_last_decision → Decision/last_active_decision\n\n');
    fprintf('4. 接线概要:\n');
    fprintf('   Low_Speed_Detection/current_state → Decision/current_state\n');
    fprintf('   G2_Manager/G2_s → Error_Calculation/G2_s\n');
    fprintf('   G2_Manager/G2_s → Reset_Flag_Detector/G2_s\n');
    fprintf('   Reset_Flag_Detector/reset_flag → SPPVT/reset_flag\n');
    fprintf('   Torque_Arbitration/final_output → Mux → UDP Send\n\n');
    fprintf('5. Mux: 改为2路 [control_enabled, final_output]\n');
    fprintf('6. UDP Send: 对应调整数据宽度\n');
    fprintf('========================================\n');
end


%% ============================================================
%% 1. Low_Speed_Detection
%% ============================================================
function create_low_speed_detection(modelName)
    fprintf('1. 创建 Low_Speed_Detection...\n');

    subsysPath = [modelName '/Low_Speed_Detection'];

    if getSimulinkBlockHandle(subsysPath) ~= -1
        delete_block(subsysPath);
    end

    add_block('simulink/Ports & Subsystems/Subsystem', subsysPath);
    delete_line(subsysPath, 'In1/1', 'Out1/1');
    delete_block([subsysPath '/In1']);
    delete_block([subsysPath '/Out1']);

    sys = subsysPath;

    %% 输入端口 (3个)
    add_block('simulink/Sources/In1', [sys '/ego_speed_ms'], ...
        'Position', [30, 43, 60, 57], 'Port', '1');
    add_block('simulink/Sources/In1', [sys '/mem_state'], ...
        'Position', [30, 163, 60, 177], 'Port', '2');
    add_block('simulink/Sources/In1', [sys '/has_history'], ...
        'Position', [30, 243, 60, 257], 'Port', '3');

    %% 输出端口
    add_block('simulink/Sinks/Out1', [sys '/current_state'], ...
        'Position', [630, 53, 660, 67], 'Port', '1');

    %% 常量
    add_block('simulink/Sources/Constant', [sys '/V_min_ms'], ...
        'Position', [30, 83, 80, 97], 'Value', 'V_min_ms');
    add_block('simulink/Sources/Constant', [sys '/Const_3_cmp'], ...
        'Position', [100, 193, 130, 207], 'Value', '3');
    add_block('simulink/Sources/Constant', [sys '/Const_S3'], ...
        'Position', [440, 23, 470, 37], 'Value', '3');
    add_block('simulink/Sources/Constant', [sys '/Const_S1'], ...
        'Position', [220, 213, 250, 227], 'Value', '1');
    add_block('simulink/Sources/Constant', [sys '/Const_S2'], ...
        'Position', [220, 273, 250, 287], 'Value', '2');

    %% 比较器
    % ego_speed_ms < V_min_ms → is_low_speed
    add_block('simulink/Logic and Bit Operations/Relational Operator', ...
        [sys '/Cmp_low_speed'], ...
        'Position', [150, 48, 180, 82], 'Operator', '<');

    % mem_state == 3 → is_S3
    add_block('simulink/Logic and Bit Operations/Relational Operator', ...
        [sys '/Cmp_is_S3'], ...
        'Position', [180, 158, 210, 192], 'Operator', '==');

    %% Switch 链
    % Sw_recovery: has_history >= 0.5 ? S1(1) : S2(2)
    add_block('simulink/Signal Routing/Switch', [sys '/Sw_recovery'], ...
        'Position', [320, 220, 360, 280], ...
        'Criteria', 'u2 >= Threshold', 'Threshold', '0.5');

    % Sw_s3_check: is_S3 ? recovery_state : mem_state
    add_block('simulink/Signal Routing/Switch', [sys '/Sw_s3_check'], ...
        'Position', [430, 145, 470, 205], ...
        'Criteria', 'u2 >= Threshold', 'Threshold', '0.5');

    % Sw_main: is_low_speed ? S3(3) : normal_state
    add_block('simulink/Signal Routing/Switch', [sys '/Sw_main'], ...
        'Position', [540, 30, 580, 90], ...
        'Criteria', 'u2 >= Threshold', 'Threshold', '0.5');

    %% 连线
    add_line(sys, 'ego_speed_ms/1', 'Cmp_low_speed/1');
    add_line(sys, 'V_min_ms/1', 'Cmp_low_speed/2');

    add_line(sys, 'mem_state/1', 'Cmp_is_S3/1');
    add_line(sys, 'Const_3_cmp/1', 'Cmp_is_S3/2');

    add_line(sys, 'Const_S1/1', 'Sw_recovery/1');
    add_line(sys, 'has_history/1', 'Sw_recovery/2');
    add_line(sys, 'Const_S2/1', 'Sw_recovery/3');

    add_line(sys, 'Sw_recovery/1', 'Sw_s3_check/1');
    add_line(sys, 'Cmp_is_S3/1', 'Sw_s3_check/2');
    add_line(sys, 'mem_state/1', 'Sw_s3_check/3');

    add_line(sys, 'Const_S3/1', 'Sw_main/1');
    add_line(sys, 'Cmp_low_speed/1', 'Sw_main/2');
    add_line(sys, 'Sw_s3_check/1', 'Sw_main/3');

    add_line(sys, 'Sw_main/1', 'current_state/1');

    set_param(subsysPath, 'Position', [100, 100, 280, 180]);
    fprintf('   ✅ Low_Speed_Detection\n\n');
end


%% ============================================================
%% 2. G2_Manager
%% ============================================================
function create_g2_manager(modelName)
    fprintf('2. 创建 G2_Manager...\n');

    subsysPath = [modelName '/G2_Manager'];

    if getSimulinkBlockHandle(subsysPath) ~= -1
        delete_block(subsysPath);
    end

    add_block('simulink/Ports & Subsystems/Subsystem', subsysPath);
    delete_line(subsysPath, 'In1/1', 'Out1/1');
    delete_block([subsysPath '/In1']);
    delete_block([subsysPath '/Out1']);

    sys = subsysPath;

    %% 输入端口
    add_block('simulink/Sources/In1', [sys '/command_type'], ...
        'Position', [30, 93, 60, 107], 'Port', '1');

    %% 输出端口
    add_block('simulink/Sinks/Out1', [sys '/G2_s'], ...
        'Position', [700, 123, 730, 137], 'Port', '1');

    %% 常量
    add_block('simulink/Sources/Constant', [sys '/Const_T_key'], ...
        'Position', [30, 143, 60, 157], 'Value', '3');
    add_block('simulink/Sources/Constant', [sys '/Const_R_key'], ...
        'Position', [30, 223, 60, 237], 'Value', '4');
    add_block('simulink/Sources/Constant', [sys '/Const_step'], ...
        'Position', [120, 333, 160, 347], 'Value', '0.2');

    %% Unit Delay: 保持G2值 (IC=4.0)
    add_block('simulink/Discrete/Unit Delay', [sys '/UD_G2'], ...
        'Position', [120, 280, 160, 310], ...
        'InitialCondition', '4', 'SampleTime', '-1');

    %% 比较器
    % command_type == 3 → is_T_key
    add_block('simulink/Logic and Bit Operations/Relational Operator', ...
        [sys '/Cmp_T_key'], ...
        'Position', [140, 98, 170, 132], 'Operator', '==');
    % command_type == 4 → is_R_key
    add_block('simulink/Logic and Bit Operations/Relational Operator', ...
        [sys '/Cmp_R_key'], ...
        'Position', [140, 198, 170, 232], 'Operator', '==');

    %% 算术: prev_G2 ± 0.2
    % Sum_sub: prev_G2 - 0.2
    add_block('simulink/Math Operations/Sum', [sys '/Sum_sub'], ...
        'Position', [260, 278, 290, 308], 'Inputs', '+-');
    % Sum_add: prev_G2 + 0.2
    add_block('simulink/Math Operations/Sum', [sys '/Sum_add'], ...
        'Position', [260, 338, 290, 368], 'Inputs', '++');

    %% Switch 链
    % Sw_R: is_R_key ? (prev+0.2) : prev_G2   ← R键(R4)延长时距,G2增大
    add_block('simulink/Signal Routing/Switch', [sys '/Sw_R'], ...
        'Position', [380, 190, 420, 250], ...
        'Criteria', 'u2 >= Threshold', 'Threshold', '0.5');

    % Sw_T: is_T_key ? (prev-0.2) : Sw_R_output  ← T键(R3)缩短时距,G2减小
    add_block('simulink/Signal Routing/Switch', [sys '/Sw_T'], ...
        'Position', [500, 90, 540, 150], ...
        'Criteria', 'u2 >= Threshold', 'Threshold', '0.5');

    %% 限幅 [1.0, 5.0]
    add_block('simulink/Discontinuities/Saturation', [sys '/Sat_G2'], ...
        'Position', [600, 108, 640, 132], ...
        'UpperLimit', '5', 'LowerLimit', '1');

    %% 连线
    % 比较器输入
    add_line(sys, 'command_type/1', 'Cmp_T_key/1');
    add_line(sys, 'Const_T_key/1', 'Cmp_T_key/2');
    add_line(sys, 'command_type/1', 'Cmp_R_key/1');
    add_line(sys, 'Const_R_key/1', 'Cmp_R_key/2');

    % 加减运算
    add_line(sys, 'UD_G2/1', 'Sum_sub/1');
    add_line(sys, 'Const_step/1', 'Sum_sub/2');
    add_line(sys, 'UD_G2/1', 'Sum_add/1');
    add_line(sys, 'Const_step/1', 'Sum_add/2');

    % Sw_R: is_R ? (prev+0.2) : prev    ← R键(R4)延长时距,G2增大
    add_line(sys, 'Sum_add/1', 'Sw_R/1');
    add_line(sys, 'Cmp_R_key/1', 'Sw_R/2');
    add_line(sys, 'UD_G2/1', 'Sw_R/3');

    % Sw_T: is_T ? (prev-0.2) : Sw_R  ← T键(R3)缩短时距,G2减小
    add_line(sys, 'Sum_sub/1', 'Sw_T/1');
    add_line(sys, 'Cmp_T_key/1', 'Sw_T/2');
    add_line(sys, 'Sw_R/1', 'Sw_T/3');

    % 限幅 → 输出 + 反馈
    add_line(sys, 'Sw_T/1', 'Sat_G2/1');
    add_line(sys, 'Sat_G2/1', 'G2_s/1');
    add_line(sys, 'Sat_G2/1', 'UD_G2/1');  % 反馈

    set_param(subsysPath, 'Position', [100, 200, 280, 250]);
    fprintf('   ✅ G2_Manager\n\n');
end


%% ============================================================
%% 3. Reset_Flag_Detector
%% ============================================================
function create_reset_flag_detector(modelName)
    fprintf('3. 创建 Reset_Flag_Detector...\n');

    subsysPath = [modelName '/Reset_Flag_Detector'];

    if getSimulinkBlockHandle(subsysPath) ~= -1
        delete_block(subsysPath);
    end

    add_block('simulink/Ports & Subsystems/Subsystem', subsysPath);
    delete_line(subsysPath, 'In1/1', 'Out1/1');
    delete_block([subsysPath '/In1']);
    delete_block([subsysPath '/Out1']);

    sys = subsysPath;

    %% 输入端口
    add_block('simulink/Sources/In1', [sys '/control_enabled'], ...
        'Position', [30, 53, 60, 67], 'Port', '1');
    add_block('simulink/Sources/In1', [sys '/G2_s'], ...
        'Position', [30, 213, 60, 227], 'Port', '2');

    %% 输出端口
    add_block('simulink/Sinks/Out1', [sys '/reset_flag'], ...
        'Position', [580, 128, 610, 142], 'Port', '1');

    %% Unit Delay
    % 上一拍 control_enabled (IC=0)
    add_block('simulink/Discrete/Unit Delay', [sys '/UD_ce'], ...
        'Position', [120, 88, 160, 112], ...
        'InitialCondition', '0', 'SampleTime', '-1');
    % 上一拍 G2_s (IC=4)
    add_block('simulink/Discrete/Unit Delay', [sys '/UD_G2'], ...
        'Position', [120, 248, 160, 272], ...
        'InitialCondition', '4', 'SampleTime', '-1');

    %% control_enabled 下降沿检测
    % prev_ce > current_ce  (1→0 时为 true)
    add_block('simulink/Logic and Bit Operations/Relational Operator', ...
        [sys '/Cmp_falling'], ...
        'Position', [250, 58, 280, 92], 'Operator', '>');

    %% G2 突变检测
    % |G2_s - prev_G2| > 0.3
    add_block('simulink/Math Operations/Sum', [sys '/Sub_G2_diff'], ...
        'Position', [250, 218, 280, 248], 'Inputs', '+-');
    add_block('simulink/Math Operations/Abs', [sys '/Abs_G2_diff'], ...
        'Position', [330, 220, 360, 244]);
    add_block('simulink/Sources/Constant', [sys '/Const_thresh'], ...
        'Position', [330, 278, 370, 292], 'Value', '0.3');
    add_block('simulink/Logic and Bit Operations/Relational Operator', ...
        [sys '/Cmp_G2_changed'], ...
        'Position', [420, 228, 450, 262], 'Operator', '>');

    %% OR: 任一条件触发 reset
    add_block('simulink/Logic and Bit Operations/Logical Operator', ...
        [sys '/OR_reset'], ...
        'Position', [500, 110, 540, 160], ...
        'Operator', 'OR', 'Inputs', '2');

    %% 连线
    % control_enabled 下降沿
    add_line(sys, 'control_enabled/1', 'UD_ce/1');
    add_line(sys, 'UD_ce/1', 'Cmp_falling/1');        % prev_ce (port 1)
    add_line(sys, 'control_enabled/1', 'Cmp_falling/2');  % current (port 2)

    % G2 突变
    add_line(sys, 'G2_s/1', 'UD_G2/1');
    add_line(sys, 'G2_s/1', 'Sub_G2_diff/1');          % G2_s (+)
    add_line(sys, 'UD_G2/1', 'Sub_G2_diff/2');         % prev_G2 (-)
    add_line(sys, 'Sub_G2_diff/1', 'Abs_G2_diff/1');
    add_line(sys, 'Abs_G2_diff/1', 'Cmp_G2_changed/1');
    add_line(sys, 'Const_thresh/1', 'Cmp_G2_changed/2');

    % OR
    add_line(sys, 'Cmp_falling/1', 'OR_reset/1');
    add_line(sys, 'Cmp_G2_changed/1', 'OR_reset/2');

    % 输出
    add_line(sys, 'OR_reset/1', 'reset_flag/1');

    set_param(subsysPath, 'Position', [100, 300, 280, 360]);
    fprintf('   ✅ Reset_Flag_Detector\n\n');
end


%% ============================================================
%% 4. Torque_Arbitration
%% ============================================================
function create_torque_arbitration(modelName)
    fprintf('4. 创建 Torque_Arbitration...\n');

    subsysPath = [modelName '/Torque_Arbitration'];

    if getSimulinkBlockHandle(subsysPath) ~= -1
        delete_block(subsysPath);
    end

    add_block('simulink/Ports & Subsystems/Subsystem', subsysPath);
    delete_line(subsysPath, 'In1/1', 'Out1/1');
    delete_block([subsysPath '/In1']);
    delete_block([subsysPath '/Out1']);

    sys = subsysPath;

    %% 输入端口
    add_block('simulink/Sources/In1', [sys '/control_output'], ...
        'Position', [30, 43, 60, 57], 'Port', '1');
    add_block('simulink/Sources/In1', [sys '/current_engine_torque'], ...
        'Position', [30, 143, 60, 157], 'Port', '2');
    add_block('simulink/Sources/In1', [sys '/decision'], ...
        'Position', [30, 213, 60, 227], 'Port', '3');

    %% 输出端口
    add_block('simulink/Sinks/Out1', [sys '/final_output'], ...
        'Position', [500, 73, 530, 87], 'Port', '1');

    %% decision == 7 → is_R7
    add_block('simulink/Sources/Constant', [sys '/Const_R7'], ...
        'Position', [100, 243, 130, 257], 'Value', '7');
    add_block('simulink/Logic and Bit Operations/Relational Operator', ...
        [sys '/Cmp_R7'], ...
        'Position', [180, 213, 210, 247], 'Operator', '==');

    %% max(control_output, current_engine_torque)
    add_block('simulink/Math Operations/MinMax', [sys '/Max_torque'], ...
        'Position', [230, 80, 270, 120], ...
        'Function', 'max', 'Inputs', '2');

    %% Switch: is_R7 ? max_torque : control_output
    add_block('simulink/Signal Routing/Switch', [sys '/Sw_arb'], ...
        'Position', [370, 50, 410, 110], ...
        'Criteria', 'u2 >= Threshold', 'Threshold', '0.5');

    %% 连线
    add_line(sys, 'decision/1', 'Cmp_R7/1');
    add_line(sys, 'Const_R7/1', 'Cmp_R7/2');

    add_line(sys, 'control_output/1', 'Max_torque/1');
    add_line(sys, 'current_engine_torque/1', 'Max_torque/2');

    % Switch: is_R7 ? max : control_output
    add_line(sys, 'Max_torque/1', 'Sw_arb/1');
    add_line(sys, 'Cmp_R7/1', 'Sw_arb/2');
    add_line(sys, 'control_output/1', 'Sw_arb/3');

    add_line(sys, 'Sw_arb/1', 'final_output/1');

    set_param(subsysPath, 'Position', [100, 400, 280, 460]);
    fprintf('   ✅ Torque_Arbitration\n\n');
end


%% ============================================================
%% 5. Y0_Latch (Unit Delay 版本，替换原 Memory 版本)
%% ============================================================
function create_y0_latch_v2(modelName)
    fprintf('5. 创建 Y0_Latch (Unit Delay版)...\n');

    subsysPath = [modelName '/Y0_Latch'];

    if getSimulinkBlockHandle(subsysPath) ~= -1
        delete_block(subsysPath);
        fprintf('   ⚠️  删除了旧版 Y0_Latch，需重新接线\n');
    end

    add_block('simulink/Ports & Subsystems/Subsystem', subsysPath);
    delete_line(subsysPath, 'In1/1', 'Out1/1');
    delete_block([subsysPath '/In1']);
    delete_block([subsysPath '/Out1']);

    sys = subsysPath;

    %% 输入端口 (2个)
    add_block('simulink/Sources/In1', [sys '/control_enabled'], ...
        'Position', [30, 55, 60, 69], 'Port', '1');
    add_block('simulink/Sources/In1', [sys '/current_engine_torque'], ...
        'Position', [30, 185, 60, 199], 'Port', '2');

    %% 输出端口 (1个)
    add_block('simulink/Sinks/Out1', [sys '/Y0'], ...
        'Position', [550, 142, 580, 158], 'Port', '1');

    %% 上升沿检测: control_enabled > prev_ce
    add_block('simulink/Discrete/Unit Delay', [sys '/UD_ce'], ...
        'Position', [120, 100, 160, 130], ...
        'InitialCondition', '0', 'SampleTime', '-1');
    add_block('simulink/Logic and Bit Operations/Relational Operator', ...
        [sys '/Rising_Edge'], ...
        'Position', [220, 53, 260, 87], 'Operator', '>');

    %% Y0 锁存
    % Switch: 上升沿 → 锁存 current_engine_torque，否则保持
    add_block('simulink/Signal Routing/Switch', [sys '/Sw_latch'], ...
        'Position', [380, 120, 420, 170], ...
        'Criteria', 'u2 > Threshold', 'Threshold', '0.5');
    % Unit Delay: 保持锁存值 (反馈)
    add_block('simulink/Discrete/Unit Delay', [sys '/UD_Y0'], ...
        'Position', [380, 220, 420, 250], ...
        'InitialCondition', '0', 'SampleTime', '-1');

    %% 连线
    % 上升沿检测
    add_line(sys, 'control_enabled/1', 'Rising_Edge/1');
    add_line(sys, 'control_enabled/1', 'UD_ce/1');
    add_line(sys, 'UD_ce/1', 'Rising_Edge/2');

    % Switch: 上升沿 ? engine_torque : 保持
    add_line(sys, 'current_engine_torque/1', 'Sw_latch/1');
    add_line(sys, 'Rising_Edge/1', 'Sw_latch/2');
    add_line(sys, 'UD_Y0/1', 'Sw_latch/3');

    % 输出 + 反馈
    add_line(sys, 'Sw_latch/1', 'Y0/1');
    add_line(sys, 'Sw_latch/1', 'UD_Y0/1');

    set_param(subsysPath, 'Position', [100, 500, 280, 560]);
    fprintf('   ✅ Y0_Latch (Unit Delay版)\n\n');
end
