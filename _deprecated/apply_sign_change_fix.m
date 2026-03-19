%% apply_sign_change_fix.m
% 应用"误差变号时扭矩突降"修复
%
% 修复内容:
%   1. StageManager  - 新增第5输出端口 sign_changed_out
%   2. Y0_Latch      - 新增第3输入端口 sign_changed，变号时重锁存 Y0
%
% 运行方法:
%   在 MATLAB 命令窗口执行: apply_sign_change_fix
%   或: apply_sign_change_fix('acc_integrated_model')
%
% 运行后需手动完成的接线 (见末尾提示)

function apply_sign_change_fix(modelName)
    if nargin < 1
        modelName = 'acc_integrated_model';
    end

    if ~bdIsLoaded(modelName)
        load_system(modelName);
    end

    fprintf('=== 应用变号修复 ===\n\n');

    %% 1. 重建 StageManager（新增 sign_changed_out 第5输出）
    fprintf('1. 重建 StageManager...\n');
    create_stage_manager(modelName);
    fprintf('   ✅ StageManager 重建完成（新增端口5: sign_changed_out）\n\n');

    %% 2. 重建 Y0_Latch（新增 sign_changed 第3输入）
    fprintf('2. 重建 Y0_Latch...\n');
    rebuild_y0_latch(modelName);
    fprintf('   ✅ Y0_Latch 重建完成（新增端口3: sign_changed）\n\n');

    save_system(modelName);

    fprintf('========================================\n');
    fprintf('子系统重建完成！\n\n');
    fprintf('⚠️  需要手动重新接线（两个子系统被删除重建，原连线已断开）:\n\n');
    fprintf('【StageManager 接线（恢复原有 + 新增）】\n');
    fprintf('  输入端口（不变）:\n');
    fprintf('    /1 error_value     ← 与之前相同\n');
    fprintf('    /2 should_upgrade  ← 与之前相同\n');
    fprintf('    /3 reset_flag      ← Reset_Flag_Detector/1\n');
    fprintf('    /4 control_enabled ← Decision/control_enabled\n');
    fprintf('  输出端口（恢复原有 + 新增第5）:\n');
    fprintf('    /1 stage_offset_out → SPPVT 控制方程\n');
    fprintf('    /2 stage_out        → (Scope/Display)\n');
    fprintf('    /3 error_sign_out   → (Scope/Display)\n');
    fprintf('    /4 cooldown_out     → (Scope/Display)\n');
    fprintf('    /5 sign_changed_out → Y0_Latch/sign_changed [新增]\n\n');
    fprintf('【Y0_Latch 接线（恢复原有 + 新增第3输入）】\n');
    fprintf('  输入端口:\n');
    fprintf('    /1 control_enabled      ← Decision/control_enabled\n');
    fprintf('    /2 current_engine_torque← Demux #5\n');
    fprintf('    /3 sign_changed         ← StageManager/5 [新增]\n');
    fprintf('  输出端口（不变）:\n');
    fprintf('    /1 Y0 → SPPVT 控制方程\n');
    fprintf('========================================\n');
end


%% ============================================================
%% 内部函数：重建 Y0_Latch（3输入版）
%% ============================================================
function rebuild_y0_latch(modelName)
    subsysPath = [modelName '/Y0_Latch'];

    if getSimulinkBlockHandle(subsysPath) ~= -1
        delete_block(subsysPath);
    end

    add_block('simulink/Ports & Subsystems/Subsystem', subsysPath);
    delete_line(subsysPath, 'In1/1', 'Out1/1');
    delete_block([subsysPath '/In1']);
    delete_block([subsysPath '/Out1']);

    sys = subsysPath;

    %% 输入端口 (3个)
    add_block('simulink/Sources/In1', [sys '/control_enabled'], ...
        'Position', [30, 55, 60, 69], 'Port', '1');
    add_block('simulink/Sources/In1', [sys '/current_engine_torque'], ...
        'Position', [30, 185, 60, 199], 'Port', '2');
    add_block('simulink/Sources/In1', [sys '/sign_changed'], ...
        'Position', [30, 265, 60, 279], 'Port', '3');

    %% 输出端口 (1个)
    add_block('simulink/Sinks/Out1', [sys '/Y0'], ...
        'Position', [600, 142, 630, 158], 'Port', '1');

    %% 上升沿检测: control_enabled(当前) > prev_ce
    add_block('simulink/Discrete/Unit Delay', [sys '/UD_ce'], ...
        'Position', [120, 100, 160, 130], ...
        'InitialCondition', '0', 'SampleTime', '-1');
    add_block('simulink/Logic and Bit Operations/Relational Operator', ...
        [sys '/Rising_Edge'], ...
        'Position', [220, 53, 260, 87], 'Operator', '>');

    %% 变号触发: sign_changed AND control_enabled
    add_block('simulink/Logic and Bit Operations/Logical Operator', ...
        [sys '/AND_sc_ce'], ...
        'Position', [220, 240, 260, 290], ...
        'Operator', 'AND', 'Inputs', '2');

    %% 合并: Rising_Edge OR (sign_changed AND control_enabled)
    add_block('simulink/Logic and Bit Operations/Logical Operator', ...
        [sys '/OR_latch'], ...
        'Position', [310, 60, 350, 110], ...
        'Operator', 'OR', 'Inputs', '2');

    %% Switch + Unit Delay 锁存
    add_block('simulink/Signal Routing/Switch', [sys '/Sw_latch'], ...
        'Position', [420, 120, 460, 170], ...
        'Criteria', 'u2 > Threshold', 'Threshold', '0.5');
    add_block('simulink/Discrete/Unit Delay', [sys '/UD_Y0'], ...
        'Position', [420, 220, 460, 250], ...
        'InitialCondition', '0', 'SampleTime', '-1');

    %% 连线
    add_line(sys, 'control_enabled/1', 'Rising_Edge/1');
    add_line(sys, 'control_enabled/1', 'UD_ce/1');
    add_line(sys, 'UD_ce/1', 'Rising_Edge/2');

    add_line(sys, 'sign_changed/1', 'AND_sc_ce/1');
    add_line(sys, 'control_enabled/1', 'AND_sc_ce/2');

    add_line(sys, 'Rising_Edge/1', 'OR_latch/1');
    add_line(sys, 'AND_sc_ce/1', 'OR_latch/2');

    add_line(sys, 'current_engine_torque/1', 'Sw_latch/1');
    add_line(sys, 'OR_latch/1', 'Sw_latch/2');
    add_line(sys, 'UD_Y0/1', 'Sw_latch/3');

    add_line(sys, 'Sw_latch/1', 'Y0/1');
    add_line(sys, 'Sw_latch/1', 'UD_Y0/1');

    set_param(subsysPath, 'Position', [100, 500, 280, 560]);
end
