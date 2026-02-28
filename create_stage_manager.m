%% create_stage_manager.m
% 生成 StageManager 子系统（纯 Simulink 模块，无 MATLAB Function）
% 用法：
%   1. 在 MATLAB 命令窗口运行：create_stage_manager('acc_integrated_model')
%   2. 或者直接运行：create_stage_manager() （默认模型名）
%   3. StageManager 子系统将被添加到模型中
%
% Simulink 2024b

function create_stage_manager(modelName)
    if nargin < 1
        modelName = 'acc_integrated_model';
    end

    % 确保模型已打开
    if ~bdIsLoaded(modelName)
        load_system(modelName);
    end

    % 子系统路径
    subsysName = 'StageManager';
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
    rho = 0.1;
    upgrade_cooldown_frames = 2;
    offset_max = 500;
    offset_min = -500;
    eps = 1e-6;

    %% ==================== 第1部分：创建所有块 ====================

    %% 输入端口
    add_block('simulink/Sources/In1', [subsysPath '/error_value'], ...
        'Position', [30, 100, 60, 114], 'Port', '1');
    add_block('simulink/Sources/In1', [subsysPath '/should_upgrade'], ...
        'Position', [30, 200, 60, 214], 'Port', '2');
    add_block('simulink/Sources/In1', [subsysPath '/reset_flag'], ...
        'Position', [30, 300, 60, 314], 'Port', '3');
    add_block('simulink/Sources/In1', [subsysPath '/control_enabled'], ...
        'Position', [30, 400, 60, 414], 'Port', '4');

    %% 输出端口
    add_block('simulink/Sinks/Out1', [subsysPath '/stage_offset_out'], ...
        'Position', [1000, 247, 1030, 263], 'Port', '1');
    add_block('simulink/Sinks/Out1', [subsysPath '/stage_out'], ...
        'Position', [1000, 307, 1030, 323], 'Port', '2');
    add_block('simulink/Sinks/Out1', [subsysPath '/error_sign_out'], ...
        'Position', [1000, 367, 1030, 383], 'Port', '3');
    add_block('simulink/Sinks/Out1', [subsysPath '/cooldown_out'], ...
        'Position', [1000, 427, 1030, 443], 'Port', '4');
    add_block('simulink/Sinks/Out1', [subsysPath '/sign_changed_out'], ...
        'Position', [1000, 487, 1030, 503], 'Port', '5');

    %% 常量块
    add_block('simulink/Sources/Constant', [subsysPath '/Const_rho'], ...
        'Position', [100, 540, 130, 560], 'Value', num2str(rho));
    add_block('simulink/Sources/Constant', [subsysPath '/Const_cooldown_init'], ...
        'Position', [500, 600, 530, 620], 'Value', num2str(upgrade_cooldown_frames));
    add_block('simulink/Sources/Constant', [subsysPath '/Const_zero'], ...
        'Position', [100, 600, 130, 620], 'Value', '0');
    add_block('simulink/Sources/Constant', [subsysPath '/Const_one'], ...
        'Position', [300, 620, 330, 640], 'Value', '1');
    add_block('simulink/Sources/Constant', [subsysPath '/Const_eps'], ...
        'Position', [360, 620, 390, 640], 'Value', num2str(eps));

    %% Reset 条件：reset_flag OR NOT(control_enabled)
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subsysPath '/NOT_enabled'], ...
        'Position', [120, 395, 150, 415], 'Operator', 'NOT');
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subsysPath '/OR_reset'], ...
        'Position', [200, 340, 240, 380], 'Operator', 'OR', 'Inputs', '2');

    %% 计算 current_sign（带 deadzone）
    add_block('simulink/Math Operations/Sign', [subsysPath '/Sign_error'], ...
        'Position', [120, 95, 150, 125]);
    add_block('simulink/Math Operations/Abs', [subsysPath '/Abs_error'], ...
        'Position', [120, 140, 150, 170]);
    add_block('simulink/Logic and Bit Operations/Relational Operator', [subsysPath '/Abs_lt_eps'], ...
        'Position', [200, 140, 240, 170], 'Operator', '<=');
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_sign_deadzone'], ...
        'Position', [280, 95, 320, 165], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');

    %% 状态：error_sign (Unit Delay)
    add_block('simulink/Discrete/Unit Delay', [subsysPath '/UnitDelay_error_sign'], ...
        'Position', [450, 37, 490, 73], ...
        'InitialCondition', '0', ...
        'SampleTime', '-1');

    %% 符号翻转检测
    % prev_sign ~= 0
    add_block('simulink/Logic and Bit Operations/Compare To Zero', [subsysPath '/PrevSign_ne_0'], ...
        'Position', [520, 40, 560, 70], 'relop', '~=');

    % current_sign ~= 0
    add_block('simulink/Logic and Bit Operations/Compare To Zero', [subsysPath '/CurrSign_ne_0'], ...
        'Position', [200, 95, 240, 125], 'relop', '~=');

    % prev_sign ~= current_sign
    add_block('simulink/Logic and Bit Operations/Relational Operator', [subsysPath '/Sign_ne'], ...
        'Position', [520, 90, 560, 120], 'Operator', '~=');

    % AND 三个条件
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subsysPath '/AND_sign_changed'], ...
        'Position', [620, 60, 660, 120], 'Operator', 'AND', 'Inputs', '3');

    %% 清零条件：OR_clear = reset OR sign_changed
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subsysPath '/OR_clear'], ...
        'Position', [700, 80, 740, 120], 'Operator', 'OR', 'Inputs', '2');

    %% 状态：cooldown (Unit Delay)
    add_block('simulink/Discrete/Unit Delay', [subsysPath '/UnitDelay_cooldown'], ...
        'Position', [450, 527, 490, 563], ...
        'InitialCondition', '0', ...
        'SampleTime', '-1');

    %% cooldown == 0 检测
    add_block('simulink/Logic and Bit Operations/Compare To Zero', [subsysPath '/Cooldown_eq_0'], ...
        'Position', [520, 530, 560, 560], 'relop', '==');

    %% 升级条件：should_upgrade AND cooldown==0 AND NOT(reset) AND NOT(sign_changed)
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subsysPath '/NOT_reset'], ...
        'Position', [300, 345, 330, 375], 'Operator', 'NOT');
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subsysPath '/NOT_sign_changed'], ...
        'Position', [700, 140, 730, 170], 'Operator', 'NOT');

    add_block('simulink/Logic and Bit Operations/Logical Operator', [subsysPath '/AND_upgrade'], ...
        'Position', [780, 190, 820, 270], 'Operator', 'AND', 'Inputs', '4');

    %% 状态：stage_offset (Unit Delay)
    add_block('simulink/Discrete/Unit Delay', [subsysPath '/UnitDelay_stage_offset'], ...
        'Position', [450, 237, 490, 273], ...
        'InitialCondition', '0', ...
        'SampleTime', '-1');
    %% 状态：stage (Unit Delay)
    add_block('simulink/Discrete/Unit Delay', [subsysPath '/UnitDelay_stage'], ...
        'Position', [450, 300, 490, 336], ...
        'InitialCondition', '1', ...
        'SampleTime', '-1');

    %% 计算升级时的 offset 增量：delta = rho * error
    add_block('simulink/Math Operations/Product', [subsysPath '/Prod_delta'], ...
        'Position', [200, 530, 240, 570], 'Inputs', '2');

    %% 新 offset = prev_offset + delta
    add_block('simulink/Math Operations/Add', [subsysPath '/Add_offset'], ...
        'Position', [300, 240, 340, 280]);

    %% 限幅
    add_block('simulink/Discontinuities/Saturation', [subsysPath '/Saturation_offset'], ...
        'Position', [380, 245, 420, 275], ...
        'UpperLimit', num2str(offset_max), ...
        'LowerLimit', num2str(offset_min));

    %% Switch: 升级时选新值，否则保持旧值
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_offset_upgrade'], ...
        'Position', [540, 220, 580, 290], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');

    %% Switch: reset/sign_changed 时归零
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_offset_reset'], ...
        'Position', [850, 220, 890, 280], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');

    %% error_sign 更新逻辑
    % if current_sign != 0: error_sign = current_sign, else: keep prev
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_error_sign_update'], ...
        'Position', [280, 30, 320, 90], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');

    % reset 时归零
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_error_sign_reset'], ...
        'Position', [780, 20, 820, 80], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');

    %% stage 更新逻辑
    add_block('simulink/Math Operations/Add', [subsysPath '/Add_stage'], ...
        'Position', [300, 310, 340, 340], 'Inputs', '++');
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_stage_upgrade'], ...
        'Position', [540, 300, 580, 360], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_stage_reset'], ...
        'Position', [850, 300, 890, 360], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');
    %% cooldown 更新逻辑
    % cooldown - 1
    add_block('simulink/Math Operations/Add', [subsysPath '/Sub_cooldown'], ...
        'Position', [380, 570, 420, 610], 'Inputs', '+-');

    % max(0, cooldown-1)
    add_block('simulink/Math Operations/MinMax', [subsysPath '/Max_cooldown_0'], ...
        'Position', [460, 575, 500, 605], 'Function', 'max', 'Inputs', '2');

    % 升级时用 cooldown_init，否则用递减值
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_cooldown_upgrade'], ...
        'Position', [600, 550, 640, 620], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');

    % reset 时归零
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Switch_cooldown_reset'], ...
        'Position', [850, 510, 890, 570], ...
        'Criteria', 'u2 ~= 0', ...
        'Threshold', '0');

    %% ==================== 第2部分：连接所有信号线 ====================

    % Reset 条件连接
    add_line(subsysPath, 'control_enabled/1', 'NOT_enabled/1');
    add_line(subsysPath, 'reset_flag/1', 'OR_reset/1');
    add_line(subsysPath, 'NOT_enabled/1', 'OR_reset/2');

    % Sign 计算连接
    add_line(subsysPath, 'error_value/1', 'Sign_error/1');
    add_line(subsysPath, 'error_value/1', 'Abs_error/1');
    add_line(subsysPath, 'Abs_error/1', 'Abs_lt_eps/1');
    add_line(subsysPath, 'Const_eps/1', 'Abs_lt_eps/2');
    add_line(subsysPath, 'Const_zero/1', 'Switch_sign_deadzone/1');
    add_line(subsysPath, 'Abs_lt_eps/1', 'Switch_sign_deadzone/2');
    add_line(subsysPath, 'Sign_error/1', 'Switch_sign_deadzone/3');

    % 符号翻转检测连接
    add_line(subsysPath, 'Switch_sign_deadzone/1', 'CurrSign_ne_0/1');
    add_line(subsysPath, 'UnitDelay_error_sign/1', 'PrevSign_ne_0/1');
    add_line(subsysPath, 'UnitDelay_error_sign/1', 'Sign_ne/1');
    add_line(subsysPath, 'Switch_sign_deadzone/1', 'Sign_ne/2');
    add_line(subsysPath, 'PrevSign_ne_0/1', 'AND_sign_changed/1');
    add_line(subsysPath, 'CurrSign_ne_0/1', 'AND_sign_changed/2');
    add_line(subsysPath, 'Sign_ne/1', 'AND_sign_changed/3');

    % 清零条件连接
    add_line(subsysPath, 'OR_reset/1', 'OR_clear/1');
    add_line(subsysPath, 'AND_sign_changed/1', 'OR_clear/2');

    % 升级条件连接
    add_line(subsysPath, 'OR_reset/1', 'NOT_reset/1');
    add_line(subsysPath, 'AND_sign_changed/1', 'NOT_sign_changed/1');
    add_line(subsysPath, 'should_upgrade/1', 'AND_upgrade/1');
    add_line(subsysPath, 'NOT_sign_changed/1', 'AND_upgrade/2');
    add_line(subsysPath, 'NOT_reset/1', 'AND_upgrade/3');
    add_line(subsysPath, 'UnitDelay_cooldown/1', 'Cooldown_eq_0/1');
    add_line(subsysPath, 'Cooldown_eq_0/1', 'AND_upgrade/4');

    % offset 计算连接
    add_line(subsysPath, 'error_value/1', 'Prod_delta/1');
    add_line(subsysPath, 'Const_rho/1', 'Prod_delta/2');
    add_line(subsysPath, 'UnitDelay_stage_offset/1', 'Add_offset/1');
    add_line(subsysPath, 'Prod_delta/1', 'Add_offset/2');
    add_line(subsysPath, 'Add_offset/1', 'Saturation_offset/1');

    % offset 选择逻辑连接
    add_line(subsysPath, 'Saturation_offset/1', 'Switch_offset_upgrade/1');
    add_line(subsysPath, 'AND_upgrade/1', 'Switch_offset_upgrade/2');
    add_line(subsysPath, 'UnitDelay_stage_offset/1', 'Switch_offset_upgrade/3');

    add_line(subsysPath, 'Const_zero/1', 'Switch_offset_reset/1');
    add_line(subsysPath, 'OR_clear/1', 'Switch_offset_reset/2');
    add_line(subsysPath, 'Switch_offset_upgrade/1', 'Switch_offset_reset/3');

    % offset 反馈和输出
    add_line(subsysPath, 'Switch_offset_reset/1', 'UnitDelay_stage_offset/1');
    add_line(subsysPath, 'Switch_offset_reset/1', 'stage_offset_out/1');

    % error_sign 更新连接
    add_line(subsysPath, 'Switch_sign_deadzone/1', 'Switch_error_sign_update/1');
    add_line(subsysPath, 'CurrSign_ne_0/1', 'Switch_error_sign_update/2');
    add_line(subsysPath, 'UnitDelay_error_sign/1', 'Switch_error_sign_update/3');

    add_line(subsysPath, 'Const_zero/1', 'Switch_error_sign_reset/1');
    add_line(subsysPath, 'OR_clear/1', 'Switch_error_sign_reset/2');
    add_line(subsysPath, 'Switch_error_sign_update/1', 'Switch_error_sign_reset/3');

    add_line(subsysPath, 'Switch_error_sign_reset/1', 'UnitDelay_error_sign/1');
    add_line(subsysPath, 'Switch_error_sign_reset/1', 'error_sign_out/1');

    % cooldown 更新连接
    add_line(subsysPath, 'UnitDelay_cooldown/1', 'Sub_cooldown/1');
    add_line(subsysPath, 'Const_one/1', 'Sub_cooldown/2');
    add_line(subsysPath, 'Sub_cooldown/1', 'Max_cooldown_0/1');
    add_line(subsysPath, 'Const_zero/1', 'Max_cooldown_0/2');

    add_line(subsysPath, 'Const_cooldown_init/1', 'Switch_cooldown_upgrade/1');
    add_line(subsysPath, 'AND_upgrade/1', 'Switch_cooldown_upgrade/2');
    add_line(subsysPath, 'Max_cooldown_0/1', 'Switch_cooldown_upgrade/3');

    add_line(subsysPath, 'Const_zero/1', 'Switch_cooldown_reset/1');
    add_line(subsysPath, 'OR_clear/1', 'Switch_cooldown_reset/2');
    add_line(subsysPath, 'Switch_cooldown_upgrade/1', 'Switch_cooldown_reset/3');

    add_line(subsysPath, 'Switch_cooldown_reset/1', 'UnitDelay_cooldown/1');
    add_line(subsysPath, 'Switch_cooldown_reset/1', 'cooldown_out/1');

    % stage 更新连接
    add_line(subsysPath, 'UnitDelay_stage/1', 'Add_stage/1');
    add_line(subsysPath, 'Const_one/1', 'Add_stage/2');

    add_line(subsysPath, 'Add_stage/1', 'Switch_stage_upgrade/1');
    add_line(subsysPath, 'AND_upgrade/1', 'Switch_stage_upgrade/2');
    add_line(subsysPath, 'UnitDelay_stage/1', 'Switch_stage_upgrade/3');

    add_line(subsysPath, 'Const_one/1', 'Switch_stage_reset/1');
    add_line(subsysPath, 'OR_clear/1', 'Switch_stage_reset/2');
    add_line(subsysPath, 'Switch_stage_upgrade/1', 'Switch_stage_reset/3');

    add_line(subsysPath, 'Switch_stage_reset/1', 'UnitDelay_stage/1');
    add_line(subsysPath, 'Switch_stage_reset/1', 'stage_out/1');

    % sign_changed 输出
    add_line(subsysPath, 'AND_sign_changed/1', 'sign_changed_out/1');

    %% ========== 设置子系统外观 ==========
    set_param(subsysPath, 'Position', [400, 200, 550, 320]);

    % 保存模型
    save_system(modelName);

    fprintf('\n========================================\n');
    fprintf('StageManager 子系统已创建成功！\n');
    fprintf('路径：%s\n', subsysPath);
    fprintf('========================================\n');
    fprintf('输入端口：\n');
    fprintf('  1. error_value      - 误差值\n');
    fprintf('  2. should_upgrade   - 升级条件\n');
    fprintf('  3. reset_flag       - 重置标志\n');
    fprintf('  4. control_enabled  - 控制使能\n');
    fprintf('输出端口：\n');
    fprintf('  1. stage_offset     - 阶段偏移\n');
    fprintf('  2. stage            - 当前阶段\n');
    fprintf('  3. error_sign       - 误差符号\n');
    fprintf('  4. cooldown         - 冷却计数\n');
    fprintf('========================================\n');
    fprintf('参数设置：\n');
    fprintf('  rho = %.2f\n', rho);
    fprintf('  cooldown_frames = %d\n', upgrade_cooldown_frames);
    fprintf('  offset_max = %d\n', offset_max);
    fprintf('  offset_min = %d\n', offset_min);
    fprintf('  eps = %.1e\n', eps);
    fprintf('========================================\n');
end
