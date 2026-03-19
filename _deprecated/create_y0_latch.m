%% create_y0_latch.m
% 生成 Y0_Latch 子系统
%
% 功能：在 ACC 启控瞬间（control_enabled 上升沿）锁存当前发动机扭矩作为 Y0
%
% 输入：
%   1. control_enabled      - ACC 控制使能信号 (0/1)
%   2. current_engine_torque - 当前油门对应的发动机扭矩 (N·m，实时值)
%
% 输出：
%   1. Y0 - 启控时刻锁存的扭矩值 (N·m)
%
% 逻辑：
%   每一步检测 control_enabled 是否出现上升沿 (0→1)
%   - 上升沿：Y0 = current_engine_torque（锁存新值）
%   - 其他时刻：Y0 保持上次锁存的值不变
%
% 实现原理（纯基础模块，无 MATLAB Function）：
%   prev_ce = Memory(control_enabled)
%   rising_edge = (control_enabled > prev_ce)   % 0→1 时为 true
%   Y0 = Switch(rising_edge ? current_engine_torque : Memory(Y0))
%
% 用法：
%   create_y0_latch()
%   create_y0_latch('acc_integrated_model')
%
% Simulink 2024b

function create_y0_latch(modelName)
    if nargin < 1
        modelName = 'acc_integrated_model';
    end

    if ~bdIsLoaded(modelName)
        load_system(modelName);
    end

    subsysName = 'Y0_Latch';
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

    %% ==================== 创建所有块 ====================

    %% 输入端口 (2个)
    add_block('simulink/Sources/In1', [subsysPath '/control_enabled'], ...
        'Position', [30, 55, 60, 69], 'Port', '1');
    add_block('simulink/Sources/In1', [subsysPath '/current_engine_torque'], ...
        'Position', [30, 185, 60, 199], 'Port', '2');

    %% 输出端口 (1个)
    add_block('simulink/Sinks/Out1', [subsysPath '/Y0'], ...
        'Position', [550, 142, 580, 158], 'Port', '1');

    %% --- 上升沿检测 ---

    % Memory 块：存储上一拍的 control_enabled
    add_block('simulink/Discrete/Memory', [subsysPath '/Mem_ce'], ...
        'Position', [120, 100, 160, 130], ...
        'InitialCondition', '0');

    % 关系运算：control_enabled > prev_ce  →  上升沿为 true
    add_block('simulink/Logic and Bit Operations/Relational Operator', ...
        [subsysPath '/Rising_Edge'], ...
        'Position', [220, 53, 260, 87], ...
        'Operator', '>');

    %% --- Y0 锁存 ---

    % Switch 块：上升沿时取 current_engine_torque，否则取 Mem_Y0
    %   Port 1 (上)：true 值 → current_engine_torque
    %   Port 2 (中)：条件   → rising_edge
    %   Port 3 (下)：false值 → Mem_Y0 输出
    add_block('simulink/Signal Routing/Switch', [subsysPath '/Sw_latch'], ...
        'Position', [380, 120, 420, 170], ...
        'Criteria', 'u2 > Threshold', ...
        'Threshold', '0.5');

    % Memory 块：存储锁存的 Y0 值（反馈回 Switch 的 false 端口）
    add_block('simulink/Discrete/Memory', [subsysPath '/Mem_Y0'], ...
        'Position', [380, 220, 420, 250], ...
        'InitialCondition', '0');

    %% ==================== 连接信号线 ====================

    % --- 上升沿检测 ---
    % control_enabled → Rising_Edge 端口1 (当前值)
    add_line(subsysPath, 'control_enabled/1', 'Rising_Edge/1');
    % control_enabled → Mem_ce (存储上一拍)
    add_line(subsysPath, 'control_enabled/1', 'Mem_ce/1');
    % Mem_ce → Rising_Edge 端口2 (上一拍值)
    add_line(subsysPath, 'Mem_ce/1', 'Rising_Edge/2');

    % --- Switch 连线 ---
    % current_engine_torque → Switch 端口1 (上升沿时的值)
    add_line(subsysPath, 'current_engine_torque/1', 'Sw_latch/1');
    % Rising_Edge → Switch 端口2 (条件)
    add_line(subsysPath, 'Rising_Edge/1', 'Sw_latch/2');
    % Mem_Y0 → Switch 端口3 (非上升沿时保持的值)
    add_line(subsysPath, 'Mem_Y0/1', 'Sw_latch/3');

    % --- 输出 + 反馈 ---
    % Switch → Y0 输出端口
    add_line(subsysPath, 'Sw_latch/1', 'Y0/1');
    % Switch → Mem_Y0 (反馈：下一拍记住当前 Y0)
    add_line(subsysPath, 'Sw_latch/1', 'Mem_Y0/1');

    %% ========== 设置子系统外观 ==========
    set_param(subsysPath, 'Position', [200, 500, 380, 560]);

    save_system(modelName);

    fprintf('\n========================================\n');
    fprintf('Y0_Latch 子系统已创建成功！\n');
    fprintf('路径：%s\n', subsysPath);
    fprintf('========================================\n');
    fprintf('输入端口：\n');
    fprintf('  1. control_enabled        ← Demux #3 (control_enabled 输出)\n');
    fprintf('  2. current_engine_torque  ← Demux #9 (Python 传入的实时油门扭矩)\n');
    fprintf('输出端口：\n');
    fprintf('  1. Y0 → 连接到 SPPVT 的 Y0 输入线\n');
    fprintf('========================================\n');
    fprintf('工作原理：\n');
    fprintf('  - control_enabled 从 0→1 时，锁存当前扭矩作为 Y0\n');
    fprintf('  - 之后 Y0 保持不变，直到下次重新启控\n');
    fprintf('========================================\n');
end
