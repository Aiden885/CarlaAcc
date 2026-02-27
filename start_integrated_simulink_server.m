%% 启动统一Simulink模型服务器
% 用于acc_integrated_model.slx的UDP通信
%
% UDP输入 (5个double):
%   [command_type, ego_speed_ms, vehicle_distance,
%    target_speed_ms, current_engine_torque]
%
% UDP输出 (2个double):
%   [control_enabled, final_output]
%
% Simulink内部维护:
%   - 状态自循环 (current_state, has_history, last_active_decision) via Unit Delay
%   - G2_Manager (根据command_type T/R键调整G2_s)
%   - Reset_Flag_Detector (control_enabled下降沿 + G2突变)
%   - Y0_Latch (control_enabled上升沿锁存current_engine_torque)
%   - Torque_Arbitration (R7时max(control_output, current_engine_torque))
%   - Low_Speed_Detection (ego_speed < V_min时强制S3)

fprintf('=== 启动统一Simulink模型服务器 ===\n\n');

try
    % 1. 加载模型
    fprintf('1. 加载模型...\n');
    model_name = 'acc_integrated_model';
    load_system(model_name);
    fprintf('   ✅ 模型已加载\n\n');

    % 2. 加载Decision LUT数据到模型工作区
    fprintf('2. 加载Decision LUT数据...\n');
    data_file = 'decision_lookup_data.mat';
    if exist(data_file, 'file')
        lut_data = load(data_file);
        mdlWks = get_param(model_name, 'ModelWorkspace');
        fn = fieldnames(lut_data);
        for i = 1:numel(fn)
            mdlWks.assignin(fn{i}, lut_data.(fn{i}));
        end
        fprintf('   ✅ 已加载 %s 到模型工作区\n\n', data_file);
    else
        fprintf('   ⚠️  未找到 %s，Decision LUT可能不一致\n\n', data_file);
    end

    % 3. 强制LUT使用工作区变量（避免模型内置旧表）
    fprintf('3. 校验并绑定Decision LUT参数...\n');
    decision_path = [model_name '/Decision'];
    lut_defs = {
        'LUT_next_state',     'next_state_table';
        'LUT_decision',       'decision_table';
        'LUT_control_enabled','control_enabled_table';
        'LUT_side_effect',    'side_effect_table';
    };
    for i = 1:size(lut_defs, 1)
        blk = [decision_path '/' lut_defs{i,1}];
        if ~isempty(find_system(model_name, 'SearchDepth', 2, 'Name', lut_defs{i,1}))
            try
                set_param(blk, 'Table', lut_defs{i,2});
                set_param(blk, 'BreakpointsForDimension1', 'state_bp');
                set_param(blk, 'BreakpointsForDimension2', 'command_bp');
                % 避免插值导致状态抖动
                set_param(blk, 'InterpMethod', 'Nearest');
                set_param(blk, 'ExtrapMethod', 'Clip');
                fprintf('   ✅ %s 绑定到 %s\n', lut_defs{i,1}, lut_defs{i,2});
            catch ME
                fprintf('   ⚠️  %s 参数绑定失败: %s\n', lut_defs{i,1}, ME.message);
            end
        end
    end
    fprintf('\n');

    % 4. 配置仿真参数
    fprintf('4. 配置仿真参数...\n');

    % 确保是Fixed-step模式
    set_param(model_name, 'SolverType', 'Fixed-step');
    set_param(model_name, 'Solver', 'FixedStepDiscrete');
    set_param(model_name, 'FixedStep', '0.05');

    % 关闭数据记录（提升性能）
    set_param(model_name, 'SaveOutput', 'off');
    set_param(model_name, 'SaveState', 'off');
    set_param(model_name, 'SaveTime', 'off');

    % 停止时间设为inf（持续运行）
    set_param(model_name, 'StopTime', 'inf');

    fprintf('   ✅ 仿真参数已配置\n');
    fprintf('      - Solver: Fixed-step (0.05s)\n');
    fprintf('      - StopTime: inf (持续运行)\n');
    fprintf('      - 数据记录: OFF\n\n');

    % 5. 检查UDP配置
    fprintf('5. 检查UDP配置...\n');

    % UDP Receive
    udp_recv = [model_name '/UDP Receive'];
    if ~isempty(find_system(model_name, 'Name', 'UDP Receive'))
        local_port = get_param(udp_recv, 'LocalPort');
        data_size = get_param(udp_recv, 'DataSize');
        fprintf('   UDP Receive:\n');
        fprintf('      - LocalPort: %s (Python发送目标)\n', local_port);
        fprintf('      - DataSize: %s\n', data_size);
        if ~strcmp(data_size, '[5 1]')
            fprintf('   ⚠️  DataSize 应为 [5 1] (5个double输入)，当前为 %s\n', data_size);
        end
    end
    % UDP Send
    udp_send = [model_name '/UDP Send'];
    if ~isempty(find_system(model_name, 'Name', 'UDP Send'))
        remote_port = get_param(udp_send, 'Port');
        fprintf('   UDP Send:\n');
        fprintf('      - RemotePort: %s (Python接收端口)\n', remote_port);
    end
    fprintf('   ✅ UDP配置正常\n\n');

    % 6. 启动仿真
    fprintf('6. 启动仿真...\n');
    set_param(model_name, 'SimulationCommand', 'start');
    fprintf('   ✅ 仿真已启动\n\n');

    fprintf('=== Simulink服务器就绪 ===\n');
    fprintf('\n');
    fprintf('UDP服务器监听中...\n');
    fprintf('   Python发送 → 端口27000 (5个double)\n');
    fprintf('   Python接收 ← 端口27001 (2个double)\n');
    fprintf('\n');
    fprintf('现在可以运行Python测试脚本:\n');
    fprintf('   python test_integrated_model_decision.py\n');
    fprintf('\n');
    fprintf('⚠️ 停止服务器: 在MATLAB命令窗口输入\n');
    fprintf('   set_param(''%s'', ''SimulationCommand'', ''stop'')\n', model_name);
    fprintf('\n');

catch ME
    fprintf('❌ 启动失败: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('   位置: %s (行 %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\n请检查:\n');
    fprintf('  1. 模型文件是否存在: %s.slx\n', model_name);
    fprintf('  2. 是否有其他程序占用端口27000或27001\n');
    fprintf('  3. 模型是否已正确配置UDP模块\n');
end
