function create_acc_decision_matlab_function()
%% 创建基于MATLAB Function的ACC决策Simulink模型框架
% 第一部分：只创建模型结构，不包含具体函数代码

    model_name = 'acc_decision_matlab_function';

    fprintf('=== 创建基于MATLAB Function的ACC决策模型框架 ===\n');

    % 清理已存在的模型
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
    if exist([model_name '.slx'], 'file')
        delete([model_name '.slx']);
    end

    % 创建新模型
    new_system(model_name);
    open_system(model_name);

    % 设置模型参数
    set_param(model_name, 'StopTime', 'inf');
    set_param(model_name, 'FixedStep', '0.05');
    set_param(model_name, 'SolverType', 'Fixed-step');
    set_param(model_name, 'Solver', 'FixedStepDiscrete');

    % 创建输入端口
    create_input_ports(model_name);

    % 创建MATLAB Function块（不设置代码）
    create_empty_matlab_function_block(model_name);

    % 创建状态反馈延迟块
    create_state_feedback_blocks(model_name);

    % 创建输出端口
    create_output_ports(model_name);

    % 等待MATLAB Function块完全创建后再连接
    pause(1);

    % 连接信号线
    connect_all_signals_fixed(model_name);

    % 保存模型
    save_system(model_name);

    fprintf('模型框架创建完成: %s.slx\n', model_name);
    fprintf('请运行 get_matlab_function_code() 获取函数代码并手动粘贴\n');
end

function create_input_ports(model_name)
    fprintf('创建输入端口...\n');

    % 输入信号定义
    inputs = {
        {'command_input', [50, 80]},
        {'ego_speed_kmh', [50, 120]},
        {'has_target', [50, 160]},
        {'current_distance', [50, 200]},
        {'reset_signal', [50, 240]}
    };

    for i = 1:length(inputs)
        name = inputs{i}{1};
        pos = inputs{i}{2};

        inport = add_block('simulink/Sources/In1', [model_name '/' name]);
        set_param(inport, 'Position', [pos(1) pos(2) pos(1)+80 pos(2)+16]);
        set_param(inport, 'Port', num2str(i));
    end

    fprintf('输入端口创建完成\n');
end

function create_empty_matlab_function_block(model_name)
    fprintf('创建MATLAB Function块...\n');

    % 添加MATLAB Function块
    matlab_func_block = add_block('simulink/User-Defined Functions/MATLAB Function', ...
                                  [model_name '/ACC_Decision_Function']);
    set_param(matlab_func_block, 'Position', [300 150 500 350]);

    fprintf('MATLAB Function块已创建\n');
end

function create_state_feedback_blocks(model_name)
    fprintf('创建状态反馈延迟块...\n');

    % 状态反馈变量及其初始值
    feedback_vars = {
        {'prev_state', [650, 80], '2'},
        {'prev_V3', [650, 110], '50.0'},
        {'prev_G1', [650, 140], '15.0'},
        {'prev_G2', [650, 170], '2.0'},
        {'prev_has_history', [650, 200], 'false'},
        {'prev_history_V3', [650, 230], '50.0'},
        {'prev_history_G1', [650, 260], '15.0'},
        {'prev_history_G2', [650, 290], '2.0'},
        {'prev_pending_adj', [650, 320], '0'}
    };

    for i = 1:length(feedback_vars)
        name = feedback_vars{i}{1};
        pos = feedback_vars{i}{2};
        init_val = feedback_vars{i}{3};

        delay_block = add_block('simulink/Discrete/Unit Delay', [model_name '/' name]);
        set_param(delay_block, 'Position', [pos(1) pos(2) pos(1)+60 pos(2)+20]);
        set_param(delay_block, 'InitialCondition', init_val);
    end

    fprintf('状态反馈块创建完成\n');
end

function create_output_ports(model_name)
    fprintf('创建输出端口...\n');

    % 输出信号定义
    outputs = {
        {'current_state', [800, 80]},
        {'control_mode', [800, 110]},
        {'acc_active', [800, 140]},
        {'control_enabled', [800, 170]},
        {'V3_kmh', [800, 200]},
        {'G1_m', [800, 230]},
        {'G2_s', [800, 260]},
        {'has_history', [800, 290]},
        {'message_code', [800, 320]},
        {'pending_distance_adj', [800, 350]},
        {'is_in_control', [800, 380]}
    };

    for i = 1:length(outputs)
        name = outputs{i}{1};
        pos = outputs{i}{2};

        outport = add_block('simulink/Sinks/Out1', [model_name '/' name]);
        set_param(outport, 'Position', [pos(1) pos(2) pos(1)+80 pos(2)+16]);
        set_param(outport, 'Port', num2str(i));
    end

    fprintf('输出端口创建完成\n');
end

function connect_all_signals_fixed(model_name)
    fprintf('连接信号线...\n');

    matlab_func_block = [model_name '/ACC_Decision_Function'];

    % 获取MATLAB Function块的端口句柄
    try
        func_ports = get_param(matlab_func_block, 'PortHandles');
    catch
        fprintf('无法获取MATLAB Function块端口，跳过连接\n');
        fprintf('请手动连接信号线，或稍后运行 connect_signals_manual\n');
        return;
    end

    % 连接输入信号
    input_names = {'command_input', 'ego_speed_kmh', 'has_target', 'current_distance', 'reset_signal'};
    for i = 1:length(input_names)
        try
            src_ports = get_param([model_name '/' input_names{i}], 'PortHandles');
            if i <= length(func_ports.Inport)
                add_line(model_name, src_ports.Outport(1), func_ports.Inport(i), 'autorouting', 'on');
                fprintf('  输入连接成功: %s\n', input_names{i});
            end
        catch ME
            fprintf('  输入连接失败: %s - %s\n', input_names{i}, ME.message);
        end
    end

    % 连接状态反馈信号
    feedback_names = {'prev_state', 'prev_V3', 'prev_G1', 'prev_G2', 'prev_has_history', ...
                     'prev_history_V3', 'prev_history_G1', 'prev_history_G2', 'prev_pending_adj'};
    for i = 1:length(feedback_names)
        try
            src_ports = get_param([model_name '/' feedback_names{i}], 'PortHandles');
            port_idx = length(input_names) + i;
            if port_idx <= length(func_ports.Inport)
                add_line(model_name, src_ports.Outport(1), func_ports.Inport(port_idx), 'autorouting', 'on');
                fprintf('  反馈连接成功: %s\n', feedback_names{i});
            end
        catch ME
            fprintf('  反馈连接失败: %s - %s\n', feedback_names{i}, ME.message);
        end
    end

    % 连接输出信号
    output_names = {'current_state', 'control_mode', 'acc_active', 'control_enabled', ...
                   'V3_kmh', 'G1_m', 'G2_s', 'has_history', 'message_code', ...
                   'pending_distance_adj', 'is_in_control'};
    for i = 1:length(output_names)
        try
            dst_ports = get_param([model_name '/' output_names{i}], 'PortHandles');
            if i <= length(func_ports.Outport)
                add_line(model_name, func_ports.Outport(i), dst_ports.Inport(1), 'autorouting', 'on');
                fprintf('  输出连接成功: %s\n', output_names{i});
            end
        catch ME
            fprintf('  输出连接失败: %s - %s\n', output_names{i}, ME.message);
        end
    end

    % 连接状态反馈回路
    feedback_connections = {
        {1, 1}, {5, 2}, {6, 3}, {7, 4}, {8, 5}, {5, 6}, {6, 7}, {7, 8}, {10, 9}
    };

    feedback_output_names = {'current_state', 'V3_kmh', 'G1_m', 'G2_s', 'has_history', ...
                            'V3_kmh', 'G1_m', 'G2_s', 'pending_distance_adj'};

    for i = 1:length(feedback_connections)
        try
            out_idx = feedback_connections{i}{1};
            fb_idx = feedback_connections{i}{2};

            dst_ports = get_param([model_name '/' feedback_names{fb_idx}], 'PortHandles');
            if out_idx <= length(func_ports.Outport)
                add_line(model_name, func_ports.Outport(out_idx), dst_ports.Inport(1), 'autorouting', 'on');
                fprintf('  反馈回路成功: %s到%s\n', feedback_output_names{i}, feedback_names{fb_idx});
            end
        catch ME
            fprintf('  反馈回路失败: %s\n', ME.message);
        end
    end

    fprintf('信号连接完成\n');
end

% 手动连接函数（备用）
function connect_signals_manual(model_name)
    fprintf('=== 手动信号连接 ===\n');
    if nargin < 1
        model_name = 'acc_decision_matlab_function';
    end

    if ~bdIsLoaded(model_name)
        fprintf('模型未加载，请先运行 create_acc_decision_matlab_function()\n');
        return;
    end

    connect_all_signals_fixed(model_name);
end

% 简单测试函数
function test_matlab_function_model()
    fprintf('\n=== 测试MATLAB Function模型 ===\n');

    model_name = 'acc_decision_matlab_function';

    if ~bdIsLoaded(model_name)
        fprintf('请先运行 create_acc_decision_matlab_function() 创建模型\n');
        return;
    end

    fprintf('测试1: 检查模型结构\n');
    try
        % 检查输入输出
        input_blocks = find_system(model_name, 'BlockType', 'Inport');
        output_blocks = find_system(model_name, 'BlockType', 'Outport');
        func_blocks = find_system(model_name, 'BlockType', 'MATLABFcn');

        fprintf('  输入端口数量: %d\n', length(input_blocks));
        fprintf('  输出端口数量: %d\n', length(output_blocks));
        fprintf('  MATLAB Function块数量: %d\n', length(func_blocks));

        if length(input_blocks) == 5 && length(output_blocks) == 11
            fprintf('  模型结构正确\n');
        else
            fprintf('  模型结构可能有问题\n');
        end

    catch ME
        fprintf('  测试失败: %s\n', ME.message);
    end

    fprintf('模型框架测试完成\n');
    fprintf('下一步：运行 get_matlab_function_code() 获取函数代码\n');
end