function create_decision_md_simulink()
%% 根据decision.md创建完整的ACC决策Simulink模型
% 完全基于decision.md的4状态7指令逻辑
% 与acc_decision.py具有相同的输入输出接口

    model_name = 'acc_decision_md';

    fprintf('=== 创建基于decision.md的ACC决策模型 ===\n');

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

    % 创建输入输出端口（与acc_decision.py接口一致）
    create_input_output_ports(model_name);

    % 添加Stateflow Chart
    chart_block = add_block('sflib/Chart', [model_name '/ACC_Decision_Chart']);
    set_param(chart_block, 'Position', [300 150 600 450]);

    % 配置Stateflow Chart数据接口
    configure_decision_md_chart(model_name);

    % 连接信号线
    connect_signals(model_name);

    % 添加子系统用于参数初始化
    create_parameter_subsystem(model_name);

    % 配置实际的状态机逻辑（调用单独的配置脚本）
    fprintf('配置decision.md状态机逻辑...\n');
    try
        configure_decision_md_stateflow();
        fprintf('✅ decision.md状态机逻辑配置完成\n');
    catch ME
        fprintf('⚠️ 状态机逻辑配置失败: %s\n', ME.message);
        fprintf('  可以稍后手动运行 configure_decision_md_stateflow.m\n');
    end

    % 保存模型
    save_system(model_name);

    fprintf('✅ 基于decision.md的ACC决策模型创建完成: %s.slx\n', model_name);
    fprintf('📋 模型包含decision.md定义的4个状态和7种指令逻辑\n');
    fprintf('🔧 请运行 test_decision_md_simulink.m 进行测试\n');

end

function create_input_output_ports(model_name)
    %% 创建与acc_decision.py兼容的输入输出端口

    fprintf('创建输入输出端口...\n');

    % 输入信号（与Python版本process_command函数参数对应）
    inputs = struct(...
        'command_input', [50, 100], ...        % 0-6: I0-I6指令编号
        'ego_speed_kmh', [50, 140], ...        % 自车速度 (km/h)
        'has_target', [50, 180], ...           % 是否有前车 (boolean)
        'current_distance', [50, 220], ...     % 当前前车距离 (m)，-1表示无前车
        'reset_signal', [50, 260] ...          % 系统重置信号
    );

    % 输出信号（与Python版本get_decision_output函数返回值对应）
    outputs = struct(...
        'current_state', [800, 80], ...        % 当前状态编号 (0-3对应S0-S3)
        'control_mode', [800, 110], ...        % 控制模式编号 (1-8对应R1-R8)
        'acc_active', [800, 140], ...          % ACC是否激活
        'control_enabled', [800, 170], ...     % 控制是否使能
        'V3_kmh', [800, 200], ...             % 目标速度参数
        'G1_m', [800, 230], ...               % 距离参数
        'G2_s', [800, 260], ...               % 时间间隔参数
        'has_history', [800, 290], ...         % 是否有历史数据
        'message_code', [800, 320], ...        % 执行结果消息编码
        'pending_distance_adj', [800, 350], ... % 待存储距离调整
        'is_in_control', [800, 380] ...        % 是否处于主动控制模式
    );

    % 创建输入端口
    input_names = fieldnames(inputs);
    for i = 1:length(input_names)
        name = input_names{i};
        pos = inputs.(name);
        inport = add_block('simulink/Sources/In1', [model_name '/' name]);
        set_param(inport, 'Position', [pos(1) pos(2) pos(1)+80 pos(2)+16]);
        set_param(inport, 'Port', num2str(i));

        % 设置默认值
        if strcmp(name, 'current_distance')
            set_param(inport, 'OutDataTypeStr', 'double');
        elseif strcmp(name, 'has_target') || strcmp(name, 'reset_signal')
            set_param(inport, 'OutDataTypeStr', 'boolean');
        end
    end

    % 创建输出端口
    output_names = fieldnames(outputs);
    for i = 1:length(output_names)
        name = output_names{i};
        pos = outputs.(name);
        outport = add_block('simulink/Sinks/Out1', [model_name '/' name]);
        set_param(outport, 'Position', [pos(1) pos(2) pos(1)+80 pos(2)+16]);
        set_param(outport, 'Port', num2str(i));
    end

    fprintf('✅ 输入输出端口创建完成\n');
end

function configure_decision_md_chart(model_name)
    %% 配置基于decision.md的Stateflow Chart

    fprintf('配置Stateflow Chart...\n');

    % 获取Chart对象
    rt = sfroot;
    chart = rt.find('-isa', 'Stateflow.Chart', 'Path', [model_name '/ACC_Decision_Chart']);

    if isempty(chart)
        error('无法找到Stateflow Chart');
    end

    % 添加输入数据
    add_chart_data(chart, 'command_input', 'Input', 'double');
    add_chart_data(chart, 'ego_speed_kmh', 'Input', 'double');
    add_chart_data(chart, 'has_target', 'Input', 'boolean');
    add_chart_data(chart, 'current_distance', 'Input', 'double');
    add_chart_data(chart, 'reset_signal', 'Input', 'boolean');

    % 添加输出数据
    add_chart_data(chart, 'current_state', 'Output', 'double', '2'); % 默认S2无史待命
    add_chart_data(chart, 'control_mode', 'Output', 'double', '0');
    add_chart_data(chart, 'acc_active', 'Output', 'boolean', 'false');
    add_chart_data(chart, 'control_enabled', 'Output', 'boolean', 'false');
    add_chart_data(chart, 'V3_kmh', 'Output', 'double', '50.0');
    add_chart_data(chart, 'G1_m', 'Output', 'double', '15.0');
    add_chart_data(chart, 'G2_s', 'Output', 'double', '2.0');
    add_chart_data(chart, 'has_history', 'Output', 'boolean', 'false');
    add_chart_data(chart, 'message_code', 'Output', 'double', '0');
    add_chart_data(chart, 'pending_distance_adj', 'Output', 'double', '0');
    add_chart_data(chart, 'is_in_control', 'Output', 'boolean', 'false');

    % 添加局部变量（历史数据）
    add_chart_data(chart, 'history_V3_kmh', 'Local', 'double', '50.0');
    add_chart_data(chart, 'history_G1_m', 'Local', 'double', '15.0');
    add_chart_data(chart, 'history_G2_s', 'Local', 'double', '2.0');
    add_chart_data(chart, 'pending_adj', 'Local', 'double', '0');

    % 添加常量
    add_chart_data(chart, 'V1_KMH', 'Constant', 'double', '30.0'); % 适速/低速界限
    add_chart_data(chart, 'V2_KMH', 'Constant', 'double', '30.0');
    add_chart_data(chart, 'SPEED_STEP', 'Constant', 'double', '1.0');
    add_chart_data(chart, 'DISTANCE_STEP', 'Constant', 'double', '1.0');

    fprintf('✅ Chart数据接口配置完成\n');
end

function create_parameter_subsystem(model_name)
    %% 创建参数初始化子系统

    fprintf('创建参数初始化子系统...\n');

    try
        % 添加常量块而不是MATLAB Function块，避免Script参数问题
        v3_block = add_block('simulink/Sources/Constant', [model_name '/V3_Constant']);
        set_param(v3_block, 'Position', [50, 500, 100, 530]);
        set_param(v3_block, 'Value', '50.0');
        set_param(v3_block, 'OutDataTypeStr', 'double');

        g1_block = add_block('simulink/Sources/Constant', [model_name '/G1_Constant']);
        set_param(g1_block, 'Position', [50, 540, 100, 570]);
        set_param(g1_block, 'Value', '15.0');
        set_param(g1_block, 'OutDataTypeStr', 'double');

        g2_block = add_block('simulink/Sources/Constant', [model_name '/G2_Constant']);
        set_param(g2_block, 'Position', [50, 580, 100, 610]);
        set_param(g2_block, 'Value', '2.0');
        set_param(g2_block, 'OutDataTypeStr', 'double');

        fprintf('✅ 参数常量块创建完成\n');

    catch ME
        fprintf('⚠️ 参数子系统创建失败: %s\n', ME.message);
        fprintf('  可以在模型创建完成后手动添加参数块\n');
    end
end

function connect_signals(model_name)
    %% 连接信号线 - 修复版本

    fprintf('连接信号线...\n');

    chart_block = [model_name '/ACC_Decision_Chart'];

    % 先等待一下，确保Chart完全创建
    pause(0.5);

    % 连接输入信号到Chart - 使用正确的端口索引
    input_names = {'command_input', 'ego_speed_kmh', 'has_target', 'current_distance', 'reset_signal'};
    for i = 1:length(input_names)
        try
            % 方法1：尝试使用端口索引连接
            src_block = [model_name '/' input_names{i}];
            dst_block = chart_block;
            add_line(model_name, [src_block '/1'], [dst_block '/' num2str(i)], 'autorouting', 'on');
            fprintf('  ✅ 连接输入: %s (端口 %d)\n', input_names{i}, i);
        catch ME1
            try
                % 方法2：尝试不同的连接方式
                ports = get_param(chart_block, 'PortHandles');
                if ~isempty(ports) && isfield(ports, 'Inport') && length(ports.Inport) >= i
                    src_handle = get_param([model_name '/' input_names{i}], 'PortHandles');
                    add_line(model_name, src_handle.Outport(1), ports.Inport(i), 'autorouting', 'on');
                    fprintf('  ✅ 连接输入: %s (句柄方式)\n', input_names{i});
                else
                    fprintf('  ⚠️ 连接输入信号 %s 失败 - 端口未就绪\n', input_names{i});
                end
            catch ME2
                fprintf('  ⚠️ 连接输入信号 %s 失败: %s\n', input_names{i}, ME1.message);
            end
        end
    end

    % 连接Chart输出到输出端口
    output_names = {'current_state', 'control_mode', 'acc_active', 'control_enabled', ...
                   'V3_kmh', 'G1_m', 'G2_s', 'has_history', 'message_code', ...
                   'pending_distance_adj', 'is_in_control'};

    for i = 1:length(output_names)
        try
            % 方法1：使用端口索引连接
            src_block = chart_block;
            dst_block = [model_name '/' output_names{i}];
            add_line(model_name, [src_block '/' num2str(i)], [dst_block '/1'], 'autorouting', 'on');
            fprintf('  ✅ 连接输出: %s (端口 %d)\n', output_names{i}, i);
        catch ME1
            try
                % 方法2：使用句柄连接
                ports = get_param(chart_block, 'PortHandles');
                if ~isempty(ports) && isfield(ports, 'Outport') && length(ports.Outport) >= i
                    dst_handle = get_param([model_name '/' output_names{i}], 'PortHandles');
                    add_line(model_name, ports.Outport(i), dst_handle.Inport(1), 'autorouting', 'on');
                    fprintf('  ✅ 连接输出: %s (句柄方式)\n', output_names{i});
                else
                    fprintf('  ⚠️ 连接输出信号 %s 失败 - 端口未就绪\n', output_names{i});
                end
            catch ME2
                fprintf('  ⚠️ 连接输出信号 %s 失败: %s\n', output_names{i}, ME1.message);
            end
        end
    end

    fprintf('✅ 信号线连接完成\n');

end

function add_chart_data(chart, name, scope, dataType, initValue)
    %% 添加Chart数据的辅助函数

    data = Stateflow.Data(chart);
    data.Name = name;
    data.Scope = scope;
    data.DataType = dataType;

    if nargin >= 5 && ~isempty(initValue)
        data.Props.InitialValue = initValue;
    end
end