function diagnose_acc_decision_model()
%% diagnose_acc_decision_model - ACC决策模型诊断脚本
% 用于诊断状态机不响应指令的根本原因

    fprintf('=== ACC决策模型诊断开始 ===\n');

    model_name = 'acc_decision_md';
    chart_path = [model_name '/ACC_Decision_Chart'];

    % 1. 检查模型和Chart是否正确加载
    fprintf('\n1. 模型加载检查:\n');
    if bdIsLoaded(model_name)
        fprintf('   ✅ 模型 %s 已加载\n', model_name);
    else
        fprintf('   ❌ 模型 %s 未加载\n', model_name);
        return;
    end

    % 2. 检查Stateflow Chart
    fprintf('\n2. Stateflow Chart检查:\n');
    try
        rt = sfroot;
        chart = rt.find('-isa', 'Stateflow.Chart', 'Path', chart_path);
        if isempty(chart)
            fprintf('   ❌ 找不到Stateflow Chart: %s\n', chart_path);
            return;
        else
            fprintf('   ✅ Chart对象找到 (ID: %d)\n', chart.Id);
        end
    catch ME
        fprintf('   ❌ Chart访问失败: %s\n', ME.message);
        return;
    end

    % 3. 检查Chart内容
    fprintf('\n3. Chart内容检查:\n');
    states = chart.find('-isa', 'Stateflow.State');
    transitions = chart.find('-isa', 'Stateflow.Transition');
    functions = chart.find('-isa', 'Stateflow.EMFunction');

    fprintf('   状态数量: %d\n', length(states));
    fprintf('   转移数量: %d\n', length(transitions));
    fprintf('   函数数量: %d\n', length(functions));

    if length(states) < 4
        fprintf('   ⚠️ 状态数量不足，期望至少4个状态\n');
    end

    % 4. 检查Chart数据接口
    fprintf('\n4. Chart数据接口检查:\n');
    chart_data = chart.find('-isa', 'Stateflow.Data');

    input_data = [];
    output_data = [];
    for i = 1:length(chart_data)
        data = chart_data(i);
        if strcmp(data.Scope, 'Input')
            input_data = [input_data; data];
        elseif strcmp(data.Scope, 'Output')
            output_data = [output_data; data];
        end
    end

    fprintf('   输入数据: %d个\n', length(input_data));
    for i = 1:length(input_data)
        fprintf('     - %s (%s)\n', input_data(i).Name, input_data(i).DataType);
    end

    fprintf('   输出数据: %d个\n', length(output_data));
    for i = 1:length(output_data)
        fprintf('     - %s (%s)\n', output_data(i).Name, output_data(i).DataType);
    end

    % 5. 检查模型连接
    fprintf('\n5. 模型连接检查:\n');
    check_model_connections(model_name);

    % 6. 执行简单仿真测试
    fprintf('\n6. 简单仿真测试:\n');
    run_simple_simulation_test(model_name);

    % 7. 检查Chart编译状态
    fprintf('\n7. Chart编译状态检查:\n');
    check_chart_compilation(chart);

    fprintf('\n=== 诊断完成 ===\n');
end

function check_model_connections(model_name)
    try
        % 获取所有模块
        blocks = find_system(model_name, 'Type', 'block');

        % 查找输入端口
        input_blocks = find_system(model_name, 'BlockType', 'Inport');
        fprintf('   输入端口数量: %d\n', length(input_blocks));

        % 查找输出端口
        output_blocks = find_system(model_name, 'BlockType', 'Outport');
        fprintf('   输出端口数量: %d\n', length(output_blocks));

        % 查找Chart块
        chart_blocks = find_system(model_name, 'MaskType', 'Stateflow');
        if isempty(chart_blocks)
            chart_blocks = find_system(model_name, 'BlockType', 'SubSystem', ...
                                     'Tag', 'Stateflow');
        end
        fprintf('   Chart块数量: %d\n', length(chart_blocks));

        if ~isempty(chart_blocks)
            chart_block = chart_blocks{1};

            % 检查Chart的端口连接
            ports = get_param(chart_block, 'PortHandles');
            fprintf('   Chart输入端口: %d个\n', length(ports.Inport));
            fprintf('   Chart输出端口: %d个\n', length(ports.Outport));

            % 检查输入连接
            unconnected_inputs = 0;
            for i = 1:length(ports.Inport)
                line = get_param(ports.Inport(i), 'Line');
                if line == -1
                    unconnected_inputs = unconnected_inputs + 1;
                end
            end

            % 检查输出连接
            unconnected_outputs = 0;
            for i = 1:length(ports.Outport)
                line = get_param(ports.Outport(i), 'Line');
                if line == -1
                    unconnected_outputs = unconnected_outputs + 1;
                end
            end

            if unconnected_inputs > 0
                fprintf('   ⚠️ 未连接输入端口: %d个\n', unconnected_inputs);
            else
                fprintf('   ✅ 所有输入端口已连接\n');
            end

            if unconnected_outputs > 0
                fprintf('   ⚠️ 未连接输出端口: %d个\n', unconnected_outputs);
            else
                fprintf('   ✅ 所有输出端口已连接\n');
            end
        end

    catch ME
        fprintf('   ❌ 连接检查失败: %s\n', ME.message);
    end
end

function run_simple_simulation_test(model_name)
    try
        % 创建简单的常量输入
        input_names = {'command_input', 'ego_speed_kmh', 'has_target', ...
                      'current_distance', 'reset_signal'};

        % 设置常量输入值
        for i = 1:length(input_names)
            sig_name = input_names{i};
            if strcmp(sig_name, 'has_target') || strcmp(sig_name, 'reset_signal')
                assignin('base', sig_name, false);
            elseif strcmp(sig_name, 'command_input')
                assignin('base', sig_name, 0);  % 降速指令
            elseif strcmp(sig_name, 'ego_speed_kmh')
                assignin('base', sig_name, 50);
            else
                assignin('base', sig_name, -1);
            end
        end

        % 配置仿真
        set_param(model_name, 'StopTime', '1.0');
        set_param(model_name, 'SaveOutput', 'on');
        set_param(model_name, 'OutputSaveName', 'yout');
        set_param(model_name, 'SaveFormat', 'Dataset');

        fprintf('   执行1秒仿真测试...\n');

        % 运行仿真
        sim_result = sim(model_name);

        % 检查输出
        if isfield(sim_result, 'yout')
            yout = sim_result.yout;
            if isa(yout, 'Simulink.SimulationData.Dataset')
                fprintf('   ✅ 仿真成功，输出Dataset包含 %d 个元素\n', yout.numElements);

                % 显示最终输出值
                fprintf('   最终输出值:\n');
                try
                    for i = 1:min(yout.numElements, 5)  % 只显示前5个
                        elem = yout.getElement(i);
                        if isa(elem.Values, 'timeseries')
                            final_val = elem.Values.Data(end);
                        else
                            final_val = elem.Values(end);
                        end
                        fprintf('     输出%d: %.3f\n', i, final_val);
                    end
                catch ME2
                    fprintf('   ⚠️ 输出值提取失败: %s\n', ME2.message);
                end
            else
                fprintf('   ⚠️ 输出格式不是Dataset: %s\n', class(yout));
            end
        else
            fprintf('   ⚠️ 没有找到输出数据\n');
        end

        % 清理工作空间
        for i = 1:length(input_names)
            evalin('base', ['clear ' input_names{i}]);
        end

    catch ME
        fprintf('   ❌ 仿真测试失败: %s\n', ME.message);

        % 清理工作空间
        input_names = {'command_input', 'ego_speed_kmh', 'has_target', ...
                      'current_distance', 'reset_signal'};
        for i = 1:length(input_names)
            try
                evalin('base', ['clear ' input_names{i}]);
            catch
            end
        end
    end
end

function check_chart_compilation(chart)
    try
        % 检查Chart是否有编译错误
        fprintf('   Chart名称: %s\n', chart.Name);
        fprintf('   Chart路径: %s\n', chart.Path);

        % 检查Chart的机器
        machine = chart.Machine;
        if ~isempty(machine)
            fprintf('   ✅ Chart属于状态机: %s\n', machine.Name);
        else
            fprintf('   ⚠️ Chart没有关联状态机\n');
        end

        % 尝试获取Chart的创建信息
        fprintf('   Chart ID: %d\n', chart.Id);
        fprintf('   Chart类型: %s\n', class(chart));

    catch ME
        fprintf('   ❌ Chart编译检查失败: %s\n', ME.message);
    end
end