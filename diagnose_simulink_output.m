function diagnose_simulink_output()
%% diagnose_simulink_output - 诊断Simulink输出问题
% =========================================================================
% 这个脚本用于诊断为什么测试没有获取到正确的输出数据
% =========================================================================

    fprintf('\n=== Simulink输出诊断开始 ===\n\n');

    model_name = 'acc_decision_md';

    % 1. 加载模型
    if ~bdIsLoaded(model_name)
        fprintf('正在加载模型: %s.slx\n', model_name);
        load_system(model_name);
    end

    % 2. 检查模型的输出配置
    fprintf('【1】检查模型输出配置\n');
    fprintf('-----------------------------------\n');

    % 获取模型的输出端口
    outports = find_system(model_name, 'SearchDepth', 1, 'BlockType', 'Outport');
    fprintf('找到 %d 个输出端口:\n', length(outports));
    for i = 1:length(outports)
        port_name = get_param(outports{i}, 'Name');
        port_num = get_param(outports{i}, 'Port');
        fprintf('  端口 %s: %s\n', port_num, port_name);
    end
    fprintf('\n');

    % 3. 运行一个简单的仿真
    fprintf('【2】运行简单仿真测试\n');
    fprintf('-----------------------------------\n');

    % 设置简单的输入
    command_input = timeseries(0, [0; 0.1]);
    ego_speed_kmh = timeseries(80, [0; 0.1]);
    has_target = timeseries(false, [0; 0.1]);
    current_distance = timeseries(-1, [0; 0.1]);
    reset_signal = timeseries(false, [0; 0.1]);

    assignin('base', 'command_input', command_input);
    assignin('base', 'ego_speed_kmh', ego_speed_kmh);
    assignin('base', 'has_target', has_target);
    assignin('base', 'current_distance', current_distance);
    assignin('base', 'reset_signal', reset_signal);

    % 配置仿真参数
    set_param(model_name, 'LoadExternalInput', 'on');
    set_param(model_name, 'ExternalInput', 'command_input, ego_speed_kmh, has_target, current_distance, reset_signal');
    set_param(model_name, 'StopTime', '0.1');
    set_param(model_name, 'SaveOutput', 'on');
    set_param(model_name, 'OutputSaveName', 'yout');
    set_param(model_name, 'SaveFormat', 'Dataset');  % 明确指定Dataset格式

    % 运行仿真
    fprintf('运行仿真...\n');
    try
        sim_result = sim(model_name);
        fprintf('仿真完成\n\n');
    catch ME
        fprintf('❌ 仿真失败: %s\n', ME.message);
        cleanup_workspace();
        return;
    end

    % 4. 详细分析输出数据结构
    fprintf('【3】分析输出数据结构\n');
    fprintf('-----------------------------------\n');

    yout = sim_result.yout;
    fprintf('输出类型: %s\n', class(yout));

    if isa(yout, 'Simulink.SimulationData.Dataset')
        fprintf('Dataset包含 %d 个元素:\n', yout.numElements);
        fprintf('\n');

        % 遍历所有元素，显示详细信息
        for i = 1:yout.numElements
            elem = yout.getElement(i);
            fprintf('元素 %d:\n', i);
            fprintf('  名称: %s\n', elem.Name);
            fprintf('  类型: %s\n', class(elem.Values));

            if isa(elem.Values, 'timeseries')
                data = elem.Values.Data;
                fprintf('  数据维度: %s\n', mat2str(size(data)));
                fprintf('  时间点数: %d\n', length(elem.Values.Time));
                fprintf('  第一个值: %s\n', mat2str(data(1,:)));
                fprintf('  最后值: %s\n', mat2str(data(end,:)));
            else
                fprintf('  值: %s\n', mat2str(elem.Values));
            end
            fprintf('\n');
        end

        % 尝试按名称访问
        fprintf('【4】测试按名称访问Dataset元素\n');
        fprintf('-----------------------------------\n');

        test_names = {'current_state', 'control_mode', 'V3_kmh', 'has_history'};
        for i = 1:length(test_names)
            name = test_names{i};
            try
                elem = yout.getElement(name);
                if isa(elem.Values, 'timeseries')
                    value = elem.Values.Data(end);
                else
                    value = elem.Values(end);
                end
                fprintf('✅ %s = %s\n', name, mat2str(value));
            catch
                % 尝试按索引访问
                try
                    if i <= yout.numElements
                        elem = yout.getElement(i);
                        if isa(elem.Values, 'timeseries')
                            value = elem.Values.Data(end);
                        else
                            value = elem.Values(end);
                        end
                        fprintf('⚠️ %s (通过索引%d访问) = %s\n', name, i, mat2str(value));
                    else
                        fprintf('❌ %s - 索引超出范围\n', name);
                    end
                catch
                    fprintf('❌ %s - 无法访问\n', name);
                end
            end
        end

    else
        fprintf('输出不是Dataset格式，类型: %s\n', class(yout));
        if isstruct(yout)
            fields = fieldnames(yout);
            fprintf('结构体字段:\n');
            for i = 1:length(fields)
                fprintf('  - %s\n', fields{i});
            end
        end
    end

    % 5. 检查Stateflow状态
    fprintf('\n【5】检查Stateflow Chart状态\n');
    fprintf('-----------------------------------\n');

    try
        rt = sfroot;
        chart_path = [model_name '/ACC_Decision_Chart'];
        chart = rt.find('-isa', 'Stateflow.Chart', 'Path', chart_path);

        if ~isempty(chart)
            fprintf('Chart找到，ID: %d\n', chart.Id);

            % 查找所有状态
            states = chart.find('-isa', 'Stateflow.State');
            fprintf('找到 %d 个状态:\n', length(states));
            for i = 1:length(states)
                fprintf('  - %s\n', states(i).Name);
            end

            % 查找默认转移
            transitions = chart.find('-isa', 'Stateflow.Transition');
            default_trans = [];
            for i = 1:length(transitions)
                if isempty(transitions(i).Source)
                    default_trans = transitions(i);
                    break;
                end
            end

            if ~isempty(default_trans)
                if ~isempty(default_trans.Destination)
                    fprintf('默认入口转移指向: %s\n', default_trans.Destination.Name);
                else
                    fprintf('默认入口转移存在但没有目标\n');
                end
            else
                fprintf('⚠️ 没有找到默认入口转移！\n');
            end
        else
            fprintf('❌ 无法找到Stateflow Chart\n');
        end
    catch ME
        fprintf('检查Stateflow时出错: %s\n', ME.message);
    end

    % 6. 测试不同的输出提取方法
    fprintf('\n【6】测试输出提取方法\n');
    fprintf('-----------------------------------\n');

    % 方法1: Simulink To Workspace blocks
    to_workspace_blocks = find_system(model_name, 'BlockType', 'ToWorkspace');
    if ~isempty(to_workspace_blocks)
        fprintf('找到 %d 个To Workspace块:\n', length(to_workspace_blocks));
        for i = 1:length(to_workspace_blocks)
            var_name = get_param(to_workspace_blocks{i}, 'VariableName');
            fprintf('  - %s\n', var_name);
            if evalin('base', ['exist(''' var_name ''', ''var'')'])
                value = evalin('base', var_name);
                fprintf('    值: %s\n', mat2str(value));
            end
        end
    else
        fprintf('没有找到To Workspace块\n');
    end

    % 清理
    cleanup_workspace();

    fprintf('\n=== 诊断完成 ===\n');
    fprintf('\n建议:\n');
    fprintf('1. 如果输出名称不匹配，需要修改extract_outputs函数使用正确的名称\n');
    fprintf('2. 如果没有默认转移，需要在Stateflow配置中添加\n');
    fprintf('3. 如果Dataset元素顺序不对，改用getElement(name)而不是getElement(index)\n');
end

function cleanup_workspace()
    % 清理工作空间变量
    vars_to_clear = {'command_input', 'ego_speed_kmh', 'has_target', ...
                     'current_distance', 'reset_signal'};
    for i = 1:length(vars_to_clear)
        try
            evalin('base', ['clear ' vars_to_clear{i}]);
        catch
        end
    end
end