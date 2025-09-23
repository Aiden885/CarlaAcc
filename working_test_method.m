% 工作的SPPVT输出获取方法
function working_test_method()

model_name = 'ACC_Decision_SPPVT_Integrated';

fprintf('🔬 开始完整的输出获取方法测试\n');

% 加载总线定义
load('DecisionSPPVTBusDefinitions.mat');
fprintf('✅ 总线定义已加载\n');

% 加载模型
if ~bdIsLoaded(model_name)
    load_system(model_name);
    fprintf('✅ 模型已加载\n');
end

% 创建测试输入数据
test_input = create_test_input();

%% 测试方法1: Dataset格式输出
fprintf('\n📋 方法1: Dataset格式输出测试\n');
success1 = test_dataset_output(model_name, test_input);

%% 测试方法2: Array格式输出
fprintf('\n📋 方法2: Array格式输出测试\n');
success2 = test_array_output(model_name, test_input);

%% 测试方法3: Structure格式输出
fprintf('\n📋 方法3: Structure格式输出测试\n');
success3 = test_structure_output(model_name, test_input);

%% 测试方法4: To Workspace块方法
fprintf('\n📋 方法4: To Workspace块方法\n');
success4 = test_workspace_blocks(model_name, test_input);

%% 汇总结果
fprintf('\n📊 测试结果汇总:\n');
if success1
    fprintf('Dataset方法: ✅ 成功\n');
else
    fprintf('Dataset方法: ❌ 失败\n');
end
if success2
    fprintf('Array方法: ✅ 成功\n');
else
    fprintf('Array方法: ❌ 失败\n');
end
if success3
    fprintf('Structure方法: ✅ 成功\n');
else
    fprintf('Structure方法: ❌ 失败\n');
end
if success4
    fprintf('Workspace块方法: ✅ 成功\n');
else
    fprintf('Workspace块方法: ❌ 失败\n');
end

if success1 || success2 || success3 || success4
    fprintf('\n🎉 找到了可行的输出获取方法！\n');
else
    fprintf('\n⚠️ 所有方法都失败，需要进一步调试\n');
end

end

%% 创建测试输入数据
function test_input = create_test_input()
    ego_speed_kmh = 50.0;
    ego_speed_ms = ego_speed_kmh / 3.6;
    V_target_kmh = 50.0;
    V_min_kmh = 30.0;
    G2_s = 2.0;

    time_points = [0, 0.05];
    test_input = struct();
    test_input.ego_speed_kmh = timeseries([ego_speed_kmh, ego_speed_kmh], time_points, 'Name', 'ego_speed_kmh');
    test_input.ego_speed_ms = timeseries([ego_speed_ms, ego_speed_ms], time_points, 'Name', 'ego_speed_ms');
    test_input.command_type = timeseries(int32([1, 1]), time_points, 'Name', 'command_type');
    test_input.command_active = timeseries(logical([true, true]), time_points, 'Name', 'command_active');
    test_input.manual_throttle_active = timeseries(logical([false, false]), time_points, 'Name', 'manual_throttle_active');
    test_input.control_error = timeseries([1.5, 1.5], time_points, 'Name', 'control_error');
    test_input.control_mode_flag = timeseries(int32([1, 1]), time_points, 'Name', 'control_mode_flag');
    test_input.V_target_kmh = timeseries([V_target_kmh, V_target_kmh], time_points, 'Name', 'V_target_kmh');
    test_input.V_min_kmh = timeseries([V_min_kmh, V_min_kmh], time_points, 'Name', 'V_min_kmh');
    test_input.G2_s = timeseries([G2_s, G2_s], time_points, 'Name', 'G2_s');
    test_input.timestamp = timeseries(time_points, time_points, 'Name', 'timestamp');

    % 设置时间单位
    field_names = fieldnames(test_input);
    for i = 1:length(field_names)
        test_input.(field_names{i}).TimeInfo.Units = 'seconds';
    end
end

%% 测试Dataset格式输出
function success = test_dataset_output(model_name, test_input)
    try
        % 配置参数
        set_param(model_name, 'StopTime', '0.05');
        set_param(model_name, 'LoadExternalInput', 'on');
        set_param(model_name, 'SaveOutput', 'on');
        set_param(model_name, 'OutputSaveName', 'yout');
        set_param(model_name, 'SaveFormat', 'Dataset');

        % 清理并设置输入
        evalin('base', 'clearvars -except DecisionSPPVTInput DecisionSPPVTOutput');
        assignin('base', 'input_data', test_input);
        set_param(model_name, 'ExternalInput', 'input_data');

        % 运行仿真
        sim_out = sim(model_name);

        % 检查输出
        if isfield(sim_out, 'yout') && isa(sim_out.yout, 'Simulink.SimulationData.Dataset')
            dataset = sim_out.yout;
            fprintf('  Dataset元素数量: %d\n', length(dataset));

            % 查找SPPVT输出
            sppvt_element = find(dataset, 'Name', 'sppvt_control_output');
            if ~isempty(sppvt_element)
                sppvt_value = double(sppvt_element.Values.Data(end));
                fprintf('  ✅ SPPVT输出值: %.6f\n', sppvt_value);
                success = true;
                return;
            end
        end

        % 检查工作区
        if evalin('base', 'exist(''yout'', ''var'')')
            yout = evalin('base', 'yout');
            if isa(yout, 'Simulink.SimulationData.Dataset')
                sppvt_element = find(yout, 'Name', 'sppvt_control_output');
                if ~isempty(sppvt_element)
                    sppvt_value = double(sppvt_element.Values.Data(end));
                    fprintf('  ✅ 工作区SPPVT输出值: %.6f\n', sppvt_value);
                    success = true;
                    return;
                end
            end
        end

        fprintf('  ❌ 未找到SPPVT输出数据\n');
        success = false;

    catch ME
        fprintf('  ❌ 错误: %s\n', ME.message);
        success = false;
    end
end

%% 测试Array格式输出
function success = test_array_output(model_name, test_input)
    try
        % 配置参数
        set_param(model_name, 'SaveFormat', 'Array');
        set_param(model_name, 'OutputSaveName', 'yout_array');

        % 清理并设置输入
        evalin('base', 'clearvars -except DecisionSPPVTInput DecisionSPPVTOutput');
        assignin('base', 'input_data', test_input);

        % 运行仿真
        sim_out = sim(model_name);

        % 检查输出
        if evalin('base', 'exist(''yout_array'', ''var'')')
            yout_array = evalin('base', 'yout_array');
            fprintf('  Array输出大小: %s\n', mat2str(size(yout_array)));
            if ~isempty(yout_array) && size(yout_array, 2) >= 8
                sppvt_value = yout_array(end, 8); % 第8列应该是sppvt_control_output
                fprintf('  ✅ Array SPPVT输出值: %.6f\n', sppvt_value);
                success = true;
                return;
            end
        end

        fprintf('  ❌ Array方法失败\n');
        success = false;

    catch ME
        fprintf('  ❌ 错误: %s\n', ME.message);
        success = false;
    end
end

%% 测试Structure格式输出
function success = test_structure_output(model_name, test_input)
    try
        % 配置参数
        set_param(model_name, 'SaveFormat', 'Structure');
        set_param(model_name, 'OutputSaveName', 'yout_struct');

        % 清理并设置输入
        evalin('base', 'clearvars -except DecisionSPPVTInput DecisionSPPVTOutput');
        assignin('base', 'input_data', test_input);

        % 运行仿真
        sim_out = sim(model_name);

        % 检查输出
        if evalin('base', 'exist(''yout_struct'', ''var'')')
            yout_struct = evalin('base', 'yout_struct');
            fprintf('  Structure字段: %s\n', strjoin(fieldnames(yout_struct), ', '));
            if isfield(yout_struct, 'signals') && length(yout_struct.signals) >= 8
                sppvt_signal = yout_struct.signals(8);
                if ~isempty(sppvt_signal.values)
                    sppvt_value = sppvt_signal.values(end);
                    fprintf('  ✅ Structure SPPVT输出值: %.6f\n', sppvt_value);
                    success = true;
                    return;
                end
            end
        end

        fprintf('  ❌ Structure方法失败\n');
        success = false;

    catch ME
        fprintf('  ❌ 错误: %s\n', ME.message);
        success = false;
    end
end

%% 测试To Workspace块方法
function success = test_workspace_blocks(model_name, test_input)
    try
        % 查找To Workspace块
        to_workspace_blocks = find_system(model_name, 'BlockType', 'ToWorkspace');
        if isempty(to_workspace_blocks)
            fprintf('  ❌ 未找到To Workspace块\n');
            success = false;
            return;
        end

        fprintf('  找到%d个To Workspace块\n', length(to_workspace_blocks));

        % 清理并设置输入
        evalin('base', 'clearvars -except DecisionSPPVTInput DecisionSPPVTOutput');
        assignin('base', 'input_data', test_input);

        % 运行仿真
        sim_out = sim(model_name);

        % 检查各个To Workspace块的输出
        for i = 1:length(to_workspace_blocks)
            block_name = to_workspace_blocks{i};
            var_name = get_param(block_name, 'VariableName');

            if evalin('base', sprintf('exist(''%s'', ''var'')', var_name))
                var_data = evalin('base', var_name);
                fprintf('  变量%s: 类型=%s, 大小=%s\n', var_name, class(var_data), mat2str(size(var_data)));

                % 如果变量名包含sppvt，认为找到了
                if contains(lower(var_name), 'sppvt') || contains(lower(block_name), 'sppvt')
                    if isnumeric(var_data) && ~isempty(var_data)
                        sppvt_value = var_data(end);
                        fprintf('  ✅ Workspace SPPVT输出值: %.6f\n', sppvt_value);
                        success = true;
                        return;
                    end
                end
            end
        end

        fprintf('  ❌ Workspace块方法未找到SPPVT数据\n');
        success = false;

    catch ME
        fprintf('  ❌ 错误: %s\n', ME.message);
        success = false;
    end
end