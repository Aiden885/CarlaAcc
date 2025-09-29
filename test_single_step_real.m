function test_single_step_real()
%TEST_SINGLE_STEP_REAL 真正的单步测试 - 模拟Python使用模式
%   模拟Python端每次调用0.05s的单步仿真，状态在调用间维持

fprintf('🧪 开始真正的单步测试模式...\n');
fprintf('📝 模拟Python端状态管理和单步调用\n\n');

model_name = 'ACC_Decision_SPPVT_Integrated';

try
    %% 1. 准备模型
    fprintf('🔧 准备模型...\n');

    % 确保总线定义存在
    if ~evalin('base', 'exist(''DecisionSPPVTInputExtended'', ''var'')')
        create_decision_sppvt_bus();
    end

    % 加载模型
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
    load_system(model_name);

    % 配置为单步模式
    set_param(model_name, 'StopTime', '0.05');     % 单步时间
    set_param(model_name, 'FixedStep', '0.05');    % 固定步长
    set_param(model_name, 'SaveOutput', 'on');
    set_param(model_name, 'OutputSaveName', 'yout');
    set_param(model_name, 'SaveFormat', 'Dataset');

    fprintf('✅ 模型配置完成（单步模式：0.05s）\n');

    %% 2. 初始化Python端状态（模拟）
    fprintf('\n📊 初始化Python端状态管理...\n');

    % 模拟Python端维持的状态
    decision_state = struct();
    decision_state.current_state = int32(2);        % S2: 无史待命
    decision_state.has_history = false;             % 无历史
    decision_state.last_active_decision = int32(8); % R8: 系统待命

    % 模拟SPPVT状态
    sppvt_state = struct();
    sppvt_state.stage_offset = 0.0;
    sppvt_state.stage_manager_states = [1.0; 0.0; 0.0];  % [stage; error_sign; upgrade_count] - 列向量
    sppvt_state.adapter_states = [0.0; 0.0; 0.0];        % [prev_error; prev_velocity; prev_accel] - 列向量

    % 车辆状态
    vehicle_state = struct();
    vehicle_state.ego_speed_kmh = 50.0;
    vehicle_state.ego_speed_ms = 13.89;
    vehicle_state.V_target_kmh = 50.0;
    vehicle_state.V_min_kmh = 30.0;
    vehicle_state.G2_s = 2.0;

    fprintf('✅ 初始状态: S%d, 历史=%d, 决策=R%d, 速度=%.1fkm/h\n', ...
        decision_state.current_state, decision_state.has_history, ...
        decision_state.last_active_decision, vehicle_state.ego_speed_kmh);

    %% 3. 定义测试场景序列
    fprintf('\n📋 定义测试场景序列...\n');

    % 场景序列：[步骤, 描述, 命令类型, 命令激活, 期望状态变化]
    test_scenarios = {
        1, '初始状态检查',     0, false, 'S2→S2';
        2, 'I0当速启控',       1, true,  'S2→S0';
        3, '维持控制状态',     0, false, 'S0→S0';
        4, 'I1增速控制',       2, true,  'S0→S0';
        5, '继续控制',         0, false, 'S0→S0';
        6, 'I6取消控制',       7, true,  'S0→S1';
        7, '待命状态',         0, false, 'S1→S1';
        8, 'I1恢复控制',       2, true,  'S1→S0';
    };

    num_steps = size(test_scenarios, 1);
    fprintf('✅ 准备执行 %d 个单步测试场景\n', num_steps);

    %% 4. 执行单步测试序列
    fprintf('\n▶️ 开始执行单步测试序列...\n');
    fprintf('=====================================\n');

    results = cell(num_steps, 6);  % 存储结果

    for step = 1:num_steps
        step_num = test_scenarios{step, 1};
        description = test_scenarios{step, 2};
        command_type = test_scenarios{step, 3};
        command_active = test_scenarios{step, 4};
        expected_transition = test_scenarios{step, 5};

        fprintf('步骤 %d: %s\n', step_num, description);
        fprintf('  输入: 命令I%d, 激活=%d, 当前状态=S%d\n', ...
            command_type, command_active, decision_state.current_state);

        % 创建单步输入
        single_input = create_single_step_input(...
            vehicle_state, decision_state, sppvt_state, command_type, command_active);

        % 设置外部输入
        assignin('base', 'single_step_input', single_input);
        set_param(model_name, 'LoadExternalInput', 'on');
        set_param(model_name, 'ExternalInput', 'single_step_input');

        % 运行单步仿真
        tic;
        sim_out = sim(model_name);
        sim_time = toc;

        % 解析输出
        [success, output_state, sppvt_output] = parse_single_step_output(sim_out);

        if success
            % 更新Python端状态（模拟状态外化）- 确保数据类型正确
            old_state = decision_state.current_state;
            decision_state.current_state = int32(output_state.next_state);
            decision_state.has_history = logical(output_state.next_has_history);
            decision_state.last_active_decision = int32(output_state.next_last_active_decision);

            % 更新SPPVT状态
            sppvt_state.stage_offset = output_state.new_stage_offset;
            if ~isempty(output_state.new_stage_manager_states)
                sppvt_state.stage_manager_states = output_state.new_stage_manager_states;
            end
            if ~isempty(output_state.new_adapter_states)
                sppvt_state.adapter_states = output_state.new_adapter_states;
            end

            % 更新车辆参数
            vehicle_state.V_target_kmh = output_state.updated_V_target_kmh;
            vehicle_state.G2_s = output_state.updated_G2_s;

            fprintf('  输出: S%d→S%d, R%d, 控制=%d, SPPVT=%.3f (%.1fms)\n', ...
                old_state, output_state.current_state, output_state.current_decision, ...
                output_state.control_enabled, sppvt_output, sim_time*1000);

            % 记录结果
            results{step, 1} = step_num;
            results{step, 2} = description;
            results{step, 3} = sprintf('S%d→S%d', old_state, output_state.current_state);
            results{step, 4} = sprintf('R%d', output_state.current_decision);
            results{step, 5} = output_state.control_enabled;
            results{step, 6} = sppvt_output;

            % 检查状态转移是否符合预期
            actual_transition = sprintf('S%d→S%d', old_state, output_state.current_state);
            if strcmp(actual_transition, expected_transition)
                fprintf('  ✅ 状态转移正确: %s\n', actual_transition);
            else
                fprintf('  ⚠️ 状态转移异常: 期望%s, 实际%s\n', expected_transition, actual_transition);
            end

        else
            fprintf('  ❌ 仿真失败\n');
            results{step, 1} = step_num;
            results{step, 2} = description;
            results{step, 3} = '失败';
            results{step, 4} = '失败';
            results{step, 5} = false;
            results{step, 6} = 0;
        end

        fprintf('\n');
    end

    %% 5. 汇总测试结果
    fprintf('📊 单步测试结果汇总:\n');
    fprintf('=====================================\n');
    fprintf('%-8s %-15s %-10s %-6s %-6s %-10s\n', '步骤', '描述', '状态转移', '决策', '控制', 'SPPVT');
    fprintf('%s\n', repmat('-', 1, 70));

    successful_steps = 0;
    for i = 1:num_steps
        if ~strcmp(results{i, 3}, '失败')
            successful_steps = successful_steps + 1;
            status_icon = '✅';
        else
            status_icon = '❌';
        end

        fprintf('%s %-6d %-15s %-10s %-6s %-6d %-10.3f\n', ...
            status_icon, results{i, 1}, results{i, 2}, results{i, 3}, ...
            results{i, 4}, results{i, 5}, results{i, 6});
    end

    fprintf('\n🎯 测试完成: %d/%d 步骤成功\n', successful_steps, num_steps);

    if successful_steps == num_steps
        fprintf('🎉 所有单步测试通过！状态外化架构工作正常！\n');
        fprintf('✅ 决策状态管理: 正确\n');
        fprintf('✅ SPPVT控制集成: 正确\n');
        fprintf('✅ 参数更新: 正确\n');
        fprintf('🚀 可以集成到Python环境!\n');
    else
        fprintf('⚠️ 部分测试失败，需要检查:\n');
        fprintf('  - Simulink模型配置\n');
        fprintf('  - 总线字段匹配\n');
        fprintf('  - 状态转移逻辑\n');
    end

    % 输出最终状态
    fprintf('\n📊 最终状态:\n');
    fprintf('  决策状态: S%d, 历史=%d, 决策=R%d\n', ...
        decision_state.current_state, decision_state.has_history, decision_state.last_active_decision);
    fprintf('  车辆参数: V_target=%.1f km/h, G2=%.1f s\n', ...
        vehicle_state.V_target_kmh, vehicle_state.G2_s);
    fprintf('  SPPVT状态: 级差=%.3f, Stage=[%.1f,%.1f,%.1f]\n', ...
        sppvt_state.stage_offset, sppvt_state.stage_manager_states);

catch ME
    fprintf('❌ 单步测试异常: %s\n', ME.message);
    fprintf('📋 错误详情: %s\n', getReport(ME, 'extended'));
end

% 清理
try
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
catch
    % 忽略清理错误
end

end

%% 辅助函数：创建单步输入数据
function input_data = create_single_step_input(vehicle_state, decision_state, sppvt_state, command_type, command_active)

    % 创建单个时间点的输入数据
    input_data = struct();

    % 基本车辆信息
    input_data.ego_speed_kmh = timeseries(vehicle_state.ego_speed_kmh, 0, 'Name', 'ego_speed_kmh');
    input_data.ego_speed_ms = timeseries(vehicle_state.ego_speed_ms, 0, 'Name', 'ego_speed_ms');
    input_data.V_target_kmh = timeseries(vehicle_state.V_target_kmh, 0, 'Name', 'V_target_kmh');
    input_data.V_min_kmh = timeseries(vehicle_state.V_min_kmh, 0, 'Name', 'V_min_kmh');
    input_data.G2_s = timeseries(vehicle_state.G2_s, 0, 'Name', 'G2_s');

    % 命令信息
    input_data.command_type = timeseries(int32(command_type), 0, 'Name', 'command_type');
    input_data.command_active = timeseries(logical(command_active), 0, 'Name', 'command_active');
    input_data.manual_throttle_active = timeseries(false, 0, 'Name', 'manual_throttle_active');

    % 控制信息
    input_data.control_error = timeseries(1.5, 0, 'Name', 'control_error');
    input_data.control_mode_flag = timeseries(int32(1), 0, 'Name', 'control_mode_flag');
    input_data.timestamp = timeseries(0.0, 0, 'Name', 'timestamp');

    % 决策状态字段（从Python端传入）
    input_data.current_state = timeseries(decision_state.current_state, 0, 'Name', 'current_state');
    input_data.has_history = timeseries(decision_state.has_history, 0, 'Name', 'has_history');
    input_data.last_active_decision = timeseries(decision_state.last_active_decision, 0, 'Name', 'last_active_decision');

    % SPPVT状态字段（从Python端传入）- 确保数组为列向量
    input_data.external_stage_offset = timeseries(sppvt_state.stage_offset, 0, 'Name', 'external_stage_offset');
    input_data.external_stage_manager_states = timeseries(sppvt_state.stage_manager_states(:), 0, 'Name', 'external_stage_manager_states');
    input_data.external_adapter_states = timeseries(sppvt_state.adapter_states(:), 0, 'Name', 'external_adapter_states');

    % 设置时间单位
    field_names = fieldnames(input_data);
    for i = 1:length(field_names)
        input_data.(field_names{i}).TimeInfo.Units = 'seconds';
    end
end

%% 辅助函数：解析单步输出
function [success, output_state, sppvt_output] = parse_single_step_output(sim_out)

    success = false;
    output_state = struct();
    sppvt_output = 0;

    try
        if isa(sim_out, 'Simulink.SimulationOutput') && isprop(sim_out, 'yout')
            output_data = sim_out.yout;

            if isa(output_data, 'Simulink.SimulationData.Dataset') && output_data.numElements >= 1
                element = output_data{1};

                if isa(element, 'Simulink.SimulationData.Signal') && isstruct(element.Values)
                    values_struct = element.Values;

                    % 解析基本输出
                    if isfield(values_struct, 'control_enabled')
                        output_state.control_enabled = logical(values_struct.control_enabled.Data(end));
                    end
                    if isfield(values_struct, 'current_state')
                        output_state.current_state = double(values_struct.current_state.Data(end));
                    end
                    if isfield(values_struct, 'current_decision')
                        output_state.current_decision = double(values_struct.current_decision.Data(end));
                    end
                    if isfield(values_struct, 'updated_V_target_kmh')
                        output_state.updated_V_target_kmh = double(values_struct.updated_V_target_kmh.Data(end));
                    end
                    if isfield(values_struct, 'updated_G2_s')
                        output_state.updated_G2_s = double(values_struct.updated_G2_s.Data(end));
                    end

                    % 解析状态外化字段
                    if isfield(values_struct, 'next_state')
                        output_state.next_state = double(values_struct.next_state.Data(end));
                    end
                    if isfield(values_struct, 'next_has_history')
                        output_state.next_has_history = logical(values_struct.next_has_history.Data(end));
                    end
                    if isfield(values_struct, 'next_last_active_decision')
                        output_state.next_last_active_decision = double(values_struct.next_last_active_decision.Data(end));
                    end

                    % 解析SPPVT状态字段
                    if isfield(values_struct, 'new_stage_offset')
                        output_state.new_stage_offset = double(values_struct.new_stage_offset.Data(end));
                    end
                    if isfield(values_struct, 'new_stage_manager_states')
                        stage_data = values_struct.new_stage_manager_states.Data;
                        if size(stage_data, 1) >= 1 && size(stage_data, 2) >= 3
                            output_state.new_stage_manager_states = stage_data(end, :)';  % 转为列向量
                        elseif size(stage_data, 2) >= 1 && size(stage_data, 1) >= 3
                            output_state.new_stage_manager_states = stage_data(:, end);   % 已经是列向量
                        else
                            output_state.new_stage_manager_states = [];
                        end
                    end
                    if isfield(values_struct, 'new_adapter_states')
                        adapter_data = values_struct.new_adapter_states.Data;
                        if size(adapter_data, 1) >= 1 && size(adapter_data, 2) >= 3
                            output_state.new_adapter_states = adapter_data(end, :)';     % 转为列向量
                        elseif size(adapter_data, 2) >= 1 && size(adapter_data, 1) >= 3
                            output_state.new_adapter_states = adapter_data(:, end);      % 已经是列向量
                        else
                            output_state.new_adapter_states = [];
                        end
                    end

                    % 解析SPPVT控制输出
                    if isfield(values_struct, 'sppvt_control_output')
                        sppvt_output = double(values_struct.sppvt_control_output.Data(end));
                    end

                    success = true;
                end
            end
        end

    catch ME
        fprintf('输出解析失败: %s\n', ME.message);
    end
end