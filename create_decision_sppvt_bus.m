function create_decision_sppvt_bus()
%CREATE_DECISION_SPPVT_BUS 创建ACC决策+SPPVT一体化总线定义

%   
%   此函数创建以下总线：
%   - DecisionSPPVTInput: 11个输入字段
%   - DecisionSPPVTOutput: 10个输出字段


    fprintf('创建ACC决策+SPPVT一体化总线定义...\n');
    
    % 清理现有总线定义
    try
        clear DecisionSPPVTInput DecisionSPPVTOutput;
    catch
        % 忽略清理错误
    end
    
    %% 创建输入总线：DecisionSPPVTInput (11个字段)
    fprintf('创建输入总线 DecisionSPPVTInput...\n');
    
    % 使用MATLAB 2024b的新语法创建总线对象
    DecisionSPPVTInput = Simulink.Bus;
    DecisionSPPVTInput.HeaderFile = '';
    DecisionSPPVTInput.Description = 'ACC决策+SPPVT一体化输入数据总线';
    DecisionSPPVTInput.DataScope = 'Auto';
    DecisionSPPVTInput.Alignment = -1;
    
    % 定义输入总线元素 - 使用MATLAB 2024b语法
    inputElements = [
        createBusElement('ego_speed_kmh', 'double', 1, '1', '自车速度(km/h)'),
        createBusElement('ego_speed_ms', 'double', 1, 'm/s', '自车速度(m/s)'),
        createBusElement('command_type', 'int32', 1, '1', '指令类型: 0=NONE, 1=I0, 2=I1, ..., 7=I6'),
        createBusElement('command_active', 'boolean', 1, '1', '指令是否激活'),
        createBusElement('manual_throttle_active', 'boolean', 1, '1', '手动油门是否激活'),
        createBusElement('control_error', 'double', 1, 'm', 'Two Mode计算的控制误差'),
        createBusElement('control_mode_flag', 'int32', 1, '1', '控制模式标志: 1=distance, 2=speed'),
        createBusElement('V_target_kmh', 'double', 1, '1', '目标速度'),
        createBusElement('V_min_kmh', 'double', 1, '1', '最小速度'),
        createBusElement('G2_s', 'double', 1, 's', '时距参数'),
        createBusElement('timestamp', 'double', 1, 's', '时间戳')
    ];
    
    DecisionSPPVTInput.Elements = inputElements;
    
    %% 创建输出总线：DecisionSPPVTOutput (10个字段)
    fprintf('创建输出总线 DecisionSPPVTOutput...\n');
    
    DecisionSPPVTOutput = Simulink.Bus;
    DecisionSPPVTOutput.HeaderFile = '';
    DecisionSPPVTOutput.Description = 'ACC决策+SPPVT一体化输出数据总线';
    DecisionSPPVTOutput.DataScope = 'Auto';
    DecisionSPPVTOutput.Alignment = -1;
    
    % 定义输出总线元素 - 匹配实际模块输出
    outputElements = [
        % 来自Decision_Function的字段
        createBusElement('control_enabled', 'boolean', 1, '1', '控制使能状态'),
        createBusElement('current_state', 'int32', 1, '1', '当前状态: 0=S0, 1=S1, 2=S2, 3=S3'),
        createBusElement('current_decision', 'int32', 1, '1', '当前决策: 0=NONE, 1=R1, ..., 8=R8'),
        createBusElement('torque_arbitration_active', 'boolean', 1, '1', '扭矩仲裁激活状态'),
        createBusElement('updated_V_target_kmh', 'double', 1, '1', '更新后的目标速度'),
        createBusElement('updated_G2_s', 'double', 1, 's', '更新后的时距参数'),
        createBusElement('debug_message', 'int32', 1, '1', '调试信息代码'),
        % 来自SPPVT模块的5个输出 - 匹配实际输出
        createBusElement('sppvt_control_output', 'double', 1, '1', 'SPPVT主控制输出'),
        createBusElement('sppvt_velocity_output', 'double', 1, '1', 'SPPVT速度输出'),
        createBusElement('sppvt_acceleration_output', 'double', 1, '1', 'SPPVT加速度输出'),
        createBusElement('sppvt_stage_output', 'double', 1, '1', 'SPPVT级差输出'),
        createBusElement('sppvt_status_output', 'double', 1, '1', 'SPPVT状态输出')
    ];
    
    DecisionSPPVTOutput.Elements = outputElements;
    
    %% 保存总线定义到基础工作区
    fprintf('保存总线定义到基础工作区...\n');
    
    % 使用MATLAB 2024b的assignin语法
    assignin('base', 'DecisionSPPVTInput', DecisionSPPVTInput);
    assignin('base', 'DecisionSPPVTOutput', DecisionSPPVTOutput);
    
    %% 创建总线定义文件 (可选，用于持久化)
    try
        fprintf('生成总线定义.mat文件...\n');
        save('DecisionSPPVTBusDefinitions.mat', 'DecisionSPPVTInput', 'DecisionSPPVTOutput');
        fprintf('总线定义已保存到: DecisionSPPVTBusDefinitions.mat\n');
    catch ME
        warning('保存总线定义文件失败: %s', ME.message);
    end
    
    %% 显示创建结果
    fprintf('\n✅ 总线定义创建完成!\n');
    fprintf('📁 输入总线: DecisionSPPVTInput (%d个字段)\n', length(inputElements));
    fprintf('📁 输出总线: DecisionSPPVTOutput (%d个字段)\n', length(outputElements));
    fprintf('\n🔍 总线详细信息:\n');
    
    % 显示输入总线信息
    fprintf('\n📥 输入总线字段:\n');
    for i = 1:length(inputElements)
        elem = inputElements(i);
        fprintf('  %2d. %-25s | %8s | %s\n', i, elem.Name, elem.DataType, elem.Description);
    end
    
    % 显示输出总线信息  
    fprintf('\n📤 输出总线字段:\n');
    for i = 1:length(outputElements)
        elem = outputElements(i);
        fprintf('  %2d. %-25s | %8s | %s\n', i, elem.Name, elem.DataType, elem.Description);
    end
    
    fprintf('\n💡 使用方法:\n');
    fprintf('   1. 在Simulink模型中使用Bus Creator/Bus Selector\n');
    fprintf('   2. 设置数据类型为 ''Bus: DecisionSPPVTInput'' 或 ''Bus: DecisionSPPVTOutput''\n');
    fprintf('   3. MATLAB Function块中使用: function output = fcn(input)\n');
    fprintf('\n');
end

function element = createBusElement(name, dataType, dimensions, units, description)
%CREATEBUSELEMENT 创建总线元素的辅助函数
%   使用MATLAB 2024b的新语法创建Simulink.BusElement
    
    element = Simulink.BusElement;
    element.Name = name;
    element.DataType = dataType;
    element.Dimensions = dimensions;
    element.DimensionsMode = 'Fixed';
    element.SampleTime = -1;  % 继承采样时间
    element.Complexity = 'real';
    element.SamplingMode = 'Sample based';
    element.DocUnits = units;
    element.Description = description;
    
    % 根据数据类型和字段名称设置合理的Min/Max值
    switch dataType
        case 'double'
            % 根据具体字段设置合理范围
            if contains(name, 'speed')
                element.Min = 0.0;      % 速度不能为负
                element.Max = 200.0;    % 最大速度200 km/h 或 m/s
            elseif contains(name, 'error')
                element.Min = -50.0;    % 控制误差范围
                element.Max = 50.0;
            elseif contains(name, 'G2')
                element.Min = 0.0;      % 时距参数合理范围（允许0初始化）
                element.Max = 5.0;
            elseif contains(name, 'accel')
                element.Min = -10.0;    % 加速度范围
                element.Max = 5.0;
            elseif contains(name, 'timestamp')
                element.Min = 0.0;      % 时间戳
                element.Max = 1e10;
            else
                element.Min = -1000.0;  % 默认范围
                element.Max = 1000.0;
            end
            
        case 'int32'
            % 整数类型的合理范围 - 注意：Min/Max必须是double类型
            if contains(name, 'command') || contains(name, 'state') || contains(name, 'decision')
                element.Min = 0.0;      % 枚举值从0开始
                element.Max = 10.0;     % 最大枚举值
            elseif contains(name, 'flag')
                element.Min = 0.0;      % 标志允许0初始化
                element.Max = 3.0;
            elseif contains(name, 'count')
                element.Min = 0.0;      % 计数从0开始
                element.Max = 1000.0;
            elseif contains(name, 'message')
                element.Min = 0.0;      % 消息代码
                element.Max = 9999.0;
            else
                element.Min = -100.0;   % 默认范围
                element.Max = 100.0;
            end
            
        case 'boolean'
            % boolean类型的Min/Max也必须是double
            element.Min = 0.0;
            element.Max = 1.0;
            
        otherwise
            % 对于其他数据类型，设置合理的默认范围
            element.Min = -1000.0;
            element.Max = 1000.0;
    end
end

%% 使用示例和测试代码
function runBusDefinitionTest()
%RUNBUSDEFINITIONTEST 测试总线定义的完整性
    
    fprintf('\n🧪 开始总线定义测试...\n');
    
    try
        % 测试输入总线
        testInput = struct();
        testInput.ego_speed_kmh = 50.0;
        testInput.ego_speed_ms = 13.89;
        testInput.command_type = int32(1);
        testInput.command_active = true;
        testInput.manual_throttle_active = false;
        testInput.control_error = 0.5;
        testInput.control_mode_flag = int32(1);
        testInput.V_target_kmh = 50.0;
        testInput.V_min_kmh = 30.0;
        testInput.G2_s = 2.0;
        testInput.timestamp = now();
        
        fprintf('✅ 输入总线测试数据创建成功\n');
        
        % 测试输出总线
        testOutput = struct();
        testOutput.target_accel = 0.5;
        testOutput.control_enabled = true;
        testOutput.current_state = int32(1);
        testOutput.current_decision = int32(2);
        testOutput.torque_arbitration_active = false;
        testOutput.updated_V_target_kmh = 50.0;
        testOutput.updated_G2_s = 2.0;
        testOutput.sppvt_stage = int32(1);
        testOutput.sppvt_upgrade_count = int32(0);
        testOutput.debug_message = int32(100);
        
        fprintf('✅ 输出总线测试数据创建成功\n');
        
        % 保存测试数据
        assignin('base', 'testDecisionSPPVTInput', testInput);
        assignin('base', 'testDecisionSPPVTOutput', testOutput);
        
        fprintf('✅ 测试数据已保存到基础工作区\n');
        fprintf('   - testDecisionSPPVTInput\n');
        fprintf('   - testDecisionSPPVTOutput\n');
        
    catch ME
        fprintf('❌ 总线定义测试失败: %s\n', ME.message);
        rethrow(ME);
    end
    
    fprintf('🎉 总线定义测试完成!\n\n');
end