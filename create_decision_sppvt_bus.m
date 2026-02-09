function create_decision_sppvt_bus()
%CREATE_DECISION_SPPVT_BUS 创建ACC决策+SPPVT一体化总线定义 (状态外化精简版本)
% DEPRECATED / 已废弃: 当前集成UDP接口不再使用本文件生成的总线。
% 保留仅供参考，请勿修改或作为后续分析依据。

%
%   此函数创建以下总线：
%   - DecisionSPPVTInputExtended: 21个输入字段 (原有11个 + 3个决策状态 + 7个SPPVT状态字段-全部标量化)
%   - DecisionSPPVTOutputExtended: 22个输出字段 (原有12个 + 3个决策状态输出 + 7个SPPVT状态输出字段-全部标量化)


    fprintf('创建ACC决策+SPPVT一体化总线定义...\n');
    
    % 清理现有总线定义
    try
        clear DecisionSPPVTInput DecisionSPPVTOutput DecisionSPPVTInputExtended DecisionSPPVTOutputExtended;
    catch
        % 忽略清理错误
    end

    %% 创建扩展输入总线：DecisionSPPVTInputExtended (21个字段)
    fprintf('创建扩展输入总线 DecisionSPPVTInputExtended (状态外化标量版本)...\n');

    % 使用MATLAB 2024b的新语法创建总线对象
    DecisionSPPVTInputExtended = Simulink.Bus;
    DecisionSPPVTInputExtended.HeaderFile = '';
    DecisionSPPVTInputExtended.Description = 'ACC决策+SPPVT一体化输入数据总线 (状态外化标量版本 - 21字段)';
    DecisionSPPVTInputExtended.DataScope = 'Auto';
    DecisionSPPVTInputExtended.Alignment = -1;

    % 定义输入总线元素 - 包含原有11个字段 + 3个决策状态字段 + 7个SPPVT状态字段(全部标量化)
    inputElements = [
        % 原有的11个字段
        createBusElement('ego_speed_kmh', 'double', 1, 'km/h', '自车速度(km/h)'),
        createBusElement('ego_speed_ms', 'double', 1, 'm/s', '自车速度(m/s)'),
        createBusElement('command_type', 'int32', 1, '1', '指令类型: 0=NONE, 1=I0, 2=I1, ..., 7=I6'),
        createBusElement('command_active', 'boolean', 1, '1', '指令是否激活'),
        createBusElement('manual_throttle_active', 'boolean', 1, '1', '手动油门是否激活'),
        createBusElement('control_error', 'double', 1, 'm', 'Two Mode计算的控制误差'),
        createBusElement('control_mode_flag', 'int32', 1, '1', '控制模式标志: 1=time, 2=speed'),
        createBusElement('V_target_kmh', 'double', 1, 'km/h', '目标速度'),
        createBusElement('V_min_kmh', 'double', 1, 'km/h', '最小速度'),
        createBusElement('G2_s', 'double', 1, 's', '时距参数'),
        createBusElement('timestamp', 'double', 1, 's', '时间戳'),

        % 决策状态外化字段 - 替代persistent变量
        createBusElement('current_state', 'int32', 1, '1', '当前状态: 0=S0, 1=S1, 2=S2, 3=S3'),
        createBusElement('has_history', 'boolean', 1, '1', '是否有历史数据'),
        createBusElement('last_active_decision', 'int32', 1, '1', '最后有效决策: 1=R1...8=R8'),

        % SPPVT状态外化字段 - 全部标量化
        createBusElement('external_stage_offset', 'double', 1, '1', '外部状态: Stage_Manager级差值'),
        createBusElement('external_stage', 'double', 1, '1', '外部状态: Stage_Manager当前阶段'),
        createBusElement('external_error_sign', 'double', 1, '1', '外部状态: 误差符号'),
        createBusElement('external_upgrade_count', 'double', 1, '1', '外部状态: 升级计数'),
        createBusElement('external_control_error', 'double', 1, 'm', '外部状态: 控制误差'),
        createBusElement('external_error_derivative', 'double', 1, 'm/s', '外部状态: 控制误差的导数'),
        createBusElement('external_error_second_derivative', 'double', 1, 'm/s^2', '外部状态: 控制误差的二阶导数')
    ];
    
    DecisionSPPVTInputExtended.Elements = inputElements;

    %% 创建扩展输出总线：DecisionSPPVTOutputExtended (22个字段)
    fprintf('创建扩展输出总线 DecisionSPPVTOutputExtended (状态外化标量版本)...\n');

    DecisionSPPVTOutputExtended = Simulink.Bus;
    DecisionSPPVTOutputExtended.HeaderFile = '';
    DecisionSPPVTOutputExtended.Description = 'ACC决策+SPPVT一体化输出数据总线 (状态外化标量版本 - 22字段)';
    DecisionSPPVTOutputExtended.DataScope = 'Auto';
    DecisionSPPVTOutputExtended.Alignment = -1;

    % 定义输出总线元素 - 包含原有12个字段 + 3个决策状态输出字段 + 7个SPPVT状态输出字段(全部标量化)
    outputElements = [
        % 原有的12个字段 - 来自Decision_Function和SPPVT模块
        createBusElement('control_enabled', 'boolean', 1, '1', '控制使能状态'),
        createBusElement('current_state', 'int32', 1, '1', '当前状态: 0=S0, 1=S1, 2=S2, 3=S3'),
        createBusElement('current_decision', 'int32', 1, '1', '当前决策: 0=NONE, 1=R1, ..., 8=R8'),
        createBusElement('torque_arbitration_active', 'boolean', 1, '1', '扭矩仲裁激活状态'),
        createBusElement('updated_V_target_kmh', 'double', 1, 'km/h', '更新后的目标速度'),
        createBusElement('updated_G2_s', 'double', 1, 's', '更新后的时距参数'),
        createBusElement('sppvt_control_output', 'double', 1, 'Nm(无量纲)', 'SPPVT主控制输出-扭矩'),
        createBusElement('sppvt_velocity_output', 'double', 1, '1/s', 'SPPVT误差一阶导数'),
        createBusElement('sppvt_acceleration_output', 'double', 1, '1/s^2', 'SPPVT误差二阶导数'),
        createBusElement('sppvt_stage_output', 'double', 1, '1', 'SPPVT阶段输出'),
        createBusElement('sppvt_status_output', 'double', 1, '1', 'SPPVT状态输出'),
        createBusElement('debug_message', 'int32', 1, '1', '调试信息代码'),

        % 决策状态输出字段 - 下一个状态信息
        createBusElement('next_state', 'int32', 1, '1', '下一个状态: 0=S0, 1=S1, 2=S2, 3=S3'),
        createBusElement('next_has_history', 'boolean', 1, '1', '下一个历史标志'),
        createBusElement('next_last_active_decision', 'int32', 1, '1', '下一个有效决策: 1=R1...8=R8'),

        % SPPVT状态输出字段 - 全部标量化
        createBusElement('new_stage_offset', 'double', 1, '1', '更新后状态: 新Stage_Manager级差值'),
        createBusElement('new_stage', 'double', 1, '1', '更新后状态: 新Stage_Manager阶段'),
        createBusElement('new_error_sign', 'double', 1, '1', '更新后状态: 新误差符号'),
        createBusElement('new_upgrade_count', 'double', 1, '1', '更新后状态: 新升级计数'),
        createBusElement('new_control_error', 'double', 1, 'm', '更新后状态: 新控制误差'),
        createBusElement('new_error_derivative', 'double', 1, 'm/s', '更新后状态: 新控制误差的导数'),
        createBusElement('new_error_second_derivative', 'double', 1, 'm/s^2', '更新后状态: 新控制误差的二阶导数')
    ];

    DecisionSPPVTOutputExtended.Elements = outputElements;
    
    %% 保存总线定义到基础工作区
    fprintf('保存扩展总线定义到基础工作区...\n');

    % 使用MATLAB 2024b的assignin语法 - 只保存扩展版本
    assignin('base', 'DecisionSPPVTInputExtended', DecisionSPPVTInputExtended);
    assignin('base', 'DecisionSPPVTOutputExtended', DecisionSPPVTOutputExtended);

    % 明确删除旧总线定义，避免混乱
    try
        evalin('base', 'clear DecisionSPPVTInput DecisionSPPVTOutput');
        fprintf('✅ 已清除旧总线定义，强制使用新名称\n');
    catch
        % 忽略清理错误
    end

    %% 创建总线定义文件 (可选，用于持久化)
    try
        fprintf('生成扩展总线定义.mat文件...\n');
        save('DecisionSPPVTBusDefinitions.mat', 'DecisionSPPVTInputExtended', 'DecisionSPPVTOutputExtended');
        fprintf('扩展总线定义已保存到: DecisionSPPVTBusDefinitions.mat\n');
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end

    %% 显示创建结果
    fprintf('\n✅ 状态外化总线定义创建完成!\n');
    fprintf('📁 精简输入总线: DecisionSPPVTInputExtended (%d个字段)\n', length(inputElements));
    fprintf('📁 精简输出总线: DecisionSPPVTOutputExtended (%d个字段)\n', length(outputElements));
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
    
    fprintf('\n💡 使用方法 (状态外化版本):\n');
    fprintf('   1. 在Simulink模型中使用Bus Creator/Bus Selector\n');
    fprintf('   2. 设置数据类型为 ''Bus: DecisionSPPVTInputExtended'' 或 ''Bus: DecisionSPPVTOutputExtended''\n');
    fprintf('   3. MATLAB Function块中使用: function output = fcn(input)\n');
    fprintf('   4. Python调用: create_decision_sppvt_bus() 创建扩展总线\n');
    fprintf('   5. 状态外化设计用于CARLA单步调用集成\n');

    % 显示状态外化的新增字段信息
    fprintf('\n🆕 新增决策状态字段 (输入总线):\n');
    fprintf('   12. current_state                | 当前决策状态: 0=S0, 1=S1, 2=S2, 3=S3\n');
    fprintf('   13. has_history                  | 是否有历史数据 (boolean)\n');
    fprintf('   14. last_active_decision         | 最后有效决策: 1=R1...8=R8\n');

    fprintf('\n🆕 新增SPPVT状态字段 (输入总线 - 全部标量化):\n');
    fprintf('   15. external_stage_offset        | Stage_Manager级差值\n');
    fprintf('   16. external_stage               | Stage_Manager当前阶段\n');
    fprintf('   17. external_error_sign          | 误差符号\n');
    fprintf('   18. external_upgrade_count       | 升级计数\n');
    fprintf('   19. external_control_error       | 控制误差\n');
    fprintf('   20. external_error_derivative    | 控制误差的导数\n');
    fprintf('   21. external_error_second_derivative | 控制误差的二阶导数\n');

    fprintf('\n🔄 新增决策状态输出字段 (输出总线):\n');
    fprintf('   13. next_state                   | 下一个状态: 0=S0, 1=S1, 2=S2, 3=S3\n');
    fprintf('   14. next_has_history             | 下一个历史标志 (boolean)\n');
    fprintf('   15. next_last_active_decision    | 下一个有效决策: 1=R1...8=R8\n');

    fprintf('\n🔄 新增SPPVT状态输出字段 (输出总线 - 全部标量化):\n');
    fprintf('   16. new_stage_offset             | 更新后Stage_Manager级差值\n');
    fprintf('   17. new_stage                    | 新Stage_Manager阶段\n');
    fprintf('   18. new_error_sign               | 新误差符号\n');
    fprintf('   19. new_upgrade_count            | 新升级计数\n');
    fprintf('   20. new_control_error            | 新控制误差\n');
    fprintf('   21. new_error_derivative         | 新控制误差的导数\n');
    fprintf('   22. new_error_second_derivative  | 新控制误差的二阶导数\n');

    fprintf('\n🔧 状态外化标量设计要点:\n');
    fprintf('   - Python维护决策状态：current_state + has_history + last_active_decision\n');
    fprintf('   - Python维护SPPVT状态：7个独立标量字段（全部标量化）\n');
    fprintf('   - 保持现有17模块架构基本不变\n');
    fprintf('   - 所有字段均为标量，避免数组访问问题\n');
    fprintf('   - 提高Simulink总线数据访问的稳定性\n');
    fprintf('   - 解决CARLA单步调用的状态持久化问题\n');

    fprintf('\n🎯 支持的算法特性:\n');
    fprintf('   - SPPVT阶段递进升级算法\n');
    fprintf('   - 误差符号变化检测和重置\n');
    fprintf('   - 级差动态计算和符号自适应\n');
    fprintf('   - 完整的ACC决策状态机\n');

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
        % 测试扩展输入总线 (21个字段 - 全部标量化)
        testInputExtended = struct();
        % 原有的11个字段
        testInputExtended.ego_speed_kmh = 50.0;
        testInputExtended.ego_speed_ms = 13.89;
        testInputExtended.command_type = int32(1);
        testInputExtended.command_active = true;
        testInputExtended.manual_throttle_active = false;
        testInputExtended.control_error = 0.5;
        testInputExtended.control_mode_flag = int32(1);
        testInputExtended.V_target_kmh = 50.0;
        testInputExtended.V_min_kmh = 30.0;
        testInputExtended.G2_s = 2.0;
        testInputExtended.timestamp = now();

        % 新增的3个决策状态字段 (状态外化版本)
        testInputExtended.current_state = int32(1);
        testInputExtended.has_history = true;
        testInputExtended.last_active_decision = int32(5);

        % 新增的7个SPPVT外部状态字段 (全部标量化)
        testInputExtended.external_stage_offset = 0.15;
        testInputExtended.external_stage = 2.0;
        testInputExtended.external_error_sign = 1.0;
        testInputExtended.external_upgrade_count = 3.0;
        testInputExtended.external_control_error = 0.3;
        testInputExtended.external_error_derivative = 0.05;
        testInputExtended.external_error_second_derivative = 0.01;

        fprintf('✅ 标量化输入总线测试数据创建成功 (21个字段)\n');

        % 测试扩展输出总线 (22个字段 - 全部标量化)
        testOutputExtended = struct();
        % 原有的12个字段
        testOutputExtended.control_enabled = true;
        testOutputExtended.current_state = int32(1);
        testOutputExtended.current_decision = int32(2);
        testOutputExtended.torque_arbitration_active = false;
        testOutputExtended.updated_V_target_kmh = 50.0;
        testOutputExtended.updated_G2_s = 2.0;
        testOutputExtended.sppvt_control_output = 0.8;
        testOutputExtended.sppvt_velocity_output = 13.5;
        testOutputExtended.sppvt_acceleration_output = 0.3;
        testOutputExtended.sppvt_stage_output = 1.2;
        testOutputExtended.sppvt_status_output = 1.0;
        testOutputExtended.debug_message = int32(100);

        % 新增的3个决策状态输出字段 (状态外化版本)
        testOutputExtended.next_state = int32(0);
        testOutputExtended.next_has_history = true;
        testOutputExtended.next_last_active_decision = int32(5);

        % 新增的7个SPPVT状态输出字段 (全部标量化)
        testOutputExtended.new_stage_offset = 0.2;
        testOutputExtended.new_stage = 3.0;
        testOutputExtended.new_error_sign = 1.0;
        testOutputExtended.new_upgrade_count = 4.0;
        testOutputExtended.new_control_error = 0.4;
        testOutputExtended.new_error_derivative = 0.06;
        testOutputExtended.new_error_second_derivative = 0.012;

        fprintf('✅ 标量化输出总线测试数据创建成功 (22个字段)\n');

        % 保存测试数据
        assignin('base', 'testDecisionSPPVTInputExtended', testInputExtended);
        assignin('base', 'testDecisionSPPVTOutputExtended', testOutputExtended);

        % 保持兼容性，同时使用原名称保存
        assignin('base', 'testDecisionSPPVTInput', testInputExtended);
        assignin('base', 'testDecisionSPPVTOutput', testOutputExtended);
        
        fprintf('✅ 测试数据已保存到基础工作区\n');
        fprintf('   - testDecisionSPPVTInput\n');
        fprintf('   - testDecisionSPPVTOutput\n');
        
    catch ME
        fprintf('❌ 总线定义测试失败: %s\n', ME.message);
        rethrow(ME);
    end
    
    fprintf('🎉 总线定义测试完成!\n\n');
end
