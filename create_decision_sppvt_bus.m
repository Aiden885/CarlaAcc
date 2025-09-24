function create_decision_sppvt_bus()
%CREATE_DECISION_SPPVT_BUS 创建ACC决策+SPPVT一体化总线定义 (状态外化版本)

%
%   此函数创建以下总线：
%   - DecisionSPPVTInputExtended: 18个输入字段 (原有11个 + 7个外部状态字段)
%   - DecisionSPPVTOutputExtended: 19个输出字段 (原有12个 + 7个状态输出字段)
%
%   状态外化设计：
%   - Python维护所有persistent状态和Unit Delay状态
%   - Simulink成为无状态的纯计算模块
%   - 通过输入输出端口实现状态的传入和回传


    fprintf('创建ACC决策+SPPVT一体化总线定义...\n');
    
    % 清理现有总线定义
    try
        clear DecisionSPPVTInput DecisionSPPVTOutput DecisionSPPVTInputExtended DecisionSPPVTOutputExtended;
    catch
        % 忽略清理错误
    end

    %% 创建扩展输入总线：DecisionSPPVTInputExtended (18个字段)
    fprintf('创建扩展输入总线 DecisionSPPVTInputExtended (状态外化版本)...\n');

    % 使用MATLAB 2024b的新语法创建总线对象
    DecisionSPPVTInputExtended = Simulink.Bus;
    DecisionSPPVTInputExtended.HeaderFile = '';
    DecisionSPPVTInputExtended.Description = 'ACC决策+SPPVT一体化输入数据总线 (状态外化扩展版本 - 18字段)';
    DecisionSPPVTInputExtended.DataScope = 'Auto';
    DecisionSPPVTInputExtended.Alignment = -1;

    % 定义输入总线元素 - 包含原有11个字段 + 新增7个外部状态字段
    inputElements = [
        % 原有的11个字段
        createBusElement('ego_speed_kmh', 'double', 1, 'km/h', '自车速度(km/h)'),
        createBusElement('ego_speed_ms', 'double', 1, 'm/s', '自车速度(m/s)'),
        createBusElement('command_type', 'int32', 1, '1', '指令类型: 0=NONE, 1=I0, 2=I1, ..., 7=I6'),
        createBusElement('command_active', 'boolean', 1, '1', '指令是否激活'),
        createBusElement('manual_throttle_active', 'boolean', 1, '1', '手动油门是否激活'),
        createBusElement('control_error', 'double', 1, 'm', 'Two Mode计算的控制误差'),
        createBusElement('control_mode_flag', 'int32', 1, '1', '控制模式标志: 1=distance, 2=speed'),
        createBusElement('V_target_kmh', 'double', 1, 'km/h', '目标速度'),
        createBusElement('V_min_kmh', 'double', 1, 'km/h', '最小速度'),
        createBusElement('G2_s', 'double', 1, 's', '时距参数'),
        createBusElement('timestamp', 'double', 1, 's', '时间戳'),

        % 新增的7个外部状态字段 - 用于状态外化的完整SPPVT状态管理
        % Stage_Manager状态 (4个字段)
        createBusElement('prev_stage', 'int32', 1, '1', '外部Stage_Manager状态: 当前阶段'),
        createBusElement('prev_error_sign', 'int32', 1, '1', '外部Stage_Manager状态: 前误差符号 (-1,0,1)'),
        createBusElement('prev_upgrade_count', 'int32', 1, '1', '外部Stage_Manager状态: 升级计数'),
        createBusElement('prev_stage_offset', 'double', 1, '1', '外部Stage_Manager状态: 级差值'),

        % SPPVT_Adapter状态 (3个字段)
        createBusElement('prev_control_error', 'double', 1, 'm', '外部SPPVT_Adapter状态: 前控制误差'),
        createBusElement('prev_velocity', 'double', 1, 'm/s', '外部SPPVT_Adapter状态: 前速度'),
        createBusElement('prev_accel', 'double', 1, 'm/s²', '外部SPPVT_Adapter状态: 前加速度')
    ];
    
    DecisionSPPVTInputExtended.Elements = inputElements;

    %% 创建扩展输出总线：DecisionSPPVTOutputExtended (19个字段)
    fprintf('创建扩展输出总线 DecisionSPPVTOutputExtended (状态外化版本)...\n');

    DecisionSPPVTOutputExtended = Simulink.Bus;
    DecisionSPPVTOutputExtended.HeaderFile = '';
    DecisionSPPVTOutputExtended.Description = 'ACC决策+SPPVT一体化输出数据总线 (状态外化扩展版本 - 19字段)';
    DecisionSPPVTOutputExtended.DataScope = 'Auto';
    DecisionSPPVTOutputExtended.Alignment = -1;

    % 定义输出总线元素 - 包含原有12个字段 + 新增7个状态输出字段
    outputElements = [
        % 原有的12个字段 - 来自Decision_Function和SPPVT模块
        createBusElement('control_enabled', 'boolean', 1, '1', '控制使能状态'),
        createBusElement('current_state', 'int32', 1, '1', '当前状态: 0=S0, 1=S1, 2=S2, 3=S3'),
        createBusElement('current_decision', 'int32', 1, '1', '当前决策: 0=NONE, 1=R1, ..., 8=R8'),
        createBusElement('torque_arbitration_active', 'boolean', 1, '1', '扭矩仲裁激活状态'),
        createBusElement('updated_V_target_kmh', 'double', 1, 'km/h', '更新后的目标速度'),
        createBusElement('updated_G2_s', 'double', 1, 's', '更新后的时距参数'),
        createBusElement('sppvt_control_output', 'double', 1, 'm/s²', 'SPPVT主控制输出'),
        createBusElement('sppvt_velocity_output', 'double', 1, 'm/s', 'SPPVT速度输出'),
        createBusElement('sppvt_acceleration_output', 'double', 1, 'm/s²', 'SPPVT加速度输出'),
        createBusElement('sppvt_stage_output', 'double', 1, '1', 'SPPVT阶段输出'),
        createBusElement('sppvt_status_output', 'double', 1, '1', 'SPPVT状态输出'),
        createBusElement('debug_message', 'int32', 1, '1', '调试信息代码'),

        % 新增的7个状态输出字段 - 用于状态外化的完整SPPVT状态回传
        % Stage_Manager状态输出 (4个字段)
        createBusElement('new_stage', 'int32', 1, '1', '更新后的Stage_Manager状态: 新阶段'),
        createBusElement('new_error_sign', 'int32', 1, '1', '更新后的Stage_Manager状态: 新误差符号'),
        createBusElement('new_upgrade_count', 'int32', 1, '1', '更新后的Stage_Manager状态: 新升级计数'),
        createBusElement('new_stage_offset', 'double', 1, '1', '更新后的Stage_Manager状态: 新级差值'),

        % SPPVT_Adapter状态输出 (3个字段)
        createBusElement('new_control_error', 'double', 1, 'm', '更新后的SPPVT_Adapter状态: 新控制误差'),
        createBusElement('new_velocity', 'double', 1, 'm/s', '更新后的SPPVT_Adapter状态: 新速度'),
        createBusElement('new_accel', 'double', 1, 'm/s²', '更新后的SPPVT_Adapter状态: 新加速度')
    ];

    DecisionSPPVTOutputExtended.Elements = outputElements;
    
    %% 保存总线定义到基础工作区
    fprintf('保存扩展总线定义到基础工作区...\n');

    % 使用MATLAB 2024b的assignin语法 - 保存扩展版本
    assignin('base', 'DecisionSPPVTInputExtended', DecisionSPPVTInputExtended);
    assignin('base', 'DecisionSPPVTOutputExtended', DecisionSPPVTOutputExtended);

    % 同时保存原名称以保持兼容性
    assignin('base', 'DecisionSPPVTInput', DecisionSPPVTInputExtended);
    assignin('base', 'DecisionSPPVTOutput', DecisionSPPVTOutputExtended);

    %% 创建总线定义文件 (可选，用于持久化)
    try
        fprintf('生成扩展总线定义.mat文件...\n');
        save('DecisionSPPVTBusDefinitions.mat', 'DecisionSPPVTInputExtended', 'DecisionSPPVTOutputExtended', ...
             'DecisionSPPVTInput', 'DecisionSPPVTOutput');
        fprintf('扩展总线定义已保存到: DecisionSPPVTBusDefinitions.mat\n');
    catch ME
        warning('保存总线定义文件失败: %s', ME.message);
    end

    %% 显示创建结果
    fprintf('\n✅ 状态外化总线定义创建完成!\n');
    fprintf('📁 扩展输入总线: DecisionSPPVTInputExtended (%d个字段)\n', length(inputElements));
    fprintf('📁 扩展输出总线: DecisionSPPVTOutputExtended (%d个字段)\n', length(outputElements));
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

    % 显示状态外化的完整字段信息
    fprintf('\n🆕 新增外部状态字段 (输入总线 - 7个字段):\n');
    fprintf('   12. prev_stage              | Stage_Manager状态: 当前阶段\n');
    fprintf('   13. prev_error_sign         | Stage_Manager状态: 前误差符号\n');
    fprintf('   14. prev_upgrade_count      | Stage_Manager状态: 升级计数\n');
    fprintf('   15. prev_stage_offset       | Stage_Manager状态: 级差值\n');
    fprintf('   16. prev_control_error      | SPPVT_Adapter状态: 前控制误差\n');
    fprintf('   17. prev_velocity           | SPPVT_Adapter状态: 前速度\n');
    fprintf('   18. prev_accel              | SPPVT_Adapter状态: 前加速度\n');

    fprintf('\n🔄 新增状态输出字段 (输出总线 - 7个字段):\n');
    fprintf('   13. new_stage               | Stage_Manager状态: 新阶段\n');
    fprintf('   14. new_error_sign          | Stage_Manager状态: 新误差符号\n');
    fprintf('   15. new_upgrade_count       | Stage_Manager状态: 新升级计数\n');
    fprintf('   16. new_stage_offset        | Stage_Manager状态: 新级差值\n');
    fprintf('   17. new_control_error       | SPPVT_Adapter状态: 新控制误差\n');
    fprintf('   18. new_velocity            | SPPVT_Adapter状态: 新速度\n');
    fprintf('   19. new_accel               | SPPVT_Adapter状态: 新加速度\n');

    fprintf('\n🔧 状态外化关键技术点:\n');
    fprintf('   - Python维护所有persistent变量和Unit Delay状态\n');
    fprintf('   - Simulink变为无状态的纯函数计算模块\n');
    fprintf('   - 每次sim()调用通过输入端口注入历史状态\n');
    fprintf('   - 通过输出端口回传更新后的状态给Python\n');
    fprintf('   - 完美解决CARLA单步调用的状态连续性问题\n');

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
        % 测试扩展输入总线 (18个字段)
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

        % 新增的7个外部状态字段 (状态外化版本)
        % Stage_Manager状态 (4个字段)
        testInputExtended.prev_stage = int32(2);
        testInputExtended.prev_error_sign = int32(1);
        testInputExtended.prev_upgrade_count = int32(3);
        testInputExtended.prev_stage_offset = 0.15;

        % SPPVT_Adapter状态 (3个字段)
        testInputExtended.prev_control_error = 0.3;
        testInputExtended.prev_velocity = 13.89;
        testInputExtended.prev_accel = 0.2;

        fprintf('✅ 扩展输入总线测试数据创建成功 (18个字段)\n');

        % 测试扩展输出总线 (19个字段)
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

        % 新增的7个状态输出字段 (状态外化版本)
        % Stage_Manager状态输出 (4个字段)
        testOutputExtended.new_stage = int32(3);
        testOutputExtended.new_error_sign = int32(1);
        testOutputExtended.new_upgrade_count = int32(4);
        testOutputExtended.new_stage_offset = 0.2;

        % SPPVT_Adapter状态输出 (3个字段)
        testOutputExtended.new_control_error = 0.4;
        testOutputExtended.new_velocity = 13.7;
        testOutputExtended.new_accel = 0.25;

        fprintf('✅ 扩展输出总线测试数据创建成功 (19个字段)\n');

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