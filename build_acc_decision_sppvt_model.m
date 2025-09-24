function build_acc_decision_sppvt_model()
%BUILD_ACC_DECISION_SPPVT_MODEL 构建ACC决策+SPPVT一体化Simulink模型
%   适用于MATLAB 2024b，使用程序化方式创建Simulink模型
%   
%   此脚本创建完整的ACC_Decision_SPPVT_Integrated.slx模型
%   包含：输入验证 -> 决策功能 -> SPPVT功能 -> 输出处理

fprintf('🚀 开始构建ACC决策+SPPVT一体化Simulink模型...\n');

%% 1. 创建新模型
model_name = 'ACC_Decision_SPPVT_Integrated';

% 检查是否存在同名模型
if bdIsLoaded(model_name)
    fprintf('⚠️  模型 %s 已加载，正在关闭...\n', model_name);
    close_system(model_name, 0);
end

% 删除已存在的模型文件
model_file = [model_name '.slx'];
if exist(model_file, 'file')
    fprintf('🗑️  删除现有模型文件: %s\n', model_file);
    delete(model_file);
end

% 创建新模型
fprintf('📄 创建新模型: %s\n', model_name);
new_system(model_name);
open_system(model_name);

%% 2. 确保总线定义在基础工作区
fprintf('🔧 确保总线定义在基础工作区...\n');

% 检查基础工作区是否已有总线定义
if ~evalin('base', 'exist(''DecisionSPPVTInput'', ''var'')') || ...
   ~evalin('base', 'exist(''DecisionSPPVTOutput'', ''var'')')
    fprintf('⚠️ 总线定义不存在，正在创建...\n');
    create_decision_sppvt_bus();
else
    fprintf('✅ 总线定义已存在于基础工作区\n');
end

% 注意：不要将总线对象复制到模型工作区，保持在基础工作区即可

%% 3. 设置模型属性 (使用MATLAB 2024b新特性)
fprintf('⚙️  配置模型参数...\n');

% 基础配置
set_param(model_name, 'SolverType', 'Fixed-step');
set_param(model_name, 'FixedStep', '0.05'); % 50ms控制周期
set_param(model_name, 'StartTime', '0.0');
set_param(model_name, 'StopTime', 'inf');

% MATLAB 2024b的新配置选项
set_param(model_name, 'EnableMultiTasking', 'on');
set_param(model_name, 'SampleTimeColors', 'on');
set_param(model_name, 'SampleTimeAnnotations', 'on');

% 数据类型和代码生成
set_param(model_name, 'DefaultParameterBehavior', 'Tunable');
set_param(model_name, 'OptimizeBlockIOStorage', 'on');
set_param(model_name, 'BooleanDataType', 'on');

%% 4. 创建输入端口和Bus Creator
fprintf('📥 创建输入接口...\n');

% 输入端口 - 使用MATLAB 2024b的新语法
input_port = add_block('simulink/Sources/In1', [model_name '/Input'], ...
    'Position', [50, 100, 100, 130]);
set_param(input_port, 'OutDataTypeStr', 'Bus: DecisionSPPVTInput');
set_param(input_port, 'PortDimensions', '1');
set_param(input_port, 'SampleTime', '0.05');

%% 5. 创建输入验证模块
fprintf('🔍 创建输入验证模块...\n');

% 输入验证MATLAB Function块
input_validator = add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [model_name '/Input_Validator'], 'Position', [200, 80, 300, 150]);

% 设置MATLAB Function代码 - MATLAB 2024b兼容方式
try
    % 方法1: 尝试使用Simulink.MATLABFunctionConfiguration
    mfConfig = Simulink.MATLABFunctionConfiguration(input_validator);
    mfConfig.FunctionScript = generate_input_validator_code();
    fprintf('✅ 输入验证器代码设置成功\n');
catch ME1
    try
        % 方法2: 尝试直接设置Script参数
        set_param(input_validator, 'Script', generate_input_validator_code());
        fprintf('✅ 输入验证器代码设置成功\n');
    catch ME2
        % 方法3: 手动设置提示
        fprintf('⚠️ 输入验证器代码需要手动设置\n');
        fprintf('💡 请双击Input_Validator块，复制以下代码:\n');
        fprintf('==================输入验证器代码开始==================\n');
        fprintf('%s\n', generate_input_validator_code());
        fprintf('==================输入验证器代码结束==================\n');
    end
end

%% 6. 创建决策功能模块
fprintf('🧠 创建决策功能模块...\n');

% 决策MATLAB Function块
decision_block = add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [model_name '/Decision_Function'], 'Position', [400, 80, 500, 150]);

% 设置决策函数代码 - MATLAB 2024b兼容方式
try
    mfConfig = Simulink.MATLABFunctionConfiguration(decision_block);
    mfConfig.FunctionScript = load_decision_function_code();
    fprintf('✅ 决策模块代码设置成功\n');
catch ME1
    try
        set_param(decision_block, 'Script', load_decision_function_code());
        fprintf('✅ 决策模块代码设置成功\n');
    catch ME2
        fprintf('⚠️ 决策模块代码需要手动设置\n');
        fprintf('💡 请双击Decision_Function块，复制decision_function.m内容\n');
    end
end

%% 7. 集成现有SPPVT模块
fprintf('🎯 集成现有SPPVT控制模块...\n');

% 检查现有SPPVT模型是否存在
if exist('sppvt_control_model.slx', 'file') == 0
    error('❌ 找不到现有SPPVT模型: sppvt_control_model.slx');
end

% 尝试使用Model Reference集成现有SPPVT模型
try
    fprintf('📦 使用Model Reference集成现有SPPVT...\n');
    sppvt_ref = add_block('simulink/Ports & Subsystems/Model', ...
        [model_name '/SPPVT_Control'], 'Position', [600, 80, 750, 150]);
    set_param(sppvt_ref, 'ModelName', 'sppvt_control_model');
    fprintf('✅ SPPVT Model Reference集成成功\n');
catch ME1
    try
        % 备选方案：创建SPPVT接口适配器
        fprintf('🔌 创建SPPVT接口适配器...\n');
        sppvt_adapter = add_block('simulink/User-Defined Functions/MATLAB Function', ...
            [model_name '/SPPVT_Adapter'], 'Position', [600, 80, 750, 150]);
        
        % 设置适配器代码
        adapter_code = generate_sppvt_adapter_code();
        try
            mfConfig = Simulink.MATLABFunctionConfiguration(sppvt_adapter);
            mfConfig.FunctionScript = adapter_code;
            fprintf('✅ SPPVT适配器创建成功\n');
        catch ME3
            fprintf('⚠️ SPPVT适配器需要手动配置\n');
            fprintf('💡 适配器功能：调用你现有的sppvt_control_model\n');
        end
    catch ME2
        error('❌ 无法集成SPPVT模块: %s', ME2.message);
    end
end

%% 7.5. 添加SPPVT状态管理子系统
fprintf('🎯 添加SPPVT状态管理模块...\n');

% 创建状态管理器MATLAB Function块
stage_manager = add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [model_name '/Stage_Manager'], 'Position', [750, 80, 850, 150]);

% 设置状态管理代码 - MATLAB 2024b兼容方式
try
    mfConfig = Simulink.MATLABFunctionConfiguration(stage_manager);
    mfConfig.FunctionScript = generate_stage_manager_code();
    fprintf('✅ SPPVT状态管理器代码设置成功\n');
catch ME1
    try
        set_param(stage_manager, 'Script', generate_stage_manager_code());
        fprintf('✅ SPPVT状态管理器代码设置成功\n');
    catch ME2
        fprintf('⚠️ SPPVT状态管理器代码需要手动设置\n');
        fprintf('💡 请双击Stage_Manager块，复制以下代码:\n');
        fprintf('==================状态管理器代码开始==================\n');
        fprintf('%s\n', generate_stage_manager_code());
        fprintf('==================状态管理器代码结束==================\n');
    end
end

%% 8. 创建输出处理模块
fprintf('📤 创建输出处理模块...\n');

% 输出格式化MATLAB Function块
output_formatter = add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [model_name '/Output_Formatter'], 'Position', [800, 80, 900, 150]);

% 设置输出格式化代码 - MATLAB 2024b兼容方式
try
    mfConfig = Simulink.MATLABFunctionConfiguration(output_formatter);
    mfConfig.FunctionScript = generate_output_formatter_code();
    fprintf('✅ 输出处理器代码设置成功\n');
catch ME1
    try
        set_param(output_formatter, 'Script', generate_output_formatter_code());
        fprintf('✅ 输出处理器代码设置成功\n');
    catch ME2
        fprintf('⚠️ 输出处理器代码需要手动设置\n');
        fprintf('💡 请双击Output_Formatter块，复制以下代码:\n');
        fprintf('==================输出处理器代码开始==================\n');
        fprintf('%s\n', generate_output_formatter_code());
        fprintf('==================输出处理器代码结束==================\n');
    end
end

%% 9. 创建输出端口
fprintf('📤 创建输出接口...\n');

% 输出端口
output_port = add_block('simulink/Sinks/Out1', [model_name '/Output'], ...
    'Position', [1000, 100, 1050, 130]);
set_param(output_port, 'OutDataTypeStr', 'Bus: DecisionSPPVTOutput');

%% 10. 创建调试和监控模块
fprintf('📊 创建调试监控模块...\n');

% 调试显示
debug_display = add_block('simulink/Sinks/Display', [model_name '/Debug_Display'], ...
    'Position', [800, 200, 900, 250]);
set_param(debug_display, 'Decimation', '10'); % 每10个样本显示一次

% 状态监控示波器
scope_block = add_block('simulink/Sinks/Scope', [model_name '/State_Monitor'], ...
    'Position', [800, 270, 900, 320]);
set_param(scope_block, 'NumInputPorts', '3');

%% 11. 创建参数管理子系统
fprintf('📋 创建参数管理模块...\n');

% 参数管理子系统
param_subsystem = add_block('simulink/Ports & Subsystems/Subsystem', ...
    [model_name '/Parameter_Manager'], 'Position', [200, 200, 300, 250]);

% 在子系统中添加参数管理逻辑
build_parameter_manager_subsystem([model_name '/Parameter_Manager']);

%% 12. 连接所有模块
fprintf('🔗 连接模块...\n');

try
    % 主信号流连接（适配SPPVT集成架构）
    add_line(model_name, 'Input/1', 'Input_Validator/1');
    add_line(model_name, 'Input_Validator/1', 'Decision_Function/1');
    
    % 检查SPPVT模块类型并相应连接
    sppvt_blocks = find_system(model_name, 'Name', 'SPPVT_Control');
    sppvt_adapters = find_system(model_name, 'Name', 'SPPVT_Adapter');
    
    if ~isempty(sppvt_blocks)
        % 使用Model Reference + Stage Manager的连接方式
        fprintf('📦 连接Model Reference SPPVT + Stage Manager...\n');
        add_line(model_name, 'Input_Validator/1', 'SPPVT_Adapter/1');
        add_line(model_name, 'Decision_Function/1', 'SPPVT_Adapter/2');

        % Stage Manager连接 (修正：4个独立输入，无循环依赖)
        add_line(model_name, 'SPPVT_Control/5', 'Stage_Manager/1');   % should_upgrade输入
        add_line(model_name, 'Input_Validator/1', 'Stage_Manager/2');  % error_value直接从输入获取
        add_line(model_name, 'Parameter_Manager/1', 'Stage_Manager/3'); % sppvt_rho参数

        % 创建延迟块解决反馈循环
        delay_block = add_block('simulink/Discrete/Unit Delay', ...
            [model_name '/Stage_Offset_Delay'], 'Position', [720, 180, 750, 200]);
        set_param(delay_block, 'InitialCondition', '0.0');

        % Stage Manager输出通过延迟块反馈
        add_line(model_name, 'Stage_Manager/1', 'Stage_Offset_Delay/1');  % new_stage_offset到延迟块
        add_line(model_name, 'Stage_Offset_Delay/1', 'Stage_Manager/4');  % 延迟的prev_stage_offset

        % SPPVT连接 (使用Stage Manager的输出)
        add_line(model_name, 'Stage_Manager/1', 'SPPVT_Adapter/3');  % new_stage_offset反馈到SPPVT
        add_line(model_name, 'SPPVT_Adapter/1', 'SPPVT_Control/1');

        % SPPVT的5个输出连接到Output_Formatter的端口2-6
        add_line(model_name, 'SPPVT_Control/1', 'Output_Formatter/2');
        add_line(model_name, 'SPPVT_Control/2', 'Output_Formatter/3');
        add_line(model_name, 'SPPVT_Control/3', 'Output_Formatter/4');
        add_line(model_name, 'SPPVT_Control/4', 'Output_Formatter/5');
        add_line(model_name, 'SPPVT_Control/5', 'Output_Formatter/6');

        % Stage Manager状态信息也传给Output_Formatter
        add_line(model_name, 'Stage_Manager/2', 'Output_Formatter/7'); % new_stage
        add_line(model_name, 'Stage_Manager/3', 'Output_Formatter/8'); % sign_changed
        add_line(model_name, 'Decision_Function/1', 'Output_Formatter/1');
    elseif ~isempty(sppvt_adapters)
        % 使用适配器的连接方式
        fprintf('🔌 连接SPPVT适配器...\n');
        add_line(model_name, 'Input_Validator/1', 'SPPVT_Adapter/1');
        add_line(model_name, 'Decision_Function/1', 'SPPVT_Adapter/2');
        add_line(model_name, 'SPPVT_Adapter/1', 'Output_Formatter/2');
        add_line(model_name, 'Decision_Function/1', 'Output_Formatter/1');
    else
        % 默认连接方式（简化版）
        fprintf('🔗 使用默认连接方式...\n');
        add_line(model_name, 'Decision_Function/1', 'Output_Formatter/1');
    end
    
    % 最终输出连接
    add_line(model_name, 'Output_Formatter/1', 'Output/1');
    
    % 调试信号连接 - 添加Bus Selector
    bus_selector = add_block('simulink/Signal Routing/Bus Selector', ...
        [model_name '/Debug_Bus_Selector'], 'Position', [650, 200, 700, 250]);
    
    % 配置Bus Selector选择debug_message字段
    set_param(bus_selector, 'OutputSignals', 'debug_message');
    
    % 连接信号线
    add_line(model_name, 'Decision_Function/1', 'Debug_Bus_Selector/1');
    add_line(model_name, 'Debug_Bus_Selector/1', 'Debug_Display/1');
    % State_Monitor也需要Bus Selector (如果State_Monitor期望非总线输入)
    % 使用相同的Bus Selector或添加新的
    add_line(model_name, 'Debug_Bus_Selector/1', 'State_Monitor/1');
    add_line(model_name, 'Output_Formatter/1', 'State_Monitor/2');
    
    % 参数连接
    add_line(model_name, 'Parameter_Manager/1', 'State_Monitor/3');
    
    fprintf('✅ 模块连接完成\n');
    
catch ME
    fprintf('⚠️  模块连接时出现问题: %s\n', ME.message);
    fprintf('💡 请手动检查并完成连接\n');
end

%% 13. 布局优化
fprintf('🎨 优化模型布局...\n');

% 使用MATLAB 2024b的自动布局功能
try
    Simulink.BlockDiagram.arrangeSystem(model_name);
    fprintf('✅ 自动布局完成\n');
catch
    fprintf('💡 请手动调整模型布局\n');
end

%% 14. 保存模型
fprintf('💾 保存模型...\n');

try
    save_system(model_name);
    fprintf('✅ 模型已保存: %s.slx\n', model_name);
catch ME
    fprintf('⚠️  保存模型失败: %s\n', ME.message);
end

%% 15. 生成构建报告
fprintf('📋 生成构建报告...\n');
generate_build_report(model_name);

fprintf('\n🎉 ACC决策+SPPVT一体化模型构建完成!\n');
fprintf('📁 模型文件: %s.slx\n', model_name);
fprintf('🔧 使用方法:\n');
fprintf('   1. 打开模型: open_system(''%s'')\n', model_name);
fprintf('   2. 运行仿真: sim(''%s'')\n', model_name);
fprintf('   3. 配置参数: 使用Parameter_Manager子系统\n');
fprintf('\n');

end

%% 辅助函数

function build_parameter_manager_subsystem(subsystem_path)
%BUILD_PARAMETER_MANAGER_SUBSYSTEM 构建参数管理子系统

% 删除默认的输入端口（参数管理器不需要输入）
try
    delete_block([subsystem_path '/In1']);
catch
    % 如果没有In1端口就忽略
end

% 添加常数块用于参数管理
const_block = add_block('simulink/Sources/Constant', ...
    [subsystem_path '/ACC_Parameters'], 'Position', [50, 50, 150, 100]);

% 设置参数结构（直接使用已加载的ModelParams）
try
    % 尝试使用已加载的参数
    set_param(const_block, 'Value', 'ModelParams.decision');
catch ME1
    try
        % 备选：创建简化参数结构
        param_struct = struct('V_target_kmh', 50.0, 'V_min_kmh', 30.0, 'G2_s', 2.0);
        assignin('base', 'simple_param_struct', param_struct);
        set_param(const_block, 'Value', 'simple_param_struct');
    catch ME2
        % 最后备选：使用常数值
        set_param(const_block, 'Value', '50.0');  % 默认目标速度
    end
end
% 修复数据类型设置 - 使用SPPVT rho参数
try
    % 设置SPPVT rho参数 (惩罚系数)
    set_param(const_block, 'Value', '0.25');  % SPPVT rho参数
    set_param(const_block, 'OutDataTypeStr', 'double');
    fprintf('✅ Parameter_Manager设置SPPVT rho参数: 0.25\n');
catch
    % 如果需要使用结构体，需要显式指定总线类型
    set_param(const_block, 'OutDataTypeStr', 'Bus: <infer from value>');
end

% 使用默认的Out1端口（重命名为更清晰的名称）
try
    set_param([subsystem_path '/Out1'], 'Name', 'Parameters_Out');
catch
    % 如果重命名失败，使用原名
end

% 连接到默认输出端口
add_line(subsystem_path, 'ACC_Parameters/1', 'Out1/1');

end

function code = generate_input_validator_code()
%GENERATE_INPUT_VALIDATOR_CODE 生成输入验证MATLAB Function代码

code = [
'function validated_input = fcn(raw_input)', newline, ...
'%#codegen', newline, ...
'% 输入验证和预处理', newline, ...
'% 输入: raw_input (DecisionSPPVTInput总线)', newline, ...
'% 输出: validated_input (验证后的总线)', newline, newline, ...
'% 初始化输出结构（避免总线类型引用问题）', newline, ...
'validated_input = raw_input;', newline, newline, ...
'% 数值范围检查', newline, ...
'validated_input.ego_speed_kmh = max(0, min(200, raw_input.ego_speed_kmh));', newline, ...
'validated_input.ego_speed_ms = max(0, min(60, raw_input.ego_speed_ms));', newline, ...
'validated_input.V_target_kmh = max(30, min(120, raw_input.V_target_kmh));', newline, ...
'validated_input.V_min_kmh = max(20, min(50, raw_input.V_min_kmh));', newline, ...
'validated_input.G2_s = max(1.0, min(3.0, raw_input.G2_s));', newline, newline, ...
'% 逻辑一致性检查', newline, ...
'if validated_input.V_min_kmh > validated_input.V_target_kmh', newline, ...
'    validated_input.V_min_kmh = validated_input.V_target_kmh - 10;', newline, ...
'end', newline, newline, ...
'% 时间戳更新', newline, ...
'validated_input.timestamp = raw_input.timestamp;', newline, ...
'end'
];

end

function code = load_decision_function_code()
%LOAD_DECISION_FUNCTION_CODE 加载决策函数代码

try
    code = fileread('decision_function.m');
    % 移除函数头，只保留函数体
    lines = strsplit(code, newline);
    func_start = find(contains(lines, 'function output = decision_function'), 1);
    if ~isempty(func_start)
        code = strjoin(lines(func_start:end), newline);
    end
catch
    % 如果文件读取失败，使用简化版本
    code = generate_simplified_decision_code();
end

end

function code = load_sppvt_function_code()
%LOAD_SPPVT_FUNCTION_CODE 加载SPPVT函数代码

try
    code = fileread('sppvt_function.m');
    % 移除函数头，只保留函数体
    lines = strsplit(code, newline);
    func_start = find(contains(lines, 'function output = sppvt_function'), 1);
    if ~isempty(func_start)
        code = strjoin(lines(func_start:end), newline);
    end
catch
    % 如果文件读取失败，使用简化版本
    code = generate_simplified_sppvt_code();
end

end

function code = generate_output_formatter_code()
%GENERATE_OUTPUT_FORMATTER_CODE 生成输出格式化代码
%   处理现有sppvt_control_model的向量输出格式

code = [
'function final_output = fcn(decision_output, sppvt_raw_output)', newline, ...
'%#codegen', newline, ...
'% 合并决策和SPPVT输出', newline, ...
'% sppvt_raw_output: 来自现有sppvt_control_model的向量输出', newline, newline, ...
'%% 初始化persistent变量', newline, ...
'persistent upgrade_count', newline, ...
'if isempty(upgrade_count)', newline, ...
'    upgrade_count = int32(0);', newline, ...
'end', newline, newline, ...
'%% 解析SPPVT原始输出（基于sppvt_longitudinal_control.py的接口）', newline, ...
'if length(sppvt_raw_output) >= 5', newline, ...
'    % 匹配你现有sppvt_control_model的输出格式', newline, ...
'    control_output = sppvt_raw_output(1);     % 控制输出', newline, ...
'    velocity = sppvt_raw_output(2);          % 一阶导数', newline, ...
'    acceleration = sppvt_raw_output(3);      % 二阶导数', newline, ...
'    jerk = sppvt_raw_output(4);              % 三阶导数', newline, ...
'    should_upgrade = sppvt_raw_output(5) > 0.5; % 升级标志', newline, ...
'    if length(sppvt_raw_output) >= 6', newline, ...
'        sppvt_stage = int32(sppvt_raw_output(6));', newline, ...
'    else', newline, ...
'        sppvt_stage = int32(1);', newline, ...
'    end', newline, ...
'else', newline, ...
'    % 默认值（SPPVT输出异常时）', newline, ...
'    control_output = 0.0;', newline, ...
'    velocity = 0.0;', newline, ...
'    acceleration = 0.0;', newline, ...
'    jerk = 0.0;', newline, ...
'    should_upgrade = false;', newline, ...
'    sppvt_stage = int32(1);', newline, ...
'end', newline, newline, ...
'%% 更新升级计数', newline, ...
'if should_upgrade', newline, ...
'    upgrade_count = upgrade_count + 1;', newline, ...
'end', newline, newline, ...
'%% 构造标准输出格式', newline, ...
'final_output = struct();', newline, newline, ...
'% SPPVT控制结果', newline, ...
'final_output.target_accel = control_output;', newline, ...
'final_output.sppvt_stage = sppvt_stage;', newline, ...
'final_output.sppvt_upgrade_count = upgrade_count;', newline, newline, ...
'% 决策状态信息', newline, ...
'final_output.control_enabled = decision_output.control_enabled;', newline, ...
'final_output.current_state = decision_output.current_state;', newline, ...
'final_output.current_decision = decision_output.current_decision;', newline, ...
'final_output.torque_arbitration_active = decision_output.torque_arbitration_active;', newline, ...
'final_output.updated_V_target_kmh = decision_output.updated_V_target_kmh;', newline, ...
'final_output.updated_G2_s = decision_output.updated_G2_s;', newline, newline, ...
'% 调试信息（区分SPPVT集成版本）', newline, ...
'final_output.debug_message = int32(3000 + double(sppvt_stage)*10 + mod(double(upgrade_count), 10));', newline, newline, ...
'% 调试输出', newline, ...
'if abs(control_output) > 0.01', newline, ...
'    fprintf("输出处理器: 加速度=%.3f, SPPVT级数=%d, 决策状态=%d\\n", ...\n', ...
'            control_output, sppvt_stage, decision_output.current_state);', newline, ...
'end', newline, ...
'end'
];

end

function code = generate_simplified_decision_code()
%GENERATE_SIMPLIFIED_DECISION_CODE 生成简化的决策代码

code = [
'function output = fcn(input)', newline, ...
'%#codegen', newline, ...
'persistent state', newline, ...
'if isempty(state)', newline, ...
'    state = int32(0);', newline, ...
'end', newline, newline, ...
'output = struct();', newline, ...
'output.control_enabled = input.command_active;', newline, ...
'output.current_state = state;', newline, ...
'output.current_decision = input.command_type;', newline, ...
'output.torque_arbitration_active = input.manual_throttle_active;', newline, ...
'output.updated_V_target_kmh = input.V_target_kmh;', newline, ...
'output.updated_G2_s = input.G2_s;', newline, ...
'output.debug_message = int32(1000);', newline, ...
'end'
];

end

function code = generate_simplified_sppvt_code()
%GENERATE_SIMPLIFIED_SPPVT_CODE 生成简化的SPPVT代码

code = [
'function output = fcn(input, decision_output)', newline, ...
'%#codegen', newline, ...
'persistent integral_term', newline, ...
'if isempty(integral_term)', newline, ...
'    integral_term = 0;', newline, ...
'end', newline, newline, ...
'if decision_output.control_enabled', newline, ...
'    kp = 0.8;', newline, ...
'    target_accel = kp * input.control_error;', newline, ...
'    target_accel = max(-4, min(2, target_accel));', newline, ...
'else', newline, ...
'    target_accel = 0;', newline, ...
'end', newline, newline, ...
'output = struct();', newline, ...
'output.target_accel = target_accel;', newline, ...
'output.sppvt_stage = int32(1);', newline, ...
'output.sppvt_upgrade_count = int32(0);', newline, ...
'output.debug_message = int32(2000);', newline, ...
'end'
];

end

function generate_build_report(model_name)
%GENERATE_BUILD_REPORT 生成构建报告

report_file = 'ACC_Decision_SPPVT_Build_Report.txt';
fid = fopen(report_file, 'w');

if fid ~= -1
    fprintf(fid, 'ACC决策+SPPVT一体化模型构建报告\n');
    fprintf(fid, '=====================================\n\n');
    fprintf(fid, '构建时间: %s\n', datestr(now));
    fprintf(fid, '模型名称: %s\n', model_name);
    fprintf(fid, 'MATLAB版本: %s\n', version);
    fprintf(fid, '\n模块列表:\n');
    
    try
        blocks = find_system(model_name, 'Type', 'block');
        for i = 1:length(blocks)
            block_type = get_param(blocks{i}, 'BlockType');
            fprintf(fid, '  - %s (%s)\n', blocks{i}, block_type);
        end
    catch
        fprintf(fid, '  模块信息获取失败\n');
    end
    
    fprintf(fid, '\n总线定义:\n');
    fprintf(fid, '  - DecisionSPPVTInput (11个字段)\n');
    fprintf(fid, '  - DecisionSPPVTOutput (10个字段)\n');
    
    fprintf(fid, '\n使用说明:\n');
    fprintf(fid, '  1. 确保已运行create_decision_sppvt_bus()\n');
    fprintf(fid, '  2. 使用sim(''%s'')运行仿真\n', model_name);
    fprintf(fid, '  3. 检查调试输出和监控信号\n');
    
    fclose(fid);
    fprintf('📋 构建报告已保存: %s\n', report_file);
end

function code = generate_stage_manager_code()
%GENERATE_STAGE_MANAGER_CODE 生成SPPVT状态管理器代码
%   包含误差符号变化检测和状态重置机制

code = [
'function [new_stage_offset, new_stage, sign_changed] = fcn(should_upgrade, error_value, sppvt_rho, prev_stage_offset)', newline, ...
'%#codegen', newline, ...
'% SPPVT状态管理器 - 包含误差符号变化检测和状态重置', newline, ...
'% 输入: should_upgrade, error_value, prev_stage_offset, sppvt_rho', newline, ...
'% 输出: new_stage_offset, new_stage, sign_changed', newline, newline, ...
'%% 持久变量：状态管理', newline, ...
'persistent current_stage prev_error_sign upgrade_count', newline, ...
'if isempty(current_stage)', newline, ...
'    current_stage = int32(1);', newline, ...
'    prev_error_sign = int32(0);  % 0=零, 1=正, -1=负', newline, ...
'    upgrade_count = int32(0);', newline, ...
'end', newline, newline, ...
'%% 计算当前误差符号', newline, ...
'if abs(error_value) < 1e-6', newline, ...
'    current_sign = int32(0);     % 接近零', newline, ...
'elseif error_value > 0', newline, ...
'    current_sign = int32(1);     % 正误差', newline, ...
'else', newline, ...
'    current_sign = int32(-1);    % 负误差', newline, ...
'end', newline, newline, ...
'%% 检测误差符号变化', newline, ...
'sign_changed = false;', newline, ...
'if (prev_error_sign ~= 0) && (current_sign ~= 0) && (prev_error_sign ~= current_sign)', newline, ...
'    % 符号变化：从非零符号变为另一个非零符号', newline, ...
'    sign_changed = true;', newline, newline, ...
'    % 重置所有状态到初始级', newline, ...
'    current_stage = int32(1);', newline, ...
'    new_stage_offset = 0.0;      % 初始级差为0', newline, ...
'    upgrade_count = int32(0);', newline, newline, ...
'    % 调试输出', newline, ...
'    if current_sign == 1', newline, ...
'        fprintf("Stage_Manager: 检测到误差符号变化(负→正), 重置到初始级\\n");', newline, ...
'    else', newline, ...
'        fprintf("Stage_Manager: 检测到误差符号变化(正→负), 重置到初始级\\n");', newline, ...
'    end', newline, newline, ...
'elseif should_upgrade && ~sign_changed', newline, ...
'    % 没有符号变化且满足升级条件', newline, ...
'    current_stage = current_stage + 1;', newline, ...
'    upgrade_count = upgrade_count + 1;', newline, newline, ...
'    % 根据误差符号计算新级差（与Python实现一致）', newline, ...
'    if error_value > 0', newline, ...
'        % 正误差：增加正级差，让正误差更大', newline, ...
'        new_stage_offset = prev_stage_offset + sppvt_rho * abs(error_value);', newline, ...
'    else', newline, ...
'        % 负误差：增加负级差，让负误差更小（绝对值更大）', newline, ...
'        new_stage_offset = prev_stage_offset - sppvt_rho * abs(error_value);', newline, ...
'    end', newline, newline, ...
'    % 调试输出', newline, ...
'    if error_value > 0', newline, ...
'        error_direction = "正";', newline, ...
'    else', newline, ...
'        error_direction = "负";', newline, ...
'    end', newline, ...
'    fprintf("Stage_Manager: 升级到第%d级, %s误差(%.3f) → 级差: %.3f → %.3f\\n", ...', newline, ...
'            current_stage, error_direction, error_value, prev_stage_offset, new_stage_offset);', newline, newline, ...
'else', newline, ...
'    % 不升级：保持当前状态', newline, ...
'    new_stage_offset = prev_stage_offset;', newline, ...
'end', newline, newline, ...
'% 更新误差符号历史（只有非零误差才更新）', newline, ...
'if current_sign ~= 0', newline, ...
'    prev_error_sign = current_sign;', newline, ...
'end', newline, newline, ...
'new_stage = current_stage;', newline, ...
'end'
];

end

end