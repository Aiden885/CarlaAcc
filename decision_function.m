function output = decision_function(input)
%DECISION_FUNCTION ACC决策状态机的MATLAB实现
%   适用于MATLAB 2024b，在Simulink中作为MATLAB Function块使用
%   
%   输入: DecisionSPPVTInput总线
%   输出: 决策相关的输出结构
%
%   状态定义 (根据decision.md和acc_decision.py):
%   S0 = 0: 在控状态 (ACTIVE_CONTROL)
%   S1 = 1: 适速有史待命 (ADAPTIVE_HISTORY_STANDBY)  
% %   S2 = 2: 适速无史待命 (ADAPTIVE_NO_HISTORY_STANDBY)
%   S3 = 3: 低速状态 (LOW_SPEED)
%
%   决策定义:
%   R1 = 1: 速度降低 (DECREASE_SPEED)
%   R2 = 2: 速度增加 (INCREASE_SPEED)
%   R3 = 3: 时距降低 (DECREASE_DISTANCE)
%   R4 = 4: 时距增加 (INCREASE_DISTANCE)
%   R5 = 5: 无继控制 (ACTIVATE_CURRENT_SPEED)
%   R6 = 6: 继承控制 (ACTIVATE_INHERITED_SPEED)
%   R7 = 7: 扭矩仲裁 (TORQUE_ARBITRATION)
%   R8 = 8: 系统待命 (SYSTEM_STANDBY)

% 保留必要的persistent变量（仅用于调试计数）
persistent debug_counter

% 初始化调试计数器
if isempty(debug_counter)
    debug_counter = int32(0);
end

% 提取输入参数
ego_speed_kmh = input.ego_speed_kmh;
% ego_speed_ms = input.ego_speed_ms;          % 未使用，决策逻辑只需要km/h
command_type = input.command_type;
command_active = input.command_active;
manual_throttle_active = input.manual_throttle_active;
% control_error = input.control_error;        % 决策模块不直接使用控制误差
% control_mode_flag = input.control_mode_flag; % 决策模块不直接使用模式标志
input_V_target_kmh = input.V_target_kmh;
input_V_min_kmh = input.V_min_kmh;
input_G2_s = input.G2_s;
% timestamp = input.timestamp;                % 决策逻辑不需要时间戳

% 更新参数
V_target_kmh = input_V_target_kmh;
V_min_kmh = input_V_min_kmh;
G2_s = input_G2_s;

%% 键盘参数修改逻辑 (在状态机逻辑之前处理)
if command_active
    speed_step = 5.0;  % 速度调整步长 (km/h)
    time_gap_step = 0.2; % 时距调整步长 (s)

    switch command_type
        case 1 % I0: Decrease speed -> E key
            V_target_kmh = max(V_min_kmh, V_target_kmh - speed_step);
            fprintf('Keyboard: E key decrease speed, V_target: %.1f km/h\n', V_target_kmh);

        case 2 % I1: Increase speed -> Q key
            V_target_kmh = min(120.0, V_target_kmh + speed_step);
            fprintf('Keyboard: Q key increase speed, V_target: %.1f km/h\n', V_target_kmh);

        case 3 % I2: Decrease time gap -> T key
            G2_s = max(1.0, G2_s - time_gap_step);
            fprintf('Keyboard: T key decrease gap, G2: %.1f s\n', G2_s);

        case 4 % I3: Increase time gap -> R key
            G2_s = min(5.0, G2_s + time_gap_step);
            fprintf('Keyboard: R key increase gap, G2: %.1f s\n', G2_s);

        case 7 % I6: Cancel ACC -> C key
            fprintf('Keyboard: C key cancel ACC\n');
            % 取消逻辑在状态机中处理
    end
end

% 增加调试计数器
debug_counter = debug_counter + 1;

%% 从输入获取决策状态信息（替代persistent变量）
% 如果总线中没有这些字段，使用默认值
if isfield(input, 'current_state')
    current_state = input.current_state;
else
    current_state = int32(2); % 默认S2状态
end

if isfield(input, 'has_history')
    has_history = input.has_history;
else
    has_history = false; % 默认无历史
end

if isfield(input, 'last_active_decision')
    last_active_decision = input.last_active_decision;
else
    last_active_decision = int32(8); % 默认R8
end

%% 获取SPPVT状态外化字段（虽然决策模块不直接使用，但需要传递）
% 这些字段将在输出中保持原值，由SPPVT模块使用
if isfield(input, 'external_stage_offset')
    external_stage_offset = input.external_stage_offset;
else
    external_stage_offset = 0.0; % 默认级差
end

if isfield(input, 'external_stage_manager_states')
    external_stage_manager_states = input.external_stage_manager_states;
else
    external_stage_manager_states = [1.0; 0.0; 0.0]; % [stage, error_sign, upgrade_count]
end

if isfield(input, 'external_adapter_states')
    external_adapter_states = input.external_adapter_states;
else
    external_adapter_states = [0.0; 0.0; 0.0]; % [prev_error, prev_velocity, prev_accel]
end

%% 状态机主逻辑 - 使用next_state管理状态转移
next_state = current_state; % 默认保持当前状态
next_has_history = has_history; % 默认保持历史状态
next_last_active_decision = last_active_decision; % 默认保持决策
torque_arbitration_active = false;

% 基于当前状态设置默认决策
if current_state == 0
    control_enabled = true;
    current_decision = last_active_decision; % 使用上次有效决策实现持续性
    % 如果历史决策无效，提供后备值
    if ~(current_decision >= 1 && current_decision <= 7)
        current_decision = int32(5); % 默认 R5: 无继控制
    end
else
    control_enabled = false;
    current_decision = int32(8); % R8: 系统待命
end

% 基于车速的自动状态转移
if ego_speed_kmh < V_min_kmh
    if current_state ~= 3
        next_state = int32(3); % 转移到S3: 低速状态
        control_enabled = false;
        current_decision = int32(8);
    end
else
    % 从S3状态恢复
    if current_state == 3
        if has_history
            next_state = int32(1); % 恢复到S1: 有史待命
        else
            next_state = int32(2); % 恢复到S2: 无史待命
        end
        control_enabled = false;
        current_decision = int32(8);
    end
end

% 处理激活的指令
if command_active
    switch current_state
        case 0 % S0: 在控状态
            [next_state, current_decision, control_enabled, torque_arbitration_active] = ...
                handle_active_control_state(command_type, ego_speed_kmh, manual_throttle_active, V_target_kmh, G2_s);

        case 1 % S1: 适速有史待命
            [next_state, current_decision, control_enabled, next_has_history, V_target_kmh, G2_s] = ...
                handle_adaptive_history_standby_state(command_type, ego_speed_kmh, V_target_kmh, G2_s, input_V_target_kmh, input_G2_s);

        case 2 % S2: 适速无史待命
            [next_state, current_decision, control_enabled, next_has_history, V_target_kmh, G2_s] = ...
                handle_adaptive_no_history_standby_state(command_type, ego_speed_kmh, input_V_target_kmh, input_G2_s);

        case 3 % S3: 低速状态
            % 低速状态下，指令不进入控制
            current_decision = int32(8);
            control_enabled = false;

        otherwise
            % 错误状态，重置到无史待命
            next_state = int32(2);
            current_decision = int32(8);
            control_enabled = false;
            next_has_history = false;
    end
end

% 更新有效决策用于持续性控制
if current_decision >= 1 && current_decision <= 7
    next_last_active_decision = current_decision;
end

%% 计算调试信息值（在赋值前准备）
control_flag = int32(control_enabled);
if mod(debug_counter, 20) == 0 % 每20个周期输出一次调试信息
    % 调试代码: 1000 + 状态*100 + 决策*10 + 控制使能标志
    debug_code = int32(1000 + current_state*100 + current_decision*10 + control_flag);
    debug_message_value = debug_code;

    % 添加MATLAB Function调试输出
    fprintf('Decision Function Debug: State=%d->%d, Decision=%d, Control=%d, Active=%d, Cmd=%d, DebugCode=%d\n', ...
            current_state, next_state, current_decision, int32(control_enabled), int32(command_active), command_type, debug_code);
else
    % 非调试周期显示简单计数器
    debug_message_value = debug_counter;
end

%% 构造输出 - 严格按照DecisionSPPVTOutputExtended总线定义顺序赋值所有18个字段
output = struct();

% 1-6: 来自决策系统的基本信息
output.control_enabled = logical(control_enabled);
output.current_state = int32(next_state);  % 修正：输出当前步计算的实际状态
output.current_decision = int32(current_decision);
output.torque_arbitration_active = logical(torque_arbitration_active);
output.updated_V_target_kmh = double(V_target_kmh);
output.updated_G2_s = double(G2_s);

% 7-11: SPPVT相关字段的默认值（这些将由后续模块填充）
output.sppvt_control_output = 0.0;
output.sppvt_velocity_output = 0.0;
output.sppvt_acceleration_output = 0.0;
output.sppvt_stage_output = 0.0;
output.sppvt_status_output = 0.0;

% 12: 调试信息
output.debug_message = debug_message_value;

% 13-15: 决策状态输出字段（状态外化支持）
output.next_state = int32(next_state);
output.next_has_history = logical(next_has_history);
output.next_last_active_decision = int32(next_last_active_decision);

% 16-18: SPPVT状态输出字段（传递外部输入，由后续SPPVT模块更新）
output.new_stage_offset = double(external_stage_offset);
output.new_stage_manager_states = double(external_stage_manager_states);
output.new_adapter_states = double(external_adapter_states);

end

%% 状态处理函数

function [new_state, decision, control_enabled, torque_arbitration] = ...
    handle_active_control_state(command_type, ego_speed_kmh, manual_throttle_active, V_target_kmh, G2_s)
%处理在控状态 S0 (根据decision.md)

new_state = int32(0); % 默认保持在控状态
control_enabled = true;
torque_arbitration = false;

% 初始化所有输出变量（确保在所有执行路径上都有定义）
decision = int32(8); % 默认系统待命

switch command_type
    case 1 % I0: 降速指令
        decision = int32(1); % R1: 速度降低
        
    case 2 % I1: 增速指令
        decision = int32(2); % R2: 速度增加
        
    case 3 % I2: 降距指令
        decision = int32(3); % R3: 时距降低
        
    case 4 % I3: 增距指令
        decision = int32(4); % R4: 时距增加
        
    case 5 % I4: 油门指令
        decision = int32(7); % R7: 扭矩仲裁
        torque_arbitration = true;
        % 状态保持S0，但控制权可能被驾驶员超越
        
    case 6 % I5: 刹车指令
        new_state = int32(1); % 转到S1: 适速有史待命
        decision = int32(8); % R8: 系统待命
        control_enabled = false;
        
    case 7 % I6: 取消ACC指令
        new_state = int32(1); % 转到S1: 适速有史待命
        decision = int32(8); % R8: 系统待命
        control_enabled = false;
        
    otherwise
        decision = int32(8); % R8: 系统待命
end
end

function [new_state, decision, control_enabled, has_history, V_target_kmh, G2_s] = ...
    handle_adaptive_history_standby_state(command_type, ego_speed_kmh, current_V_target, current_G2, input_V_target, input_G2)
%处理适速有史待命状态 S1 (根据decision.md)

new_state = int32(1); % 默认保持S1状态
control_enabled = false;
has_history = true; % 保持历史数据

% 初始化所有输出变量（确保在所有执行路径上都有定义）
V_target_kmh = current_V_target;
G2_s = current_G2;
decision = int32(8); % 默认系统待命

switch command_type
    case 1 % I0: 降速指令
        new_state = int32(0); % 转到S0: 在控状态
        decision = int32(5); % R5: 无继控制 (当速启控)
        control_enabled = true;
        V_target_kmh = input_V_target; % 使用当前速度作为目标
        G2_s = input_G2;
        
    case 2 % I1: 增速指令
        new_state = int32(0); % 转到S0: 在控状态
        decision = int32(6); % R6: 继承控制 (使用历史速度)
        control_enabled = true;
        % V_target_kmh和G2_s保持历史值不变
        
    otherwise % I2-I6: 其他指令
        decision = int32(8); % R8: 系统待命
        % 状态保持S1，参数保持历史值
end
end

function [new_state, decision, control_enabled, has_history, V_target_kmh, G2_s] = ...
    handle_adaptive_no_history_standby_state(command_type, ego_speed_kmh, input_V_target, input_G2)
%处理适速无史待命状态 S2 (根据decision.md)

new_state = int32(2); % 默认保持S2状态
control_enabled = false;
has_history = false; % 仍然无历史数据

% 初始化所有输出变量（确保在所有执行路径上都有定义）
V_target_kmh = input_V_target;
G2_s = input_G2;
decision = int32(8); % 默认系统待命

switch command_type
    case 1 % I0: 降速指令
        new_state = int32(0); % 转到S0: 在控状态
        decision = int32(5); % R5: 无继控制 (当速启控)
        control_enabled = true;
        has_history = true; % 启控后产生历史数据
        V_target_kmh = input_V_target; % 使用当前速度作为目标
        G2_s = input_G2;
        
    otherwise % I1-I6: 其他指令
        decision = int32(8); % R8: 系统待命
        % 状态保持S2，无历史数据
        % V_target_kmh和G2_s已在函数开头初始化
end
end