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

% 使用persistent变量在MATLAB Function块中保持状态
persistent current_state prev_state state_entry_time
persistent V_target_kmh V_min_kmh G2_s
persistent has_history % 记录是否有历史速度数据
persistent decision_history control_history
persistent debug_counter
persistent last_active_decision % 保存最后一个有效决策用于持续性

% 初始化persistent变量
if isempty(current_state)
    current_state = int32(2); % S2: 适速无史待命状态 (初始状态)
    prev_state = int32(2);
    state_entry_time = 0.0;
    V_target_kmh = 50.0;
    V_min_kmh = 30.0;
    G2_s = 2.0;
    has_history = false; % 初始无历史数据
    decision_history = zeros(1, 10, 'int32'); % 最近10个决策
    control_history = zeros(1, 10); % 最近10个控制输出
    debug_counter = int32(0);
    last_active_decision = int32(8); % 初始无有效决策
end

% 提取输入参数
ego_speed_kmh = input.ego_speed_kmh;
ego_speed_ms = input.ego_speed_ms;
command_type = input.command_type;
command_active = input.command_active;
manual_throttle_active = input.manual_throttle_active;
control_error = input.control_error;
control_mode_flag = input.control_mode_flag;
input_V_target_kmh = input.V_target_kmh;
input_V_min_kmh = input.V_min_kmh;
input_G2_s = input.G2_s;
timestamp = input.timestamp;

% 更新参数
V_target_kmh = input_V_target_kmh;
V_min_kmh = input_V_min_kmh;
G2_s = input_G2_s;

% 增加调试计数器
debug_counter = debug_counter + 1;

%% 状态机主逻辑
prev_state = current_state;
torque_arbitration_active = false;

% 基于当前状态设置默认值，将决策持续性作为S0状态的默认行为
if current_state == 0
    control_enabled = true;
    current_decision = last_active_decision; % 默认使用上一个有效决策以实现持续性
    % fprintf('DEBUG: S0 using last_active_decision=%d\n', last_active_decision);
    % 如果历史决策无效（例如刚启动时），提供一个后备值
    if ~(current_decision >= 1 && current_decision <= 7)
        current_decision = int32(5); % 默认 R5: 无继控制
        % fprintf('DEBUG: Invalid history decision, using default R5\n');
    else
        % fprintf('DEBUG: Using valid history decision R%d\n', current_decision);
    end
else
    control_enabled = false;
    current_decision = int32(8); % R8: 其他所有状态默认系统待命
    % fprintf('DEBUG: Non-S0 state, using R8\n');
end

% 首先，检查基于车速的自动状态转移（这可能会覆盖上面的默认值）
if ego_speed_kmh < V_min_kmh
    if current_state ~= 3
        current_state = int32(3); % S3: 低速状态
        control_enabled = false;
        current_decision = int32(8);
    end
else
    % 只有当前在S3状态时才需要恢复，避免误覆盖其他状态的决策
    if current_state == 3
        if has_history
            current_state = int32(1); % S1: 恢复到有历史待命
        else
            current_state = int32(2); % S2: 恢复到无历史待命
        end
        control_enabled = false;
        current_decision = int32(8);
    end
    % 如果不在S3状态，则不修改control_enabled和current_decision，保持之前设置的默认值
end

% 如果有激活的指令，它将覆盖当前的默认决策
if command_active
    switch current_state
        case 0 % S0: 在控状态
            [current_state, current_decision, control_enabled, torque_arbitration_active] = ...
                handle_active_control_state(command_type, ego_speed_kmh, manual_throttle_active, V_target_kmh, G2_s);

        case 1 % S1: 适速有史待命
            [current_state, current_decision, control_enabled, has_history, V_target_kmh, G2_s] = ...
                handle_adaptive_history_standby_state(command_type, ego_speed_kmh, V_target_kmh, G2_s, input_V_target_kmh, input_G2_s);

        case 2 % S2: 适速无史待命
            [current_state, current_decision, control_enabled, has_history, V_target_kmh, G2_s] = ...
                handle_adaptive_no_history_standby_state(command_type, ego_speed_kmh, input_V_target_kmh, input_G2_s);

        case 3 % S3: 低速状态
            % 低速状态下，激活的指令不会进入控制
            current_decision = int32(8);
            control_enabled = false;

        otherwise
            % 错误状态，重置到无史待命
            current_state = int32(2);
            current_decision = int32(8);
            control_enabled = false;
            has_history = false;
    end

end
% 用于处理 command_active 为 false 的大型 'else' 块不再需要了

% 记录状态变化时间
if current_state ~= prev_state
    state_entry_time = timestamp;
end

% 更新历史记录
decision_history = [decision_history(2:end), current_decision];

% 统一更新有效决策用于持续性控制 (独立于指令激活状态)
% 只要当前周期产生了有效的控制决策（R1-R7），就保存为下一周期的持续性决策
if current_decision >= 1 && current_decision <= 7
    % fprintf('DEBUG: Saving valid decision R%d to last_active_decision\n', current_decision);
    last_active_decision = current_decision;
else
    % fprintf('DEBUG: Decision R%d invalid, not saving\n', current_decision);
end

%% 构造输出 - 严格按照DecisionSPPVTOutput总线定义顺序赋值
% 按照总线定义的严格顺序 (1-12) 赋值所有字段
output = struct();

% 1-6: 来自决策系统的信息
output.control_enabled = logical(control_enabled);
output.current_state = int32(current_state);
output.current_decision = int32(current_decision);
output.torque_arbitration_active = logical(torque_arbitration_active);
output.updated_V_target_kmh = double(V_target_kmh);
output.updated_G2_s = double(G2_s);

% 7: 调试信息（必须在SPPVT字段之前赋值）
control_flag = int32(control_enabled);
if mod(debug_counter, 20) == 0 % 每20个周期输出一次调试信息
    % 调试代码: 1000 + 状态*100 + 决策*10 + 控制使能标志
    debug_code = int32(1000 + current_state*100 + current_decision*10 + control_flag);
    output.debug_message = debug_code;
    
    % 添加MATLAB Function调试输出
    fprintf('Decision Function Debug: State=%d, Decision=%d, Control=%d, Active=%d, Cmd=%d, DebugCode=%d\n', ...
            current_state, current_decision, int32(control_enabled), int32(command_active), command_type, debug_code);
else
    % 非调试周期显示简单计数器
    output.debug_message = debug_counter;
end

% 8-12: SPPVT相关字段的默认值（这些将由后续模块填充）
output.sppvt_control_output = 0.0;
output.sppvt_velocity_output = 0.0;
output.sppvt_acceleration_output = 0.0;
output.sppvt_stage_output = 0.0;
output.sppvt_status_output = 0.0;

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