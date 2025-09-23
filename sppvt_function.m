function output = sppvt_function(input, decision_output)
%SPPVT_FUNCTION SPPVT纵向控制算法的MATLAB实现
%   适用于MATLAB 2024b，在Simulink中作为MATLAB Function块使用
%   
%   输入: 
%   - input: DecisionSPPVTInput总线
%   - decision_output: 决策模块输出
%   输出: SPPVT控制相关的输出结构
%
%   SPPVT算法实现Set Point Pre-Positioning Velocity Tracking控制
%   基于现有Python实现的sppvt_longitudinal_control.py移植

% 使用persistent变量保持SPPVT控制器内部状态
persistent sppvt_state sppvt_params
persistent prev_error integral_term dt_sppvt
persistent stage upgrade_count control_mode
persistent debug_sppvt_counter

% 初始化persistent变量
if isempty(sppvt_state)
    % SPPVT控制器状态初始化
    sppvt_state = struct();
    sppvt_state.initialized = false;
    
    % SPPVT参数初始化（基于原Python实现）
    sppvt_params = struct();
    sppvt_params.kp = 0.8;           % 比例增益
    sppvt_params.ki = 0.1;           % 积分增益  
    sppvt_params.kd = 0.05;          % 微分增益
    sppvt_params.max_accel = 2.0;    % 最大加速度 m/s²
    sppvt_params.max_decel = -4.0;   % 最大减速度 m/s²
    sppvt_params.deadzone = 0.1;     % 死区
    
    % 控制器内部状态
    prev_error = 0.0;
    integral_term = 0.0;
    dt_sppvt = 0.05;                 % 控制步长 50ms
    
    % SPPVT阶段和升级计数
    stage = int32(1);                % 默认阶段1
    upgrade_count = int32(0);
    control_mode = 'speed';          % 默认速度控制
    
    debug_sppvt_counter = int32(0);
end

% 提取输入参数
ego_speed_kmh = input.ego_speed_kmh;
ego_speed_ms = input.ego_speed_ms;
control_error = input.control_error;
control_mode_flag = input.control_mode_flag;
V_target_kmh = input.V_target_kmh;
timestamp = input.timestamp;

% 从决策输出获取控制使能状态
control_enabled = decision_output.control_enabled;
torque_arbitration_active = decision_output.torque_arbitration_active;

% 增加调试计数器
debug_sppvt_counter = debug_sppvt_counter + 1;

%% SPPVT主控制逻辑

target_accel = 0.0; % 默认输出加速度

if control_enabled
    % 根据控制模式标志设置SPPVT模式
    if control_mode_flag == 1
        control_mode = 'distance';
        error_value = control_error; % 距离误差
    else
        control_mode = 'speed'; 
        error_value = control_error; % 速度误差
    end
    
    % SPPVT PID控制计算
    [target_accel, stage, upgrade_count] = compute_sppvt_control(...
        error_value, control_mode, ego_speed_ms, V_target_kmh, ...
        prev_error, integral_term, dt_sppvt, sppvt_params, ...
        stage, upgrade_count);
    
    % 更新控制器状态
    prev_error = error_value;
    
    % 扭矩仲裁处理
    if torque_arbitration_active
        % 在扭矩仲裁模式下，可能需要调整控制输出
        target_accel = apply_torque_arbitration_logic(target_accel, ego_speed_ms);
    end
    
else
    % 控制未使能时，输出零加速度并重置状态
    target_accel = 0.0;
    integral_term = 0.0; % 重置积分项
    stage = int32(1);    % 重置阶段
end

% 限制输出范围
target_accel = max(sppvt_params.max_decel, min(sppvt_params.max_accel, target_accel));

%% 构造输出
output = struct();
output.target_accel = target_accel;
output.sppvt_stage = stage;
output.sppvt_upgrade_count = upgrade_count;

% 生成调试信息
if mod(debug_sppvt_counter, 20) == 0 % 每20个周期输出一次
    % 调试代码: 2000 + 阶段*10 + (升级次数 mod 10)
    debug_code = 2000 + double(stage)*10 + mod(double(upgrade_count), 10);
    output.debug_message = int32(debug_code);
else
    output.debug_message = debug_sppvt_counter;
end

end

%% SPPVT核心控制算法

function [accel, new_stage, new_upgrade_count] = compute_sppvt_control(...
    error_value, control_mode, ego_speed_ms, V_target_kmh, ...
    prev_error, integral_term, dt, params, current_stage, upgrade_count)
%COMPUTE_SPPVT_CONTROL SPPVT控制算法核心计算
%   基于原Python实现的sppvt_longitudinal_control移植

new_stage = current_stage;
new_upgrade_count = upgrade_count;

% 死区处理
if abs(error_value) < params.deadzone
    accel = 0.0;
    return;
end

%% PID控制计算

% 比例项
P_term = params.kp * error_value;

% 积分项更新
integral_term = integral_term + error_value * dt;

% 积分饱和限制
integral_limit = 5.0; % 积分限幅
integral_term = max(-integral_limit, min(integral_limit, integral_term));
I_term = params.ki * integral_term;

% 微分项
derivative = (error_value - prev_error) / dt;
D_term = params.kd * derivative;

% PID输出
pid_output = P_term + I_term + D_term;

%% SPPVT预设定逻辑

% 基础加速度
base_accel = pid_output;

% 根据控制模式和当前状态进行SPPVT调整
switch control_mode
    case 'distance'
        % 距离控制模式的SPPVT逻辑
        accel = apply_distance_sppvt_logic(base_accel, error_value, ego_speed_ms, ...
            current_stage, params);
            
    case 'speed'
        % 速度控制模式的SPPVT逻辑
        target_speed_ms = V_target_kmh / 3.6;
        accel = apply_speed_sppvt_logic(base_accel, error_value, ego_speed_ms, ...
            target_speed_ms, current_stage, params);
            
    otherwise
        accel = base_accel;
end

%% SPPVT阶段管理

% 根据误差大小和控制效果确定SPPVT阶段
error_threshold_high = 2.0;  % 高误差阈值
error_threshold_low = 0.5;   % 低误差阈值

if abs(error_value) > error_threshold_high && current_stage < 3
    % 误差较大，升级到更高阶段
    new_stage = current_stage + 1;
    new_upgrade_count = upgrade_count + 1;
    
    % 应用更激进的控制策略
    accel = accel * 1.2; % 增加20%的控制强度
    
elseif abs(error_value) < error_threshold_low && current_stage > 1
    % 误差较小，可以降级到更温和的阶段
    new_stage = max(1, current_stage - 1);
    
    % 应用更温和的控制策略
    accel = accel * 0.8; % 降低20%的控制强度
end

% 限制加速度范围
accel = max(params.max_decel, min(params.max_accel, accel));

end

function accel = apply_distance_sppvt_logic(base_accel, distance_error, ego_speed, stage, params)
%APPLY_DISTANCE_SPPVT_LOGIC 距离控制模式的SPPVT逻辑

accel = base_accel;

% 根据距离误差的符号确定控制方向
if distance_error > 0
    % 实际距离小于期望距离，需要减速或保持
    accel = min(accel, 0.5); % 限制最大加速度
    
    % 根据SPPVT阶段调整
    switch stage
        case 1
            accel = accel * 0.8;  % 温和控制
        case 2
            accel = accel * 1.0;  % 正常控制
        case 3
            accel = accel * 1.2;  % 积极控制
    end
    
else
    % 实际距离大于期望距离，可以加速
    accel = max(accel, -1.0); % 限制最大减速度
    
    % SPPVT预设定：根据距离余量进行加速预设定
    distance_margin = abs(distance_error);
    if distance_margin > 10.0 % 距离余量较大
        accel = accel + 0.3; % 增加预设定加速度
    end
end

% 速度相关的安全限制
if ego_speed > 25.0 % 高速时更保守
    accel = accel * 0.9;
elseif ego_speed < 5.0 % 低速时更积极
    accel = max(accel, -0.5); % 避免过度减速
end

end

function accel = apply_speed_sppvt_logic(base_accel, speed_error, ego_speed, target_speed, stage, params)
%APPLY_SPEED_SPPVT_LOGIC 速度控制模式的SPPVT逻辑

accel = base_accel;

% 速度差预测和预设定
speed_diff = target_speed - ego_speed;

if speed_diff > 0
    % 需要加速到目标速度
    % SPPVT预设定：根据速度差进行预加速
    if abs(speed_diff) > 5.0 % 速度差较大
        accel = accel + 0.5; % 预设定加速
    elseif abs(speed_diff) > 2.0 % 中等速度差
        accel = accel + 0.2; % 适度预设定
    end
    
else
    % 需要减速到目标速度
    % SPPVT预设定：提前减速
    if abs(speed_diff) > 5.0 % 速度差较大
        accel = accel - 0.5; % 预设定减速
    elseif abs(speed_diff) > 2.0 % 中等速度差
        accel = accel - 0.2; % 适度预设定
    end
end

% 根据SPPVT阶段调整控制强度
switch stage
    case 1 % 基础阶段
        accel = accel * 0.8;
    case 2 % 标准阶段
        accel = accel * 1.0;
    case 3 % 高级阶段
        accel = accel * 1.1;
        % 高级阶段加入更复杂的预测逻辑
        if abs(speed_diff) < 1.0 && abs(speed_error) < 0.5
            accel = accel * 0.7; % 接近目标时更温和
        end
end

end

function adjusted_accel = apply_torque_arbitration_logic(base_accel, ego_speed)
%APPLY_TORQUE_ARBITRATION_LOGIC 扭矩仲裁逻辑
%   在扭矩仲裁模式下调整SPPVT输出

adjusted_accel = base_accel;

% 扭矩仲裁时的调整策略
if base_accel > 0
    % 加速指令：在仲裁模式下适当降低
    adjusted_accel = base_accel * 0.8;
else
    % 减速指令：在仲裁模式下保持原有强度（安全优先）
    adjusted_accel = base_accel;
end

% 速度相关的仲裁调整
if ego_speed > 20.0 % 高速时更保守
    adjusted_accel = adjusted_accel * 0.9;
end

end