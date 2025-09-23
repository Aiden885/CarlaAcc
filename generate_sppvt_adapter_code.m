function [error_value, dt, stage_offset, kp, max_accel, max_decel, prev_error_out, prev_velocity_out, prev_accel_out, delta, eta, mode_flag] = fcn(validated_input, decision_output)
%#codegen
% SPPVT接口适配器 - 12个独立输出端口版本
% 输入: validated_input (验证后的输入), decision_output (决策输出)
% 输出: 12个独立信号匹配sppvt_control_model的12个输入端口

%% 初始化persistent变量（SPPVT状态管理）
persistent current_stage_offset prev_error prev_velocity prev_accel
if isempty(current_stage_offset)
    current_stage_offset = 0.0;
    prev_error = 0.0;
    prev_velocity = 13.89;  % 初始速度设为当前车速 (50 km/h = 13.89 m/s)
    prev_accel = 0.1;       % 初始加速度设为小的正值避免除零
end

%% 根据决策结果确定控制误差
if decision_output.control_enabled
    % 控制激活时使用实际误差
    error_value = validated_input.control_error;
else
    % 控制未激活时误差为0
    error_value = 0.0;
end

%% 分配12个独立输出信号
dt = 0.05;                                      % [2] dt 时间步长 (硬编码为50ms)
stage_offset = current_stage_offset;            % [3] 当前级差 (状态变量)
kp = 1.0;                                       % [4] sppvt_kp 比例系数 (硬编码)
max_accel = 2.0;                                % [5] max_accel 最大加速度 (硬编码)
max_decel = -3.0;                               % [6] max_decel 最大减速度 (硬编码)
prev_error_out = prev_error;                    % [7] 上次误差 (状态变量)
prev_velocity_out = prev_velocity;              % [8] 上次速度 (状态变量)
prev_accel_out = prev_accel;                    % [9] 上次加速度 (状态变量)
delta = 0.05;                                   % [10] sppvt_delta 控制参数 (与Python实现一致)
eta = 0.2;                                      % [11] sppvt_eta 控制参数 (与Python实现一致)
mode_flag = double(validated_input.control_mode_flag); % [12] 控制模式标志

% 更新历史状态
prev_error = error_value;

% 调试输出 (修复类型转换) - 添加详细参数检查，特别是历史状态和关键参数
if abs(error_value) > 0.01
    fprintf("SPPVT Adapter: Error=%.3f, Speed=%.3f, dt=%.3f, delta=%.3f, eta=%.3f, Mode=%d, Control=%d\n", ...
            error_value, validated_input.ego_speed_ms, dt, delta, eta, int32(validated_input.control_mode_flag), int32(decision_output.control_enabled));
end
end