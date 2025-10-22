function [error_value, dt, stage_offset, kp, max_accel, max_decel, prev_error_out, prev_velocity_out, prev_accel_out, delta, eta, mode_flag, adapter_states_out] = fcn(validated_input, decision_output, external_stage_offset, external_adapter_states)
%#codegen
% SPPVT接口适配器 - 支持外部状态注入版本 (通过Bus Selector)
% 输入:
%   validated_input - 验证后的输入总线
%   decision_output - 决策输出总线
%   external_stage_offset - 外部级差状态 (从Bus Selector1/1)
%   external_adapter_states - 外部适配器状态 [prev_error, prev_velocity, prev_accel] (从Bus Selector1/3)
% 输出:
%   1-12: 12个独立信号匹配sppvt_control_model的12个输入端口
%   13: adapter_states_out - [new_control_error, new_velocity, new_accel] 用于Output_Formatter

%% 直接使用外部状态（从Bus Selector传入）
% 修复：直接使用Bus Selector提取的字段，不需要isfield()检查
current_stage_offset = external_stage_offset;

% 从3元素数组中提取 [prev_error, prev_velocity, prev_accel]
if length(external_adapter_states) >= 3
    prev_error = external_adapter_states(1);
    prev_velocity = external_adapter_states(2);
    prev_accel = external_adapter_states(3);
else
    % 如果数组长度不足，使用默认值
    prev_error = 0.0;
    prev_velocity = 13.89;  % 50 km/h = 13.89 m/s
    prev_accel = 0.0;
    fprintf("SPPVT Adapter: Warning - external_adapter_states length < 3, using defaults\n");
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

%% 输出adapter状态数组（第13个输出）
% 用于Output_Formatter的new_adapter_states字段
% 格式: [new_control_error, new_velocity, new_accel]
% 注意：第2、3个元素是占位值，将在Output_Formatter中被SPPVT实际输出替换
adapter_states_out = [error_value; 0.0; 0.0];

% 调试输出 - 显示关键参数
if abs(error_value) > 0.01
    fprintf("SPPVT Adapter: Error=%.3f, Speed=%.3f, Stage=%.3f, PrevAccel=%.3f, Mode=%d, Control=%d\n", ...
            error_value, validated_input.ego_speed_ms, current_stage_offset, prev_accel, ...
            int32(validated_input.control_mode_flag), int32(decision_output.control_enabled));
end
end