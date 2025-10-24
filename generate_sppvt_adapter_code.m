function [error_value, dt, stage_offset, kp, max_accel, max_decel, prev_error_out, prev_velocity_out, prev_accel_out, delta, eta, mode_flag, new_control_error, new_error_derivative, new_error_second_derivative] = fcn(validated_input, decision_output, external_stage_offset, external_control_error, external_error_derivative, external_error_second_derivative)
%#codegen
% SPPVT接口适配器 - 支持外部状态注入版本 (全部标量化)
% 输入:
%   validated_input - 验证后的输入总线
%   decision_output - 决策输出总线
%   external_stage_offset - 外部级差状态
%   external_control_error - 外部控制误差
%   external_error_derivative - 外部控制误差的导数
%   external_error_second_derivative - 外部控制误差的二阶导数
% 输出:
%   1-12: 12个独立信号匹配sppvt_control_model的12个输入端口
%   13-15: 新的误差状态（标量化）- new_control_error, new_error_derivative, new_error_second_derivative

%% 直接使用外部状态（从Bus Selector传入）- 全部标量化
current_stage_offset = external_stage_offset;

% 直接使用输入的标量参数（不再需要从数组提取）
% 注意：velocity和accel实际上是误差的一阶和二阶导数
prev_error = external_control_error;
prev_velocity = external_error_derivative;           % 控制误差的一阶导数
prev_accel = external_error_second_derivative;       % 控制误差的二阶导数

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

%% 输出新的误差状态（第13-15个输出）- 全部标量化
% 用于Output_Formatter的误差状态字段
% 注意：这些是当前计算的误差值，导数将由SPPVT模块计算
new_control_error = error_value;
new_error_derivative = external_error_derivative;  % 保持输入值，将由SPPVT更新
new_error_second_derivative = external_error_second_derivative;  % 保持输入值，将由SPPVT更新

% 调试输出 - 显示关键参数
if abs(error_value) > 0.01
    fprintf("SPPVT Adapter: Error=%.3f, Speed=%.3f, Stage=%.3f, PrevError=%.3f, Mode=%d, Control=%d\n", ...
            error_value, validated_input.ego_speed_ms, current_stage_offset, prev_error, ...
            int32(validated_input.control_mode_flag), int32(decision_output.control_enabled));
end
end