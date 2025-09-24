function [error_value, dt, stage_offset, kp, max_accel, max_decel, prev_error_out, prev_velocity_out, prev_accel_out, delta, eta, mode_flag] = fcn(validated_input, decision_output)
%#codegen
% SPPVT接口适配器 - 支持外部状态注入版本 (通过总线)
% 输入: validated_input (验证后的输入，含外部状态), decision_output (决策输出)
% 输出: 12个独立信号匹配sppvt_control_model的12个输入端口

%% 使用外部状态（从总线输入中获取）
% 检查是否有外部状态字段，如果有则使用外部状态，否则使用内部persistent变量
persistent internal_stage_offset internal_prev_error internal_prev_velocity internal_prev_accel

% 初始化内部persistent变量作为备份
if isempty(internal_stage_offset)
    internal_stage_offset = 0.0;
    internal_prev_error = 0.0;
    internal_prev_velocity = 13.89;  % 50 km/h = 13.89 m/s
    internal_prev_accel = 0.1;
end

% 优先使用外部状态，如果不可用则回退到内部状态
if isfield(validated_input, 'external_stage_offset') && ...
   isfield(validated_input, 'external_prev_error') && ...
   isfield(validated_input, 'external_prev_velocity') && ...
   isfield(validated_input, 'external_prev_accel')
    % 使用外部状态（Python状态管理器注入）
    current_stage_offset = validated_input.external_stage_offset;
    prev_error = validated_input.external_prev_error;
    prev_velocity = validated_input.external_prev_velocity;
    prev_accel = validated_input.external_prev_accel;
    fprintf("SPPVT Adapter: 使用外部状态 - Stage=%.3f, PrevErr=%.3f\n", current_stage_offset, prev_error);
else
    % 回退到内部persistent变量
    current_stage_offset = internal_stage_offset;
    prev_error = internal_prev_error;
    prev_velocity = internal_prev_velocity;
    prev_accel = internal_prev_accel;
    fprintf("SPPVT Adapter: 使用内部状态 (外部状态不可用)\n");
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

% 更新内部状态（仅在使用内部状态时）
if ~(isfield(validated_input, 'external_stage_offset') && ...
     isfield(validated_input, 'external_prev_error') && ...
     isfield(validated_input, 'external_prev_velocity') && ...
     isfield(validated_input, 'external_prev_accel'))
    % 只有在使用内部状态时才更新内部persistent变量
    internal_prev_error = error_value;
    internal_prev_velocity = validated_input.ego_speed_ms;
    % internal_prev_accel 需要通过Stage_Manager反馈更新
end

% 调试输出 - 显示状态来源和关键参数
if abs(error_value) > 0.01
    state_source = "External";
    if ~isfield(validated_input, 'external_stage_offset')
        state_source = "Internal";
    end
    fprintf("SPPVT Adapter (%s): Error=%.3f, Speed=%.3f, Stage=%.3f, Mode=%d, Control=%d\n", ...
            state_source, error_value, validated_input.ego_speed_ms, current_stage_offset, ...
            int32(validated_input.control_mode_flag), int32(decision_output.control_enabled));
end
end