function integrated_output = fcn(decision_output, sppvt_output1, sppvt_output2, sppvt_output3, sppvt_output4, sppvt_output5, new_stage_offset, new_stage, new_error_sign, new_upgrade_count, new_control_error, new_error_derivative, new_error_second_derivative)
%#codegen
% Output_Formatter - 21/22-field state externalization version (all scalar)
% Integrates decision, SPPVT output and state output for Python state capture
%
% Inputs:
%   decision_output - DecisionSPPVTOutput from Decision_Function (includes next_state and other decision state output fields)
%   sppvt_output1-5 - 5 outputs from SPPVT_Control (control, velocity, acceleration, jerk, status)
%   new_stage_offset - Stage offset (scalar)
%   new_stage - Stage Manager new stage (scalar)
%   new_error_sign - Stage Manager new error sign (scalar)
%   new_upgrade_count - Stage Manager new upgrade count (scalar)
%   new_control_error - New control error (scalar)
%   new_error_derivative - New control error derivative (scalar)
%   new_error_second_derivative - New control error second derivative (scalar)
%
% Output: integrated_output - DecisionSPPVTOutputExtended (22 fields)

%% 构造集成输出结构
integrated_output = struct();

%% 按照总线定义顺序赋值所有字段（确保代码生成兼容性）
% 根据DecisionSPPVTOutputExtended总线定义的22字段顺序

% 1-6: 来自决策系统的信息
integrated_output.control_enabled = decision_output.control_enabled;
integrated_output.current_state = decision_output.current_state;
integrated_output.current_decision = decision_output.current_decision;
integrated_output.torque_arbitration_active = decision_output.torque_arbitration_active;
integrated_output.updated_V_target_kmh = decision_output.updated_V_target_kmh;
integrated_output.updated_G2_s = decision_output.updated_G2_s;

% 7-11: SPPVT controller outputs (double precision)
integrated_output.sppvt_control_output = double(sppvt_output1);
integrated_output.sppvt_velocity_output = double(sppvt_output2);
integrated_output.sppvt_acceleration_output = double(sppvt_output3);
% 注意：sppvt_output4是jerk，不是stage，stage来自Stage_Manager
integrated_output.sppvt_stage_output = double(new_stage);  % 从Stage_Manager获取stage值（标量）
integrated_output.sppvt_status_output = double(sppvt_output5);

% 12: Debug information (composite code)
debug_code = int32(3000 + decision_output.current_state*100 + ...
                 mod(int32(sppvt_output5), 10)*10 + ...
                 int32(decision_output.control_enabled));
integrated_output.debug_message = debug_code;

% 13-15: Decision module state outputs
integrated_output.next_state = decision_output.next_state;
integrated_output.next_has_history = decision_output.next_has_history;
integrated_output.next_last_active_decision = decision_output.next_last_active_decision;

% 16-22: SPPVT state outputs consumed in Python (全部标量化)

% 16: Stage offset (scalar)
integrated_output.new_stage_offset = double(new_stage_offset);

% 17-19: Stage Manager states (全部标量)
integrated_output.new_stage = double(new_stage);
integrated_output.new_error_sign = double(new_error_sign);
integrated_output.new_upgrade_count = double(new_upgrade_count);

% 20-22: Adapter states - 控制误差及其导数 (全部标量)
% 修正：用SPPVT实际计算的导数值替换Adapter传来的占位值
integrated_output.new_control_error = double(new_control_error);
integrated_output.new_error_derivative = double(sppvt_output2);  % 误差的一阶导数 (velocity)
integrated_output.new_error_second_derivative = double(sppvt_output3);  % 误差的二阶导数 (acceleration)

% 调试输出（22字段版本，包含完整状态外化信息）
if mod(decision_output.debug_message, 50) == 0
    fprintf("Output_Formatter Input: State=%d, Decision=%d, Control=%d, Debug=%d\n", ...
            int32(decision_output.current_state), int32(decision_output.current_decision), ...
            int32(decision_output.control_enabled), int32(decision_output.debug_message));
    fprintf("Integrated Output: State=S%d->S%d, Decision=R%d, Control=%d, SPPVT=%.3f\n", ...
            int32(decision_output.current_state), int32(decision_output.next_state), ...
            int32(decision_output.current_decision), int32(decision_output.control_enabled), double(sppvt_output1));

    % 决策状态外化调试输出
    fprintf("Decision State: next_state=S%d, next_history=%d, next_decision=R%d\n", ...
            int32(decision_output.next_state), int32(decision_output.next_has_history), ...
            int32(decision_output.next_last_active_decision));

    % SPPVT状态外化调试输出（标量版本）
    fprintf("SPPVT State: StageOffset=%.3f, Stage=%.0f, ErrorSign=%.0f, UpgradeCount=%.0f\n", ...
            double(new_stage_offset), double(new_stage), double(new_error_sign), double(new_upgrade_count));
    fprintf("Error States: ControlError=%.3f, Derivative=%.3f, SecondDerivative=%.3f\n", ...
            double(new_control_error), double(sppvt_output2), double(sppvt_output3));
end

end