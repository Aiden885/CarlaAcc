function integrated_output = fcn(decision_output, sppvt_output1, sppvt_output2, sppvt_output3, sppvt_output4, sppvt_output5, new_stage_offset, stage_manager_states_out, adapter_states_out)
%#codegen
% Output_Formatter - 17/18-field state externalization version
% Integrates decision, SPPVT output and state output for Python state capture
%
% Inputs:
%   decision_output - DecisionSPPVTOutput from Decision_Function (includes next_state and other decision state output fields)
%   sppvt_output1-5 - 5 outputs from SPPVT_Control (control, velocity, acceleration, stage, status)
%   new_stage_offset - Stage offset output from Stage_Manager
%   stage_manager_states_out - [new_stage, new_error_sign, new_upgrade_count] array
%   adapter_states_out - [new_control_error, new_velocity, new_accel] array
%
% Output: integrated_output - DecisionSPPVTOutputExtended (18 fields)

%% 构造集成输出结构
integrated_output = struct();

%% 按照总线定义顺序赋值所有字段（确保代码生成兼容性）
% 根据DecisionSPPVTOutputExtended总线定义的18字段顺序

% 1-6: 来自决策系统的信息
integrated_output.control_enabled = decision_output.control_enabled;
integrated_output.current_state = decision_output.current_state;
integrated_output.current_decision = decision_output.current_decision;
integrated_output.torque_arbitration_active = decision_output.torque_arbitration_active;
integrated_output.updated_V_target_kmh = decision_output.updated_V_target_kmh;
integrated_output.updated_G2_s = decision_output.updated_G2_s;

% 7-11: 来自SPPVT控制系统的输出（5个信号）- 确保类型转换
integrated_output.sppvt_control_output = double(sppvt_output1);
integrated_output.sppvt_velocity_output = double(sppvt_output2);
integrated_output.sppvt_acceleration_output = double(sppvt_output3);
% 注意：sppvt_output4是jerk，不是stage！stage来自Stage_Manager
integrated_output.sppvt_stage_output = double(stage_manager_states_out(1));  % 从Stage_Manager获取stage值
integrated_output.sppvt_status_output = double(sppvt_output5);

% 12: 调试信息（组合状态码）
debug_code = int32(3000 + decision_output.current_state*100 + ...
                 mod(int32(sppvt_output5), 10)*10 + ...
                 int32(decision_output.control_enabled));
integrated_output.debug_message = debug_code;

% 13-15: 来自决策系统的状态输出字段（新增）
integrated_output.next_state = decision_output.next_state;
integrated_output.next_has_history = decision_output.next_has_history;
integrated_output.next_last_active_decision = decision_output.next_last_active_decision;

% 16-18: SPPVT状态输出字段 - 用于Python状态管理

% 16: Stage offset状态 (标量)
integrated_output.new_stage_offset = double(new_stage_offset);

% 17: Stage Manager状态 (数组) - [new_stage, new_error_sign, new_upgrade_count]
integrated_output.new_stage_manager_states = double(stage_manager_states_out);

% 18: Adapter状态 (数组) - [new_control_error, new_velocity, new_accel]
% 修正：用SPPVT实际计算的导数值替换adapter传来的占位值
adapter_states_corrected = adapter_states_out;

% 调试：打印 adapter_states_out 的维度和值
fprintf('DEBUG: adapter_states_out size = [%d, %d], values = [', int32(size(adapter_states_out, 1)), int32(size(adapter_states_out, 2)));
for i = 1:length(adapter_states_out)
    fprintf('%.3f ', adapter_states_out(i));
end
fprintf(']\n');

adapter_states_corrected(2) = double(sppvt_output2);  % new_velocity = 误差的一阶导数
adapter_states_corrected(3) = double(sppvt_output3);  % new_accel = 误差的二阶导数

fprintf('DEBUG: adapter_states_corrected size = [%d, %d], values = [', int32(size(adapter_states_corrected, 1)), int32(size(adapter_states_corrected, 2)));
for i = 1:length(adapter_states_corrected)
    fprintf('%.3f ', adapter_states_corrected(i));
end
fprintf(']\n');

integrated_output.new_adapter_states = double(adapter_states_corrected);

% 维度检查断言 - 确保输出是3元素
assert(length(adapter_states_corrected) == 3, 'ERROR: adapter_states_corrected must have 3 elements!');
fprintf('DEBUG: Assigned new_adapter_states, size = [%d, %d]\n', ...
        int32(size(integrated_output.new_adapter_states, 1)), ...
        int32(size(integrated_output.new_adapter_states, 2)));

% 调试输出（18字段版本，包含完整状态外化信息）
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

  % SPPVT状态外化调试输出
  fprintf("SPPVT State: StageOffset=%.3f, StageStates=[%.0f,%.0f,%.0f], AdapterStates=[%.3f,%.3f,%.3f]\n", ...
          double(new_stage_offset), ...
          stage_manager_states_out(1), stage_manager_states_out(2), stage_manager_states_out(3), ...
          adapter_states_corrected(1), adapter_states_corrected(2), adapter_states_corrected(3));
end

end
