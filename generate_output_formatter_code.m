function integrated_output = fcn(decision_output, sppvt_output1, sppvt_output2, sppvt_output3, sppvt_output4, sppvt_output5, new_stage_offset, stage_manager_states_out, adapter_states_out)
%#codegen
% Output_Formatter - 17/18-field state externalization version
% 集成决策、SPPVT输出和状态输出，用于Python状态捕获
%
% 输入:
%   decision_output - Decision_Function的DecisionSPPVTOutput (包含next_state等决策状态输出字段)
%   sppvt_output1-5 - SPPVT_Control的5个输出 (control, velocity, acceleration, stage, status)
%   new_stage_offset - Stage_Manager的级差输出
%   stage_manager_states_out - [new_stage, new_error_sign, new_upgrade_count] 数组
%   adapter_states_out - [new_control_error, new_velocity, new_accel] 数组
%
% 输出: integrated_output - DecisionSPPVTOutputExtended (18字段)

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
integrated_output.sppvt_stage_output = double(sppvt_output4);
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
% 特殊处理：用实际SPPVT控制输出覆盖new_accel
adapter_states_corrected = adapter_states_out;
adapter_states_corrected(3) = double(sppvt_output1);  % 用真实SPPVT输出作为new_accel

integrated_output.new_adapter_states = double(adapter_states_corrected);

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
