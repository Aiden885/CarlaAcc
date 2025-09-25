function integrated_output = fcn(decision_output, sppvt_output1, sppvt_output2, sppvt_output3, sppvt_output4, sppvt_output5, new_stage_offset, stage_manager_states_out, adapter_states_out)
%#codegen
% Output_Formatter - 14/15-field state externalization version
% 集成决策、SPPVT输出和状态输出，用于Python状态捕获
%
% 输入:
%   decision_output - Decision_Function的DecisionSPPVTOutput
%   sppvt_output1-5 - SPPVT_Control的5个输出 (control, velocity, acceleration, stage, status)
%   new_stage_offset - Stage_Manager的级差输出
%   stage_manager_states_out - [new_stage, new_error_sign, new_upgrade_count] 数组
%   adapter_states_out - [new_control_error, new_velocity, new_accel] 数组
%
% 输出: integrated_output - DecisionSPPVTOutputExtended (15字段)

%% 构造集成输出结构
integrated_output = struct();

%% 按照总线定义顺序赋值所有字段（确保代码生成兼容性）

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

%% 新增的3个状态输出字段 - 用于Python状态管理

% 13: Stage offset状态 (标量)
integrated_output.new_stage_offset = double(new_stage_offset);

% 14: Stage Manager状态 (数组) - [new_stage, new_error_sign, new_upgrade_count]
integrated_output.new_stage_manager_states = double(stage_manager_states_out);

% 15: Adapter状态 (数组) - [new_control_error, new_velocity, new_accel]
% 特殊处理：用实际SPPVT控制输出覆盖new_accel
adapter_states_corrected = adapter_states_out;
adapter_states_corrected(3) = double(sppvt_output1);  % 用真实SPPVT输出作为new_accel

integrated_output.new_adapter_states = double(adapter_states_corrected);

% 调试输出（扩展版本，包含状态外化信息） - 添加输入检查
if mod(decision_output.debug_message, 50) == 0
  fprintf("Output_Formatter Input: State=%d, Decision=%d, Control=%d, Debug=%d\n", ...
          int32(decision_output.current_state), int32(decision_output.current_decision), ...
          int32(decision_output.control_enabled), int32(decision_output.debug_message));
  fprintf("Integrated Output: State=S%d, Decision=D%d, Control=%d, SPPVT=%.3f\n", ...
          int32(decision_output.current_state), int32(decision_output.current_decision), ...
          int32(decision_output.control_enabled), double(sppvt_output1));

  % 新增：状态外化调试输出
  fprintf("State Externalization: StageOffset=%.3f, StageStates=[%.0f,%.0f,%.0f], AdapterStates=[%.3f,%.3f,%.3f]\n", ...
          double(new_stage_offset), ...
          stage_manager_states_out(1), stage_manager_states_out(2), stage_manager_states_out(3), ...
          adapter_states_corrected(1), adapter_states_corrected(2), adapter_states_corrected(3));
end

end
