function integrated_output = fcn(decision_output, sppvt_output1, sppvt_output2, sppvt_output3, sppvt_output4, sppvt_output5)
%#codegen
% Output_Formatter - 集成决策和SPPVT的5个输出
% 输入: decision_output (决策输出), sppvt_output1-5 (SPPVT的5个输出)
% 输出: integrated_output (集成输出总线)

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

% 7: 调试信息（组合状态码）
debug_code = int32(3000 + decision_output.current_state*100 + ...
                 mod(int32(sppvt_output5), 10)*10 + ...
                 int32(decision_output.control_enabled));
integrated_output.debug_message = debug_code;

% 8-12: 来自SPPVT控制系统的输出（5个信号）- 确保类型转换
integrated_output.sppvt_control_output = double(sppvt_output1);
integrated_output.sppvt_velocity_output = double(sppvt_output2);
integrated_output.sppvt_acceleration_output = double(sppvt_output3);
integrated_output.sppvt_stage_output = double(sppvt_output4);
integrated_output.sppvt_status_output = double(sppvt_output5);

% 调试输出（修复类型转换） - 添加输入检查
if mod(decision_output.debug_message, 50) == 0
  fprintf("Output_Formatter Input: State=%d, Decision=%d, Control=%d, Debug=%d\n", ...
          int32(decision_output.current_state), int32(decision_output.current_decision), ...
          int32(decision_output.control_enabled), int32(decision_output.debug_message));
  fprintf("Integrated Output: State=S%d, Decision=D%d, Control=%d, SPPVT=%.3f\n", ...
          int32(decision_output.current_state), int32(decision_output.current_decision), ...
          int32(decision_output.control_enabled), double(sppvt_output1));
end

end
