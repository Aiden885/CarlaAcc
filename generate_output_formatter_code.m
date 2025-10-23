function integrated_output = fcn(decision_output, sppvt_output1, sppvt_output2, sppvt_output3, sppvt_output4, sppvt_output5, new_stage_offset, stage_manager_states_out, adapter_states_out)
%#codegen
% Output_Formatter - 17/18-field state externalization version
% Integrates decision, SPPVT output and state output for Python state capture
%
% Inputs:
%   decision_output - DecisionSPPVTOutput from Decision_Function (includes next_state and other decision state output fields)
%   sppvt_output1-5 - 5 outputs from SPPVT_Control (control, velocity, acceleration, stage, status)
% 16: Stage offset (scalar)
%   stage_manager_states_out - [new_stage, new_error_sign, new_upgrade_count] array
%   adapter_states_out - [new_control_error, new_velocity, new_accel] array
%
% Output: integrated_output - DecisionSPPVTOutputExtended (18 fields)

%% 鏋勯€犻泦鎴愯緭鍑虹粨鏋?
integrated_output = struct();

%% 鎸夌収鎬荤嚎瀹氫箟椤哄簭璧嬪€兼墍鏈夊瓧娈碉紙纭繚浠ｇ爜鐢熸垚鍏煎鎬э級
% 鏍规嵁DecisionSPPVTOutputExtended鎬荤嚎瀹氫箟鐨?8瀛楁椤哄簭

% 1-6: 鏉ヨ嚜鍐崇瓥绯荤粺鐨勪俊鎭?
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
% 娉ㄦ剰锛歴ppvt_output4鏄痡erk锛屼笉鏄痵tage锛乻tage鏉ヨ嚜Stage_Manager
integrated_output.sppvt_stage_output = double(stage_manager_states_out(1));  % 浠嶴tage_Manager鑾峰彇stage鍊?
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

% 16-18: SPPVT state outputs consumed in Python

% 16: Stage offset (scalar)
integrated_output.new_stage_offset = double(new_stage_offset);

% 17: Stage Manager state [new_stage, new_error_sign, new_upgrade_count]
integrated_output.new_stage_manager_states = double(stage_manager_states_out);

% 18: Adapter鐘舵€?(鏁扮粍) - [new_control_error, new_velocity, new_accel]
% 淇锛氱敤SPPVT瀹為檯璁＄畻鐨勫鏁板€兼浛鎹dapter浼犳潵鐨勫崰浣嶅€?
adapter_states_corrected = adapter_states_out;

% 璋冭瘯锛氭墦鍗?adapter_states_out 鐨勭淮搴﹀拰鍊?
fprintf('DEBUG: adapter_states_out size = [%d, %d], values = [', int32(size(adapter_states_out, 1)), int32(size(adapter_states_out, 2)));
for i = 1:length(adapter_states_out)
    fprintf('%.3f ', adapter_states_out(i));
end
fprintf(']
');

adapter_states_corrected(2) = double(sppvt_output2);  % new_velocity = 璇樊鐨勪竴闃跺鏁?
adapter_states_corrected(3) = double(sppvt_output3);  % new_accel = 璇樊鐨勪簩闃跺鏁?

fprintf('DEBUG: adapter_states_corrected size = [%d, %d], values = [', int32(size(adapter_states_corrected, 1)), int32(size(adapter_states_corrected, 2)));
for i = 1:length(adapter_states_corrected)
    fprintf('%.3f ', adapter_states_corrected(i));
end
fprintf(']
');

integrated_output.new_adapter_states = double(adapter_states_corrected);
fprintf('DEBUG: integrated_output.new_adapter_states = [');
for i = 1:length(integrated_output.new_adapter_states)
    fprintf('%.3f ', integrated_output.new_adapter_states(i));
end
fprintf(']\\n');

% 缁村害妫€鏌ユ柇瑷€ - 纭繚杈撳嚭鏄?鍏冪礌
assert(length(adapter_states_corrected) == 3, 'ERROR: adapter_states_corrected must have 3 elements!');
fprintf('DEBUG: Assigned new_adapter_states, size = [%d, %d]\n', ...
        int32(size(integrated_output.new_adapter_states, 1)), ...
        int32(size(integrated_output.new_adapter_states, 2)));

% 璋冭瘯杈撳嚭锛?8瀛楁鐗堟湰锛屽寘鍚畬鏁寸姸鎬佸鍖栦俊鎭級
if mod(decision_output.debug_message, 50) == 0
  fprintf("Output_Formatter Input: State=%d, Decision=%d, Control=%d, Debug=%d\n", ...
          int32(decision_output.current_state), int32(decision_output.current_decision), ...
          int32(decision_output.control_enabled), int32(decision_output.debug_message));
  fprintf("Integrated Output: State=S%d->S%d, Decision=R%d, Control=%d, SPPVT=%.3f\n", ...
          int32(decision_output.current_state), int32(decision_output.next_state), ...
          int32(decision_output.current_decision), int32(decision_output.control_enabled), double(sppvt_output1));

  % 鍐崇瓥鐘舵€佸鍖栬皟璇曡緭鍑?
  fprintf("Decision State: next_state=S%d, next_history=%d, next_decision=R%d\n", ...
          int32(decision_output.next_state), int32(decision_output.next_has_history), ...
          int32(decision_output.next_last_active_decision));

  % SPPVT鐘舵€佸鍖栬皟璇曡緭鍑?
  fprintf("SPPVT State: StageOffset=%.3f, StageStates=[%.0f,%.0f,%.0f], AdapterStates=[%.3f,%.3f,%.3f]\n", ...
          double(new_stage_offset), ...
          stage_manager_states_out(1), stage_manager_states_out(2), stage_manager_states_out(3), ...
          adapter_states_corrected(1), adapter_states_corrected(2), adapter_states_corrected(3));
end

end
