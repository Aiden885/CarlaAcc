function validated_input = fcn(raw_input)
%#codegen
%
% 输入验证和预处理
% 输入: raw_input (DecisionSPPVTInput总线)
% 输出: validated_input (经过验证和修正的总线)

validated_input = raw_input;

% 接下来，在输出变量上执行数值范围检查，防止异常值进入下游模块。
validated_input.ego_speed_kmh = max(0, min(200, raw_input.ego_speed_kmh));
validated_input.ego_speed_ms = max(0, min(60, raw_input.ego_speed_ms));
validated_input.V_target_kmh = max(30, min(150, raw_input.V_target_kmh));
validated_input.V_min_kmh = max(20, min(50, raw_input.V_min_kmh));
validated_input.G2_s = max(1.0, min(8.0, raw_input.G2_s));


% 确保最低车速始终小于或等于目标车速。
if validated_input.V_min_kmh > validated_input.V_target_kmh
    validated_input.V_min_kmh = validated_input.V_target_kmh - 10;
end

% 时间戳直接透传
validated_input.timestamp = raw_input.timestamp;

end