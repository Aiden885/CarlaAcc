% 自动修复Simulink模型配置

% 修复SPPVT模型
set_param('sppvt_control_model', 'StopTime', 'inf');
set_param('sppvt_control_model', 'Solver', 'FixedStepDiscrete');
set_param('sppvt_control_model', 'FixedStep', '0.05');
fprintf('✅ SPPVT模型配置已修复\n');


fprintf('✅ SPPVT参数已设置\n');
