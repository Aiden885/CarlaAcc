%LOAD_ACC_SPPVT_PARAMETERS ACC决策+SPPVT一体化模型参数加载脚本
%   适用于MATLAB 2024b，加载模型中使用的所有参数到工作区
%   
%   此脚本直接执行参数加载，无需函数调用

fprintf('⚙️  加载ACC决策+SPPVT一体化模型参数...\n');

%% 基础控制参数
params = struct();

%% ACC决策参数
params.decision = struct();

% 速度相关参数
params.decision.V_target_default_kmh = 50.0;    % 默认目标速度 km/h
params.decision.V_min_kmh = 30.0;               % 最小控制速度 km/h
params.decision.V_max_kmh = 120.0;              % 最大控制速度 km/h
params.decision.V_threshold_kmh = 50.0;         % 两模式切换阈值速度 km/h

% 时距相关参数
params.decision.G2_default_s = 2.0;             % 默认时距参数 s
params.decision.G2_min_s = 1.0;                 % 最小时距 s
params.decision.G2_max_s = 3.0;                 % 最大时距 s

% 控制启动条件
params.decision.min_activation_speed_kmh = 30.0; % 最低启控速度 km/h
params.decision.max_activation_speed_kmh = 100.0; % 最高启控速度 km/h

% 状态转换时间参数
params.decision.standby_timeout_s = 10.0;       % 待命状态超时时间 s
params.decision.arbitration_timeout_s = 5.0;    % 扭矩仲裁超时时间 s

%% SPPVT控制参数
params.sppvt = struct();

% PID控制参数
params.sppvt.kp = 0.8;                          % 比例增益
params.sppvt.ki = 0.1;                          % 积分增益
params.sppvt.kd = 0.05;                         % 微分增益

% 控制限制
params.sppvt.max_accel_ms2 = 2.0;               % 最大加速度 m/s²
params.sppvt.max_decel_ms2 = -4.0;              % 最大减速度 m/s²
params.sppvt.deadzone = 0.1;                    % 控制死区

% 积分项限制
params.sppvt.integral_limit = 5.0;              % 积分饱和限制
params.sppvt.integral_reset_threshold = 10.0;   % 积分复位阈值

% SPPVT阶段参数
params.sppvt.stage1_gain = 0.8;                 % 阶段1增益系数
params.sppvt.stage2_gain = 1.0;                 % 阶段2增益系数
params.sppvt.stage3_gain = 1.2;                 % 阶段3增益系数

% SPPVT升级阈值
params.sppvt.upgrade_error_threshold = 2.0;     % 升级误差阈值
params.sppvt.downgrade_error_threshold = 0.5;   % 降级误差阈值

%% 两模式控制参数
params.two_mode = struct();

% 模式切换参数
params.two_mode.transition_zone_kmh = 2.8;      % 模式切换过渡区 km/h
params.two_mode.hysteresis_kmh = 1.0;           % 迟滞范围 km/h

% 距离控制模式参数
params.two_mode.distance_kp = 0.5;              % 距离控制比例系数
params.two_mode.distance_ki = 0.02;             % 距离控制积分系数
params.two_mode.distance_kd = 0.1;              % 距离控制微分系数

% 速度控制模式参数
params.two_mode.speed_kp = 0.8;                 % 速度控制比例系数
params.two_mode.speed_ki = 0.05;                % 速度控制积分系数
params.two_mode.speed_kd = 0.02;                % 速度控制微分系数

%% 系统参数
params.system = struct();

% 采样时间
params.system.sample_time_s = 0.05;             % 系统采样时间 s (20Hz)
params.system.dt = params.system.sample_time_s;

% 通信和接口参数
params.system.timeout_s = 1.0;                  % 通信超时时间 s
params.system.retry_count = 3;                  % 重试次数

%% 安全参数
params.safety = struct();

% 速度安全限制
params.safety.emergency_brake_accel_ms2 = -6.0; % 紧急制动加速度 m/s²
params.safety.max_safe_speed_kmh = 130.0;       % 最大安全速度 km/h
params.safety.min_safe_distance_m = 5.0;        % 最小安全距离 m

% 扭矩仲裁参数
params.safety.arbitration_gain_factor = 0.8;    % 仲裁模式增益因子
params.safety.manual_override_threshold = 0.5;  % 手动接管阈值

%% 调试和监控参数
params.debug = struct();

% 调试输出控制
params.debug.enable_console_output = true;      % 启用控制台输出
params.debug.output_decimation = 20;            % 输出抽取倍数
params.debug.enable_state_logging = true;       % 启用状态记录

% 性能监控
params.debug.execution_time_warning_ms = 10;    % 执行时间警告阈值 ms
params.debug.memory_usage_warning_mb = 50;      % 内存使用警告阈值 MB

%% 仿真参数
params.simulation = struct();

% 仿真时间设置
params.simulation.start_time_s = 0.0;           % 仿真开始时间 s
params.simulation.stop_time_s = inf;            % 仿真结束时间 s
params.simulation.max_step_size_s = 0.01;       % 最大步长 s

% 求解器设置
params.simulation.solver_type = 'FixedStepDiscrete'; % 求解器类型
params.simulation.relative_tolerance = 1e-3;    % 相对误差容限
params.simulation.absolute_tolerance = 1e-6;    % 绝对误差容限

%% 参数验证
fprintf('🔍 验证参数合理性...\n');

% 验证速度参数
assert(params.decision.V_min_kmh < params.decision.V_max_kmh, ...
    '最小速度不能大于等于最大速度');
assert(params.decision.V_target_default_kmh >= params.decision.V_min_kmh && ...
       params.decision.V_target_default_kmh <= params.decision.V_max_kmh, ...
    '默认目标速度超出合理范围');

% 验证时距参数
assert(params.decision.G2_min_s > 0 && params.decision.G2_max_s < 5, ...
    '时距参数范围不合理');

% 验证SPPVT控制参数
assert(params.sppvt.kp > 0 && params.sppvt.kp < 5, ...
    'SPPVT比例增益超出合理范围');
assert(params.sppvt.max_decel_ms2 < 0 && params.sppvt.max_accel_ms2 > 0, ...
    'SPPVT加速度限制设置错误');

% 验证采样时间
assert(params.system.sample_time_s > 0.01 && params.system.sample_time_s < 0.1, ...
    '采样时间设置不合理');

fprintf('✅ 参数验证通过\n');

%% 显示参数摘要
fprintf('\n📊 参数配置摘要:\n');
fprintf('================================\n');
fprintf('🎯 决策参数:\n');
fprintf('   目标速度: %.1f km/h (%.1f - %.1f km/h)\n', ...
    params.decision.V_target_default_kmh, ...
    params.decision.V_min_kmh, params.decision.V_max_kmh);
fprintf('   默认时距: %.1f s (%.1f - %.1f s)\n', ...
    params.decision.G2_default_s, ...
    params.decision.G2_min_s, params.decision.G2_max_s);

fprintf('\n🎮 SPPVT参数:\n');
fprintf('   PID增益: Kp=%.2f, Ki=%.2f, Kd=%.2f\n', ...
    params.sppvt.kp, params.sppvt.ki, params.sppvt.kd);
fprintf('   加速度限制: %.1f ~ %.1f m/s²\n', ...
    params.sppvt.max_decel_ms2, params.sppvt.max_accel_ms2);

fprintf('\n⚙️  系统参数:\n');
fprintf('   采样时间: %.0f ms (%.0f Hz)\n', ...
    params.system.sample_time_s * 1000, 1/params.system.sample_time_s);
fprintf('   求解器: %s\n', params.simulation.solver_type);

fprintf('\n🛡️  安全参数:\n');
fprintf('   最大安全速度: %.0f km/h\n', params.safety.max_safe_speed_kmh);
fprintf('   最小安全距离: %.1f m\n', params.safety.min_safe_distance_m);
fprintf('   紧急制动: %.1f m/s²\n', params.safety.emergency_brake_accel_ms2);

%% 加载参数到工作区
fprintf('\n📦 加载参数到MATLAB工作区...\n');

% 加载到基础工作区
assignin('base', 'ModelParams', params);
assignin('base', 'DecisionParams', params.decision);
assignin('base', 'SPPVTParams', params.sppvt);
assignin('base', 'TwoModeParams', params.two_mode);
assignin('base', 'SystemParams', params.system);
assignin('base', 'SafetyParams', params.safety);

fprintf('✅ 参数已加载到基础工作区变量:\n');
fprintf('   - ModelParams (完整参数结构)\n');
fprintf('   - DecisionParams (决策参数)\n');
fprintf('   - SPPVTParams (SPPVT参数)\n');
fprintf('   - TwoModeParams (两模式参数)\n');
fprintf('   - SystemParams (系统参数)\n');
fprintf('   - SafetyParams (安全参数)\n');

%% 保存参数到文件
fprintf('\n💾 保存参数配置到文件...\n');

% 保存为.mat文件
try
    save('ACC_SPPVT_Parameters.mat', 'params');
    fprintf('✅ 参数已保存到: ACC_SPPVT_Parameters.mat\n');
catch ME
    fprintf('❌ 保存参数文件失败: %s\n', ME.message);
end

fprintf('\n✅ 参数配置完成!\n\n');