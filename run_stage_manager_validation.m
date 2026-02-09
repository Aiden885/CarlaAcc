function results = run_stage_manager_validation()
%RUN_STAGE_MANAGER_VALIDATION Comprehensive test for StageManager.
% - Creates stage_manager_test.slx if needed
% - Builds inputs, runs simulation, and validates expected behaviors.
%
% StageManager outputs are IMMEDIATE (not delayed by Unit Delay).
% The Unit Delay only affects internal state feedback, not the output ports.

model = 'stage_manager_test';

%% ========== 1. 创建或加载测试模型 ==========
if ~exist([model '.slx'], 'file')
    fprintf('Creating test model: %s.slx\n', model);
    new_system(model);
    save_system(model);
end

if ~bdIsLoaded(model)
    load_system(model);
end

% Ensure StageManager exists, if not create it
if isempty(find_system(model, 'SearchDepth', 1, 'Name', 'StageManager'))
    fprintf('StageManager not found, creating it...\n');
    create_stage_manager(model);
end

%% ========== 2. 清理旧的测试线束 ==========
stagePath = [model '/StageManager'];
ph = get_param(stagePath, 'PortHandles');
ports = [ph.Inport ph.Outport];
for i = 1:numel(ports)
    lh = get_param(ports(i), 'Line');
    if lh ~= -1
        delete_line(lh);
    end
end

blocksToDelete = {
    [model '/SM_error'], [model '/SM_upgrade'], [model '/SM_reset'], [model '/SM_enable'], ...
    [model '/SM_stage_offset'], [model '/SM_stage'], [model '/SM_error_sign'], [model '/SM_cooldown']
};
for i = 1:numel(blocksToDelete)
    if getSimulinkBlockHandle(blocksToDelete{i}) ~= -1
        delete_block(blocksToDelete{i});
    end
end

%% ========== 3. 仿真参数 ==========
dt = 0.05;
stopTime = 2.0;
set_param(model, 'Solver', 'FixedStepDiscrete', 'FixedStep', num2str(dt), 'StopTime', num2str(stopTime));

%% ========== 4. 设计测试输入 ==========
% 时间轴
t = (0:dt:stopTime)';
N = numel(t);

% --- error 信号 ---
% 0.00~0.05: 0 (reset期间)
% 0.05~0.45: +0.5 (正误差)
% 0.45~0.55: 1e-7 (死区内，sign=0)
% 0.55~0.85: -0.5 (负误差，符号翻转)
% 0.85~1.20: +0.4 (正误差，符号翻转)
% 1.20~1.95: +1200 (大正误差，测试限幅)
% 1.95~2.00: 0
err = zeros(N, 1);
err(t >= 0.05 & t < 0.45) = 0.5;
err(t >= 0.45 & t < 0.55) = 1e-7;  % 死区
err(t >= 0.55 & t < 0.85) = -0.5;
err(t >= 0.85 & t < 1.20) = 0.4;
err(t >= 1.20 & t < 1.95) = 1200;
err(t >= 1.95) = 0.0;

% --- should_upgrade 脉冲 ---
upgrade = zeros(N, 1);
pulse_times = [0.15, 0.25, 0.35, 0.65, 0.75, 0.90, 1.30, 1.45, 1.60, 1.75, 1.90];
for pt = pulse_times
    idx_pt = find(abs(t - pt) < 1e-9, 1);
    if ~isempty(idx_pt)
        upgrade(idx_pt) = 1;
    end
end

% --- reset_flag ---
% t=0~0.05: 1 (初始reset)
% t=1.20: 1 (中途reset)
reset_flag = zeros(N, 1);
reset_flag(t <= 0.05) = 1;
idx_reset = find(abs(t - 1.20) < 1e-9, 1);
if ~isempty(idx_reset)
    reset_flag(idx_reset) = 1;
end

% --- control_enabled ---
% t=1.00~1.10: 0 (禁用)
control_enabled = ones(N, 1);
control_enabled(t >= 1.00 & t <= 1.10) = 0;

%% ========== 5. 创建 timeseries 并导入工作区 ==========
assignin('base', 'sm_error_ts', timeseries(err, t));
assignin('base', 'sm_upgrade_ts', timeseries(upgrade, t));
assignin('base', 'sm_reset_ts', timeseries(reset_flag, t));
assignin('base', 'sm_enable_ts', timeseries(control_enabled, t));

%% ========== 6. 添加测试线束块 ==========
% 输入源
add_block('simulink/Sources/From Workspace', [model '/SM_error'], ...
    'Position', [50, 50, 120, 80], 'VariableName', 'sm_error_ts');
add_block('simulink/Sources/From Workspace', [model '/SM_upgrade'], ...
    'Position', [50, 120, 120, 150], 'VariableName', 'sm_upgrade_ts');
add_block('simulink/Sources/From Workspace', [model '/SM_reset'], ...
    'Position', [50, 190, 120, 220], 'VariableName', 'sm_reset_ts');
add_block('simulink/Sources/From Workspace', [model '/SM_enable'], ...
    'Position', [50, 260, 120, 290], 'VariableName', 'sm_enable_ts');

% 输出接收
add_block('simulink/Sinks/To Workspace', [model '/SM_stage_offset'], ...
    'Position', [600, 50, 680, 80], 'VariableName', 'sm_stage_offset', 'SaveFormat', 'StructureWithTime');
add_block('simulink/Sinks/To Workspace', [model '/SM_stage'], ...
    'Position', [600, 120, 680, 150], 'VariableName', 'sm_stage', 'SaveFormat', 'StructureWithTime');
add_block('simulink/Sinks/To Workspace', [model '/SM_error_sign'], ...
    'Position', [600, 190, 680, 220], 'VariableName', 'sm_error_sign', 'SaveFormat', 'StructureWithTime');
add_block('simulink/Sinks/To Workspace', [model '/SM_cooldown'], ...
    'Position', [600, 260, 680, 290], 'VariableName', 'sm_cooldown', 'SaveFormat', 'StructureWithTime');

% 连接输入
add_line(model, 'SM_error/1',   'StageManager/1', 'autorouting', 'on');
add_line(model, 'SM_upgrade/1', 'StageManager/2', 'autorouting', 'on');
add_line(model, 'SM_reset/1',   'StageManager/3', 'autorouting', 'on');
add_line(model, 'SM_enable/1',  'StageManager/4', 'autorouting', 'on');

% 连接输出
add_line(model, 'StageManager/1', 'SM_stage_offset/1', 'autorouting', 'on');
add_line(model, 'StageManager/2', 'SM_stage/1', 'autorouting', 'on');
add_line(model, 'StageManager/3', 'SM_error_sign/1', 'autorouting', 'on');
add_line(model, 'StageManager/4', 'SM_cooldown/1', 'autorouting', 'on');

save_system(model);

%% ========== 7. 运行仿真 ==========
fprintf('\nRunning simulation...\n');
simOut = sim(model);

%% ========== 8. 提取输出数据 ==========
if evalin('base', 'exist(''sm_stage_offset'',''var'')')
    s_offset = evalin('base', 'sm_stage_offset');
    s_stage = evalin('base', 'sm_stage');
    s_sign = evalin('base', 'sm_error_sign');
    s_cool = evalin('base', 'sm_cooldown');
else
    s_offset = simOut.get('sm_stage_offset');
    s_stage = simOut.get('sm_stage');
    s_sign = simOut.get('sm_error_sign');
    s_cool = simOut.get('sm_cooldown');
end

t_out = s_offset.time;
stage_offset = s_offset.signals.values;
stage = s_stage.signals.values;
error_sign = s_sign.signals.values;
cooldown = s_cool.signals.values;

%% ========== 9. 辅助函数 ==========
idx = @(time) find(abs(t_out - time) < dt/2, 1, 'first');
tol = 1e-4;

%% ========== 10. 手动模拟期望值 ==========
% StageManager 参数
rho = 0.1;
cooldown_init = 2;
offset_max = 500;
offset_min = -500;
eps_dz = 1e-6;

% 模拟状态
sim_offset = 0;
sim_stage = 1;
sim_sign = 0;
sim_cool = 0;

expected = struct();
expected.t = t;
expected.offset = zeros(N, 1);
expected.stage = ones(N, 1);
expected.sign = zeros(N, 1);
expected.cool = zeros(N, 1);

for k = 1:N
    % 当前输入
    e = err(k);
    up = upgrade(k);
    rst = reset_flag(k);
    en = control_enabled(k);

    % 计算 current_sign (带死区)
    if abs(e) <= eps_dz
        curr_sign = 0;
    else
        curr_sign = sign(e);
    end

    % 符号翻转检测
    sign_changed = (sim_sign ~= 0) && (curr_sign ~= 0) && (sim_sign ~= curr_sign);

    % Reset 条件
    do_reset = (rst == 1) || (en == 0);

    % 清零条件
    do_clear = do_reset || sign_changed;

    if do_clear
        sim_offset = 0;
        sim_stage = 1;
        sim_cool = 0;
        sim_sign = 0;
    else
        % 升级条件
        can_upgrade = (up == 1) && (sim_cool == 0);

        if can_upgrade
            sim_stage = sim_stage + 1;
            sim_offset = sim_offset + rho * e;
            sim_offset = max(offset_min, min(offset_max, sim_offset));
            sim_cool = cooldown_init;
        else
            % cooldown 递减
            if sim_cool > 0
                sim_cool = sim_cool - 1;
            end
        end

        % 更新 error_sign
        if curr_sign ~= 0
            sim_sign = curr_sign;
        end
    end

    % 记录期望值
    expected.offset(k) = sim_offset;
    expected.stage(k) = sim_stage;
    expected.sign(k) = sim_sign;
    expected.cool(k) = sim_cool;
end

%% ========== 11. 验证关键时刻 ==========
key_times = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, ...
             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, ...
             0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, ...
             1.30, 1.45, 1.60, 1.75, 1.90];

checks = {};

for kt = key_times
    k_in = find(abs(t - kt) < 1e-9, 1);
    k_out = idx(kt);

    if isempty(k_in) || isempty(k_out)
        continue;
    end

    exp_off = expected.offset(k_in);
    exp_stg = expected.stage(k_in);
    exp_sgn = expected.sign(k_in);
    exp_coo = expected.cool(k_in);

    act_off = stage_offset(k_out);
    act_stg = stage(k_out);
    act_sgn = error_sign(k_out);
    act_coo = cooldown(k_out);

    ok_off = abs(act_off - exp_off) < tol;
    ok_stg = act_stg == exp_stg;
    ok_sgn = act_sgn == exp_sgn;
    ok_coo = act_coo == exp_coo;

    ok_all = ok_off && ok_stg && ok_sgn && ok_coo;

    desc = sprintf('t=%.2f: stg=%d off=%.2f sgn=%d cool=%d', kt, exp_stg, exp_off, exp_sgn, exp_coo);
    checks(end+1, :) = {desc, ok_all, struct('exp_off', exp_off, 'act_off', act_off, ...
        'exp_stg', exp_stg, 'act_stg', act_stg, 'exp_sgn', exp_sgn, 'act_sgn', act_sgn, ...
        'exp_coo', exp_coo, 'act_coo', act_coo)};
end

%% ========== 12. 特定行为验证 ==========
% 验证符号翻转重置 (t=0.55)
k55_in = find(abs(t - 0.55) < 1e-9, 1);
k55_out = idx(0.55);
if ~isempty(k55_in) && ~isempty(k55_out)
    % 在 t=0.55 从死区进入负区，应该不算翻转（因为前一个 sign=0 不触发）
    % 但 t=0.50 时 error=1e-7，curr_sign=0，sim_sign 保持之前的 +1
    % t=0.55 时 error=-0.5，curr_sign=-1，prev_sign=+1，发生翻转
    ok_flip = (stage(k55_out) == 1) && (abs(stage_offset(k55_out)) < tol);
    checks(end+1, :) = {'Sign flip at t=0.55 resets state', ok_flip, []};
end

% 验证 control_enabled=0 重置 (t=1.00)
k100_out = idx(1.00);
if ~isempty(k100_out)
    ok_disable = (stage(k100_out) == 1) && (abs(stage_offset(k100_out)) < tol);
    checks(end+1, :) = {'control_enabled=0 at t=1.00 resets state', ok_disable, []};
end

% 验证 reset_flag 重置 (t=1.20)
k120_out = idx(1.20);
if ~isempty(k120_out)
    ok_rst = (stage(k120_out) == 1) && (abs(stage_offset(k120_out)) < tol);
    checks(end+1, :) = {'reset_flag=1 at t=1.20 resets state', ok_rst, []};
end

% 验证限幅 (t=1.90)
k190_out = idx(1.90);
if ~isempty(k190_out)
    ok_sat = abs(stage_offset(k190_out) - 500) < tol;
    checks(end+1, :) = {'Saturation at t=1.90: offset=500', ok_sat, []};
end

%% ========== 13. 汇总结果 ==========
results = struct();
results.passed = 0;
results.failed = 0;
results.details = checks;

fprintf('\n========================================\n');
fprintf('StageManager Validation Results\n');
fprintf('========================================\n');

for i = 1:size(checks, 1)
    name = checks{i, 1};
    ok = checks{i, 2};
    detail = checks{i, 3};

    if ok
        fprintf('[PASS] %s\n', name);
        results.passed = results.passed + 1;
    else
        fprintf('[FAIL] %s\n', name);
        if ~isempty(detail)
            fprintf('       Expected: stg=%d off=%.4f sgn=%d cool=%d\n', ...
                detail.exp_stg, detail.exp_off, detail.exp_sgn, detail.exp_coo);
            fprintf('       Actual:   stg=%d off=%.4f sgn=%d cool=%d\n', ...
                detail.act_stg, detail.act_off, detail.act_sgn, detail.act_coo);
        end
        results.failed = results.failed + 1;
    end
end

fprintf('========================================\n');
fprintf('Summary: %d PASSED, %d FAILED\n', results.passed, results.failed);
fprintf('========================================\n');

%% ========== 14. Debug 快照 ==========
fprintf('\nDebug Snapshot (selected times):\n');
fprintf('  time | stg | offset   | cool | sign | err     | up | rst | en\n');
fprintf('  -----|-----|----------|------|------|---------|----|----|----\n');

debug_times = [0.05, 0.15, 0.20, 0.35, 0.55, 0.65, 0.85, 0.90, 1.00, 1.20, 1.30, 1.90];
for dt_val = debug_times
    k_in = find(abs(t - dt_val) < 1e-9, 1);
    k_out = idx(dt_val);
    if isempty(k_in) || isempty(k_out)
        continue;
    end
    fprintf('  %4.2f | %3.0f | %8.3f | %4.0f | %4.0f | %7.1f | %2.0f | %2.0f | %2.0f\n', ...
        dt_val, stage(k_out), stage_offset(k_out), cooldown(k_out), error_sign(k_out), ...
        err(k_in), upgrade(k_in), reset_flag(k_in), control_enabled(k_in));
end

%% ========== 15. 保存结果 ==========
assignin('base', 'sm_validation_results', results);
assignin('base', 'sm_expected', expected);
assignin('base', 'sm_t', t);
assignin('base', 'sm_err', err);
assignin('base', 'sm_upgrade', upgrade);
assignin('base', 'sm_reset_flag', reset_flag);
assignin('base', 'sm_control_enabled', control_enabled);

%% ========== 16. 绘图比较 ==========
if results.failed > 0
    figure('Name', 'StageManager Validation Debug', 'NumberTitle', 'off');

    subplot(4, 1, 1);
    plot(t_out, stage_offset, 'b-', 'LineWidth', 1.5); hold on;
    plot(t, expected.offset, 'r--', 'LineWidth', 1);
    legend('Actual', 'Expected');
    ylabel('stage\_offset');
    title('StageManager Output Comparison');
    grid on;

    subplot(4, 1, 2);
    plot(t_out, stage, 'b-', 'LineWidth', 1.5); hold on;
    plot(t, expected.stage, 'r--', 'LineWidth', 1);
    legend('Actual', 'Expected');
    ylabel('stage');
    grid on;

    subplot(4, 1, 3);
    plot(t_out, cooldown, 'b-', 'LineWidth', 1.5); hold on;
    plot(t, expected.cool, 'r--', 'LineWidth', 1);
    legend('Actual', 'Expected');
    ylabel('cooldown');
    grid on;

    subplot(4, 1, 4);
    plot(t_out, error_sign, 'b-', 'LineWidth', 1.5); hold on;
    plot(t, expected.sign, 'r--', 'LineWidth', 1);
    legend('Actual', 'Expected');
    ylabel('error\_sign');
    xlabel('Time (s)');
    grid on;
end

fprintf('\nValidation complete. Results saved to workspace.\n');
end
