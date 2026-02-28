%% plot_simulink_trace.m
% 从 Simulink To File 保存的 .mat 文件读取数据并绘图
%
% acc_simulink_log.mat 格式 (acc_log 矩阵, 19×N):
%   行1  : 仿真时间 (s)
%   行2  : command_type
%   行3  : ego_speed_ms
%   行4  : vehicle_distance
%   行5  : target_speed_ms
%   行6  : current_engine_torque (Nm)
%   行7  : G2_s (desired time gap, s)
%   行8  : control_error_signed (s)
%   行9  : reset_flag
%   行10 : Y0 (Nm)
%   行11 : final_output (Nm)
%   行12 : control_enabled
%   行13 : current_state
%   行14 : decision_out
%   行15 : stage_offset
%   行16 : stage
%   行17 : error_sign
%   行18 : cooldown
%   行19 : sign_changed
%
% 用法:
%   plot_simulink_trace              % 自动找最新 acc_simulink_log*.mat
%   plot_simulink_trace('xxx.mat')   % 指定文件

function plot_simulink_trace(mat_path)
    if nargin < 1
        mat_path = find_latest_mat();
    end

    if ~exist(mat_path, 'file')
        error('文件不存在: %s', mat_path);
    end

    %% 加载数据
    raw = load(mat_path);
    if ~isfield(raw, 'acc_log')
        error('mat 文件中找不到变量 acc_log，请确认由 add_simulink_logging 生成');
    end

    %% 解析 timeseries 格式（Simulink 2024b To File 默认格式）
    %  acc_log.Time : N×1 时间向量
    %  acc_log.Data : N×18 信号矩阵，列顺序 = Mux 输入顺序
    ts = raw.acc_log;
    if ~isa(ts, 'timeseries')
        error('acc_log 不是 timeseries 类型（实际类型: %s），请检查 To File 保存格式', class(ts));
    end

    time_raw = ts.Time;      % N×1
    data     = ts.Data;      % N×18

    N = length(time_raw);
    fprintf('已读取数据: %d 帧 × %d 信号 (%s)\n', N, size(data,2), mat_path);

    %% 维度校验
    if size(data,2) < 18
        fprintf('\n⚠️  信号列数不足（%d < 18）\n', size(data,2));
        fprintf('   请确认 Logging 子系统的 18 个输入端口已全部连线后重新仿真\n');
        error('信号列数不足，无法绘图');
    end
    if N < 2
        error('数据帧数不足（%d 帧），请确保仿真正常运行后再画图', N);
    end

    %% 时间归零
    time_s = time_raw - time_raw(1);

    fprintf('有效时长: %.1f s\n', time_s(end));

    %% 按列解析信号（列序 = Mux 端口序 = port_labels 顺序）
    % 列1  command_type
    % 列2  ego_speed_ms
    % 列3  vehicle_distance
    % 列4  target_speed_ms
    % 列5  current_engine_torque
    % 列6  G2_s
    % 列7  control_error_signed
    % 列8  reset_flag
    % 列9  Y0
    % 列10 final_output
    % 列11 control_enabled
    % 列12 current_state
    % 列13 decision_out
    % 列14 stage_offset
    % 列15 stage
    % 列16 error_sign
    % 列17 cooldown
    % 列18 sign_changed
    ego_speed_ms          = data(:,2);
    vehicle_distance      = data(:,3);
    target_speed_ms       = data(:,4);
    current_engine_torque = data(:,5);
    G2_s                  = data(:,6);
    control_error_signed  = data(:,7);
    reset_flag            = data(:,8);
    Y0                    = data(:,9);
    final_output          = data(:,10);
    control_enabled       = data(:,11);
    current_state         = data(:,12);
    decision_out          = data(:,13);
    stage_offset          = data(:,14);
    stage                 = data(:,15);
    sign_changed          = data(:,18);

    %% 派生量
    ego_speed_kmh    = ego_speed_ms    * 3.6;
    target_speed_kmh = target_speed_ms * 3.6;
    desired_gap      = G2_s;
    actual_gap       = control_error_signed + G2_s;
    errors           = control_error_signed;
    % 分离加速/制动扭矩
    drive_torque     = max(final_output, 0);
    brake_torque     = max(-final_output, 0);

    N      = length(time_s);
    mk_idx = 1:10:N;

    %% 配色
    c_desired  = [0.00 0.45 0.74];
    c_actual   = [0.85 0.33 0.10];
    c_error    = [0.93 0.69 0.13];
    c_ego      = [0.49 0.18 0.56];
    c_target   = [0.47 0.67 0.19];
    c_drive    = [0.17 0.63 0.17];
    c_brake    = [0.58 0.40 0.74];
    c_dist     = [0.09 0.75 0.81];
    c_Y0       = [0.85 0.33 0.10];
    c_soffset  = [0.00 0.45 0.74];
    c_ce       = [0.30 0.75 0.93];
    c_reset    = [0.93 0.30 0.30];
    c_text     = [0 0 0];
    c_grid     = [0.85 0.85 0.85];

    %% 布局：9个子图
    num_plots   = 9;
    plot_left   = 0.08;
    plot_width  = 0.74;
    plot_h      = 0.083;
    gap         = 0.018;
    bottom_start = 0.03;

    pos = zeros(num_plots, 4);
    for k = 1:num_plots
        pos(k,:) = [plot_left, bottom_start + (num_plots-k)*(plot_h+gap), plot_width, plot_h];
    end

    figure('Name', 'ACC Simulink Results', ...
           'Position', [50, 30, 1440, 1100], 'Color', 'w');

    %% ===== 子图1: 时距跟踪 =====
    ax1 = axes('Position', pos(1,:));
    plot(time_s, desired_gap, '-o', 'Color', c_desired, 'LineWidth', 1.8, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx); hold on;
    plot(time_s, actual_gap,  '-s', 'Color', c_actual,  'LineWidth', 1.8, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx);
    ylabel('Time Gap (s)', 'FontSize', 10, 'FontWeight', 'bold');
    title('Time Gap Tracking', 'FontSize', 12, 'FontWeight', 'bold');
    lg = legend('Desired Gap', 'Actual Gap', 'FontSize', 9);
    set(lg, 'Position', [plot_left+plot_width+0.01, pos(1,2)+pos(1,4)*0.2, 0.14, 0.055]);
    grid on; set(ax1, 'GridColor', c_grid, 'GridAlpha', 0.8);

    %% ===== 子图2: 误差 =====
    ax2 = axes('Position', pos(2,:));
    plot(time_s, errors, '-d', 'Color', c_error, 'LineWidth', 1.8, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx); hold on;
    yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
    % 标记变号事件
    sc_idx = find(sign_changed > 0.5);
    if ~isempty(sc_idx)
        scatter(time_s(sc_idx), errors(sc_idx), 25, 'r', 'filled', 'DisplayName', 'Sign Changed');
    end
    ylabel('Error (s)', 'FontSize', 10, 'FontWeight', 'bold');
    title('Time Gap Error', 'FontSize', 12, 'FontWeight', 'bold');
    lg = legend('Error', 'Zero', 'FontSize', 9);
    set(lg, 'Position', [plot_left+plot_width+0.01, pos(2,2)+pos(2,4)*0.2, 0.14, 0.055]);
    grid on; set(ax2, 'GridColor', c_grid, 'GridAlpha', 0.8);

    %% ===== 子图3: 车速 =====
    ax3 = axes('Position', pos(3,:));
    plot(time_s, ego_speed_kmh,    '-v', 'Color', c_ego,    'LineWidth', 1.8, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx); hold on;
    plot(time_s, target_speed_kmh, '-^', 'Color', c_target, 'LineWidth', 1.8, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx);
    ylabel('Speed (km/h)', 'FontSize', 10, 'FontWeight', 'bold');
    title('Vehicle Speeds', 'FontSize', 12, 'FontWeight', 'bold');
    lg = legend('Ego', 'Target', 'FontSize', 9);
    set(lg, 'Position', [plot_left+plot_width+0.01, pos(3,2)+pos(3,4)*0.2, 0.14, 0.055]);
    grid on; set(ax3, 'GridColor', c_grid, 'GridAlpha', 0.8);

    %% ===== 子图4: 车距 =====
    ax4 = axes('Position', pos(4,:));
    plot(time_s, vehicle_distance, '-o', 'Color', c_dist, 'LineWidth', 1.8, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx);
    ylabel('Distance (m)', 'FontSize', 10, 'FontWeight', 'bold');
    title('Vehicle Distance', 'FontSize', 12, 'FontWeight', 'bold');
    lg = legend('Distance', 'FontSize', 9);
    set(lg, 'Position', [plot_left+plot_width+0.01, pos(4,2)+pos(4,4)*0.2, 0.14, 0.04]);
    grid on; set(ax4, 'GridColor', c_grid, 'GridAlpha', 0.8);

    %% ===== 子图5: 输出扭矩（驱动/制动分离）=====
    ax5 = axes('Position', pos(5,:));
    plot(time_s, drive_torque, '-',  'Color', c_drive, 'LineWidth', 1.8); hold on;
    plot(time_s, -brake_torque, '-', 'Color', c_brake, 'LineWidth', 1.8);
    yline(0, '--k', 'LineWidth', 1.0, 'Alpha', 0.6);
    ylabel('Torque (Nm)', 'FontSize', 10, 'FontWeight', 'bold');
    title('Final Output Torque (Drive / Brake)', 'FontSize', 12, 'FontWeight', 'bold');
    lg = legend('Drive', 'Brake (neg)', 'FontSize', 9);
    set(lg, 'Position', [plot_left+plot_width+0.01, pos(5,2)+pos(5,4)*0.2, 0.14, 0.055]);
    grid on; set(ax5, 'GridColor', c_grid, 'GridAlpha', 0.8);

    %% ===== 子图6: Y0 与 current_engine_torque =====
    ax6 = axes('Position', pos(6,:));
    plot(time_s, Y0,                    '-',  'Color', c_Y0,    'LineWidth', 1.8); hold on;
    plot(time_s, current_engine_torque, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
    ylabel('Torque (Nm)', 'FontSize', 10, 'FontWeight', 'bold');
    title('Y0 vs Current Engine Torque', 'FontSize', 12, 'FontWeight', 'bold');
    lg = legend('Y0', 'Eng Torque', 'FontSize', 9);
    set(lg, 'Position', [plot_left+plot_width+0.01, pos(6,2)+pos(6,4)*0.2, 0.14, 0.055]);
    grid on; set(ax6, 'GridColor', c_grid, 'GridAlpha', 0.8);

    %% ===== 子图7: stage_offset 与 stage =====
    ax7 = axes('Position', pos(7,:));
    yyaxis left;
    plot(time_s, stage_offset, '-', 'Color', c_soffset, 'LineWidth', 1.8);
    ylabel('Stage Offset', 'FontSize', 10, 'FontWeight', 'bold');
    yyaxis right;
    stairs(time_s, stage, '-', 'Color', [0.93 0.69 0.13], 'LineWidth', 1.5);
    ylabel('Stage', 'FontSize', 10, 'FontWeight', 'bold');
    title('SPPVT Stage Offset & Stage', 'FontSize', 12, 'FontWeight', 'bold');
    lg = legend('stage\_offset', 'stage', 'FontSize', 9);
    set(lg, 'Position', [plot_left+plot_width+0.01, pos(7,2)+pos(7,4)*0.2, 0.14, 0.055]);
    grid on; set(ax7, 'GridColor', c_grid, 'GridAlpha', 0.8);

    %% ===== 子图8: control_enabled & reset_flag =====
    ax8 = axes('Position', pos(8,:));
    stairs(time_s, control_enabled, '-', 'Color', c_ce,    'LineWidth', 1.8); hold on;
    stairs(time_s, reset_flag,      '-', 'Color', c_reset, 'LineWidth', 1.5);
    ylim([-0.2, 1.5]);
    yticks([0 1]); yticklabels({'OFF','ON'});
    ylabel('Flag', 'FontSize', 10, 'FontWeight', 'bold');
    title('Control Enabled & Reset Flag', 'FontSize', 12, 'FontWeight', 'bold');
    lg = legend('ctrl\_enabled', 'reset\_flag', 'FontSize', 9);
    set(lg, 'Position', [plot_left+plot_width+0.01, pos(8,2)+pos(8,4)*0.2, 0.14, 0.055]);
    grid on; set(ax8, 'GridColor', c_grid, 'GridAlpha', 0.8);

    %% ===== 子图9: 决策状态 =====
    ax9 = axes('Position', pos(9,:));
    stairs(time_s, current_state, '-',  'Color', [0.49 0.18 0.56], 'LineWidth', 1.8); hold on;
    stairs(time_s, decision_out,  '--', 'Color', [0.17 0.63 0.17], 'LineWidth', 1.5);
    xlabel('Time (s)', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel('State / Decision', 'FontSize', 10, 'FontWeight', 'bold');
    title('State Machine: Current State & Decision', 'FontSize', 12, 'FontWeight', 'bold');
    lg = legend('current\_state', 'decision', 'FontSize', 9);
    set(lg, 'Position', [plot_left+plot_width+0.01, pos(9,2)+pos(9,4)*0.2, 0.14, 0.055]);
    grid on; set(ax9, 'GridColor', c_grid, 'GridAlpha', 0.8);

    %% 联动 x 轴
    linkaxes([ax1 ax2 ax3 ax4 ax5 ax6 ax7 ax8 ax9], 'x');

    %% 统计输出
    ctrl_on = control_enabled > 0.5;
    fprintf('\n========== 统计 ==========\n');
    fprintf('总帧数: %d，时长: %.1f s\n', N, time_s(end));
    fprintf('控制激活帧数: %d (%.1f%%)\n', sum(ctrl_on), 100*mean(ctrl_on));
    fprintf('期望时距: %.3f ~ %.3f s\n', min(desired_gap), max(desired_gap));
    if any(ctrl_on)
        fprintf('实际时距(激活段): %.3f ~ %.3f s (均值 %.3f)\n', ...
            min(actual_gap(ctrl_on)), max(actual_gap(ctrl_on)), mean(actual_gap(ctrl_on)));
        fprintf('误差(激活段): %.3f ~ %.3f s (均值 %.4f)\n', ...
            min(errors(ctrl_on)), max(errors(ctrl_on)), mean(errors(ctrl_on)));
    end
    fprintf('自车速度: %.1f ~ %.1f km/h\n', min(ego_speed_kmh), max(ego_speed_kmh));
    fprintf('前车速度: %.1f ~ %.1f km/h\n', min(target_speed_kmh), max(target_speed_kmh));
    fprintf('车间距离: %.1f ~ %.1f m\n', min(vehicle_distance), max(vehicle_distance));
    fprintf('输出扭矩: %.1f ~ %.1f Nm\n', min(final_output), max(final_output));
    fprintf('Y0 范围:  %.1f ~ %.1f Nm\n', min(Y0), max(Y0));
    fprintf('stage_offset 范围: %.3f ~ %.3f\n', min(stage_offset), max(stage_offset));
    n_resets = sum(diff(reset_flag) > 0.5);
    n_sc     = sum(diff(sign_changed) > 0.5);
    fprintf('Reset 触发次数: %d，变号次数: %d\n', n_resets, n_sc);
    fprintf('============================\n');
end


%% ============================================================
function mat_path = find_latest_mat()
    files = dir('acc_simulink_log*.mat');
    if isempty(files)
        error('当前目录下没有找到 acc_simulink_log*.mat 文件，请先运行仿真');
    end
    [~, idx] = max([files.datenum]);
    mat_path = files(idx).name;
    fprintf('自动选择: %s\n', mat_path);
end
