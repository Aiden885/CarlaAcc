%% plot_simulink_trace.m
% 用法：
%   plot_simulink_trace              % 自动找最新的 acc_results_*.csv
%   plot_simulink_trace('xxx.csv')   % 指定文件

function plot_simulink_trace(csv_path)
    if nargin < 1
        csv_path = find_latest_acc_results();
    end

    if ~exist(csv_path, 'file')
        error('文件不存在: %s', csv_path);
    end

    %% 读取数据
    T = readtable(csv_path);
    fprintf('已读取 %d 行数据: %s\n', height(T), csv_path);

    %% 提取列 
    % Step, Desired Gap (s), Actual Gap (s), Error (s),
    % Ego Speed (km/h), Target Speed (km/h),
    % Incremental Torque (Nm), Incremental Brake Torque (Nm),
    % Vehicle Distance (m)
    steps       = T{:, 1};
    desired_gap = T{:, 2};
    actual_gap  = T{:, 3};
    errors      = T{:, 4};
    ego_speed   = T{:, 5};
    target_speed = T{:, 6};
    inc_torque  = T{:, 7};
    inc_brake   = T{:, 8};
    % 兼容旧CSV（8列）和新CSV（9列）
    if width(T) >= 9
        veh_dist = T{:, 9};
    else
        veh_dist = zeros(size(steps));
    end

    %% 横轴：step → 秒 (每步 0.05s)
    time_s = steps * 0.05;

    %% MATLAB 经典配色 (与 Python 一致)
    color_desired = [0.00 0.45 0.74];  % #0072BD
    color_actual  = [0.85 0.33 0.10];  % #D95319
    color_error   = [0.93 0.69 0.13];  % #EDB120
    color_ego     = [0.49 0.18 0.56];  % #7E2F8E
    color_target  = [0.47 0.67 0.19];  % #77AC30
    color_inc_t   = [0.17 0.63 0.17];  % #2ca02c
    color_inc_b   = [0.58 0.40 0.74];  % #9467bd
    color_dist    = [0.09 0.75 0.81];  % #17becf

    text_color = [0 0 0];
    grid_color = [0.85 0.85 0.85];  % #d9d9d9

    %% 创建 figure
    fig = figure('Name', 'ACC Results', ...
                 'Position', [50, 50, 1400, 1050], 'Color', 'w');

    %% markevery 等价：每 10 个点画一个 marker
    N = length(time_s);
    mk_idx = 1:10:N;  % marker 索引

    %% 统一子图位置 [left, bottom, width, height]
    % 所有子图宽度一致，legend 放在绘图区右侧固定位置
    num_plots  = 6;
    plot_left  = 0.08;
    plot_width = 0.76;   % 留出右侧空间给 legend
    plot_h     = 0.11;   % 每个子图高度
    gap        = 0.04;   % 子图间距
    bottom_start = 0.04;

    pos = zeros(num_plots, 4);
    for k = 1:num_plots
        pos(k,:) = [plot_left, bottom_start + (num_plots-k)*(plot_h+gap), plot_width, plot_h];
    end

    %% ===== 子图1: Time Gap Tracking =====
    ax1 = axes('Position', pos(1,:));
    plot(time_s, desired_gap, '-o', 'Color', color_desired, 'LineWidth', 2.0, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx); hold on;
    plot(time_s, actual_gap, '-s', 'Color', color_actual, 'LineWidth', 2.0, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx);
    ylabel('Time Gap (s)', 'Color', text_color, 'FontSize', 11, 'FontWeight', 'bold');
    title('Time Gap Tracking', 'Color', text_color, 'FontSize', 13, 'FontWeight', 'bold');
    lg1 = legend('Desired Time Gap', 'Actual Time Gap', 'FontSize', 10);
    set(lg1, 'Position', [plot_left+plot_width+0.01, pos(1,2)+pos(1,4)*0.3, 0.13, 0.06]);
    grid on;
    set(ax1, 'GridColor', grid_color, 'GridLineStyle', '-', 'GridAlpha', 0.8, 'LineWidth', 1.2);

    %% ===== 子图2: Time Gap Error =====
    ax2 = axes('Position', pos(2,:));
    plot(time_s, errors, '-d', 'Color', color_error, 'LineWidth', 2.0, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx); hold on;
    yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, 'Alpha', 0.8);
    ylabel('Error (s)', 'Color', text_color, 'FontSize', 11, 'FontWeight', 'bold');
    title('Time Gap Error (Actual - Desired)', 'Color', text_color, 'FontSize', 13, 'FontWeight', 'bold');
    lg2 = legend('Time Gap Error', 'Zero Reference', 'FontSize', 10);
    set(lg2, 'Position', [plot_left+plot_width+0.01, pos(2,2)+pos(2,4)*0.3, 0.13, 0.06]);
    grid on;
    set(ax2, 'GridColor', grid_color, 'GridLineStyle', '-', 'GridAlpha', 0.8, 'LineWidth', 1.2);

    %% ===== 子图3: Vehicle Speeds =====
    ax3 = axes('Position', pos(3,:));
    plot(time_s, ego_speed, '-v', 'Color', color_ego, 'LineWidth', 2.0, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx); hold on;
    plot(time_s, target_speed, '-^', 'Color', color_target, 'LineWidth', 2.0, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx);
    ylabel('Speed (km/h)', 'Color', text_color, 'FontSize', 11, 'FontWeight', 'bold');
    title('Vehicle Speeds', 'Color', text_color, 'FontSize', 13, 'FontWeight', 'bold');
    lg3 = legend('Ego Speed', 'Target Speed', 'FontSize', 10);
    set(lg3, 'Position', [plot_left+plot_width+0.01, pos(3,2)+pos(3,4)*0.3, 0.13, 0.06]);
    grid on;
    set(ax3, 'GridColor', grid_color, 'GridLineStyle', '-', 'GridAlpha', 0.8, 'LineWidth', 1.2);

    %% ===== 子图4: Incremental Engine Torque =====
    ax4 = axes('Position', pos(4,:));
    plot(time_s, inc_torque, '--x', 'Color', color_inc_t, 'LineWidth', 2.0, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx); hold on;
    yline(0, '--k', 'LineWidth', 1.5, 'Alpha', 0.7);
    ylabel('Torque (Nm)', 'Color', text_color, 'FontSize', 11, 'FontWeight', 'bold');
    title('Incremental Engine Torque', 'Color', text_color, 'FontSize', 13, 'FontWeight', 'bold');
    lg4 = legend('Inc Torque', 'FontSize', 10);
    set(lg4, 'Position', [plot_left+plot_width+0.01, pos(4,2)+pos(4,4)*0.3, 0.13, 0.04]);
    grid on;
    set(ax4, 'GridColor', grid_color, 'GridLineStyle', '-', 'GridAlpha', 0.8, 'LineWidth', 1.2);

    %% ===== 子图5: Incremental Brake Torque =====
    ax5 = axes('Position', pos(5,:));
    plot(time_s, inc_brake, '--x', 'Color', color_inc_b, 'LineWidth', 2.0, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx); hold on;
    yline(0, '--k', 'LineWidth', 1.5, 'Alpha', 0.7);
    ylabel('Brake Torque (Nm)', 'Color', text_color, 'FontSize', 11, 'FontWeight', 'bold');
    title('Incremental Brake Torque', 'Color', text_color, 'FontSize', 13, 'FontWeight', 'bold');
    lg5 = legend('Inc Brake Torque', 'FontSize', 10);
    set(lg5, 'Position', [plot_left+plot_width+0.01, pos(5,2)+pos(5,4)*0.3, 0.13, 0.04]);
    grid on;
    set(ax5, 'GridColor', grid_color, 'GridLineStyle', '-', 'GridAlpha', 0.8, 'LineWidth', 1.2);

    %% ===== 子图6: Vehicle Distance =====
    ax6 = axes('Position', pos(6,:));
    plot(time_s, veh_dist, '-o', 'Color', color_dist, 'LineWidth', 2.0, ...
         'MarkerSize', 2, 'MarkerIndices', mk_idx);
    xlabel('Time (s)', 'Color', text_color, 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Distance (m)', 'Color', text_color, 'FontSize', 11, 'FontWeight', 'bold');
    title('Vehicle Distance', 'Color', text_color, 'FontSize', 13, 'FontWeight', 'bold');
    lg6 = legend('Vehicle Distance', 'FontSize', 10);
    set(lg6, 'Position', [plot_left+plot_width+0.01, pos(6,2)+pos(6,4)*0.3, 0.13, 0.04]);
    grid on;
    set(ax6, 'GridColor', grid_color, 'GridLineStyle', '-', 'GridAlpha', 0.8, 'LineWidth', 1.2);

    %% 联动 x 轴
    linkaxes([ax1 ax2 ax3 ax4 ax5 ax6], 'x');

    %% 统计输出
    fprintf('\n========== 统计 ==========\n');
    fprintf('数据点: %d, 时长: %.1f s\n', N, time_s(end));
    fprintf('期望时距: %.3f s\n', desired_gap(1));
    fprintf('实际时距: %.3f ~ %.3f s (均值 %.3f)\n', min(actual_gap), max(actual_gap), mean(actual_gap));
    fprintf('误差: %.3f ~ %.3f s (均值 %.3f)\n', min(errors), max(errors), mean(errors));
    fprintf('自车速度: %.1f ~ %.1f km/h\n', min(ego_speed), max(ego_speed));
    fprintf('前车速度: %.1f ~ %.1f km/h\n', min(target_speed), max(target_speed));
    fprintf('增量扭矩: %.1f ~ %.1f Nm\n', min(inc_torque), max(inc_torque));
    fprintf('增量制动: %.1f ~ %.1f Nm\n', min(inc_brake), max(inc_brake));
    fprintf('两车距离: %.1f ~ %.1f m (均值 %.1f)\n', min(veh_dist), max(veh_dist), mean(veh_dist));
    fprintf('============================\n');
end


function csv_path = find_latest_acc_results()
%FIND_LATEST_ACC_RESULTS 自动查找最新的 acc_results_YYYYMMDD_HHMMSS.csv
%   按文件名排序（时间戳命名，字典序即时间序）

    files = dir('acc_results_*.csv');
    if isempty(files)
        error('当前目录下没有找到 acc_results_*.csv 文件');
    end

    % 按文件名字典序排序，取最后一个（最新的时间戳）
    names = {files.name};
    [~, idx] = sort(names);
    csv_path = files(idx(end)).name;
    fprintf('自动选择: %s\n', csv_path);
end
