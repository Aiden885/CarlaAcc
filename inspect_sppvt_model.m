% 检查sppvt_control_model.slx的输入输出端口
% 这个脚本用于了解模型的接口结构

clear; clc;

model_name = 'sppvt_control_model';

% 加载模型
fprintf('正在加载模型: %s\n', model_name);
load_system(model_name);

fprintf('\n========================================\n');
fprintf('模型端口信息\n');
fprintf('========================================\n');

% 查找所有Inport块
inports = find_system(model_name, 'BlockType', 'Inport');
fprintf('\n【输入端口 Inports】共 %d 个:\n', length(inports));
for i = 1:length(inports)
    port_num = get_param(inports{i}, 'Port');
    port_name = get_param(inports{i}, 'Name');
    fprintf('  [%s] 端口%s: %s\n', inports{i}, port_num, port_name);
end

% 查找所有Outport块
outports = find_system(model_name, 'BlockType', 'Outport');
fprintf('\n【输出端口 Outports】共 %d 个:\n', length(outports));
for i = 1:length(outports)
    port_num = get_param(outports{i}, 'Port');
    port_name = get_param(outports{i}, 'Name');
    fprintf('  [%s] 端口%s: %s\n', outports{i}, port_num, port_name);
end

% 查看模型配置参数
fprintf('\n========================================\n');
fprintf('模型配置信息\n');
fprintf('========================================\n');

% 检查外部输入配置
try
    load_ext_input = get_param(model_name, 'LoadExternalInput');
    fprintf('LoadExternalInput: %s\n', load_ext_input);

    if strcmp(load_ext_input, 'on')
        ext_input = get_param(model_name, 'ExternalInput');
        fprintf('ExternalInput: %s\n', ext_input);
    end
catch
    fprintf('无外部输入配置或配置已禁用\n');
end

% 检查求解器配置
solver_type = get_param(model_name, 'SolverType');
fprintf('SolverType: %s\n', solver_type);

stop_time = get_param(model_name, 'StopTime');
fprintf('StopTime: %s\n', stop_time);

% 保存端口信息到文件
fprintf('\n正在保存端口信息到文件...\n');
fid = fopen('sppvt_model_ports_info.txt', 'w');

fprintf(fid, '==============================================\n');
fprintf(fid, 'SPPVT Control Model 端口信息\n');
fprintf(fid, '==============================================\n\n');

fprintf(fid, '【输入端口 Inports】共 %d 个:\n', length(inports));
for i = 1:length(inports)
    port_num = get_param(inports{i}, 'Port');
    port_name = get_param(inports{i}, 'Name');
    fprintf(fid, '  端口 %s: %s (路径: %s)\n', port_num, port_name, inports{i});
end

fprintf(fid, '\n【输出端口 Outports】共 %d 个:\n', length(outports));
for i = 1:length(outports)
    port_num = get_param(outports{i}, 'Port');
    port_name = get_param(outports{i}, 'Name');
    fprintf(fid, '  端口 %s: %s (路径: %s)\n', port_num, port_name, outports{i});
end

fprintf(fid, '\n【模型配置】\n');
fprintf(fid, 'SolverType: %s\n', solver_type);
fprintf(fid, 'StopTime: %s\n', stop_time);

fclose(fid);

fprintf('✅ 端口信息已保存到 sppvt_model_ports_info.txt\n');

% 关闭模型（不保存）
close_system(model_name, 0);

fprintf('\n完成！\n');