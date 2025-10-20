function change_model_frequency(target_frequency_hz)
% CHANGE_MODEL_FREQUENCY 修改ACC_Decision_SPPVT_Integrated模型的采样频率
%
% 用法:
%   change_model_frequency(50)   % 设置为50Hz (0.02秒)
%   change_model_frequency(20)   % 设置为20Hz (0.05秒)
%   change_model_frequency(100)  % 设置为100Hz (0.01秒)
%
% 示例:
%   change_model_frequency(20)   % 修改为20Hz

if nargin < 1
    error('请指定目标频率，例如: change_model_frequency(20)');
end

if target_frequency_hz <= 0 || target_frequency_hz > 1000
    error('频率必须在0到1000Hz之间');
end

% 计算固定步长
fixed_step = 1.0 / target_frequency_hz;

fprintf('========================================\n');
fprintf('修改模型频率\n');
fprintf('========================================\n');
fprintf('目标频率: %.2f Hz\n', target_frequency_hz);
fprintf('固定步长: %.4f 秒 (%.2f ms)\n', fixed_step, fixed_step * 1000);
fprintf('========================================\n\n');

% 模型名称
model_name = 'ACC_Decision_SPPVT_Integrated';

% 加载模型
fprintf('1. 加载模型...\n');
load_system(model_name);

% 显示当前配置
fprintf('\n2. 当前配置:\n');
current_step = str2double(get_param(model_name, 'FixedStep'));
current_freq = 1.0 / current_step;
fprintf('   当前固定步长: %.4f 秒\n', current_step);
fprintf('   当前频率: %.2f Hz\n', current_freq);

% 修改配置
fprintf('\n3. 修改配置...\n');

% 设置求解器类型为固定步长
set_param(model_name, 'SolverType', 'Fixed-step');

% 设置固定步长
set_param(model_name, 'FixedStep', num2str(fixed_step, '%.6f'));

% 设置开始和结束时间
set_param(model_name, 'StartTime', '0.0');
set_param(model_name, 'StopTime', 'inf');  % 设置为无限，根据需要调整

fprintf('   ✅ 固定步长已设置为: %.4f 秒\n', fixed_step);
fprintf('   ✅ 采样频率已设置为: %.2f Hz\n', target_frequency_hz);

% 验证修改
fprintf('\n4. 验证修改...\n');
new_step = str2double(get_param(model_name, 'FixedStep'));
new_freq = 1.0 / new_step;
fprintf('   新固定步长: %.4f 秒\n', new_step);
fprintf('   新频率: %.2f Hz\n', new_freq);

if abs(new_freq - target_frequency_hz) < 0.01
    fprintf('   ✅ 频率设置成功！\n');
else
    fprintf('   ⚠️ 频率设置可能有误差\n');
end

% 保存模型
fprintf('\n5. 保存模型...\n');
save_system(model_name);
fprintf('   ✅ 模型已保存\n');

% 关闭模型
close_system(model_name, 0);

fprintf('\n========================================\n');
fprintf('✅ 频率修改完成！\n');
fprintf('========================================\n');
fprintf('新配置:\n');
fprintf('  频率: %.2f Hz\n', target_frequency_hz);
fprintf('  固定步长: %.4f 秒 (%.2f ms)\n', fixed_step, fixed_step * 1000);
fprintf('\n💡 提示:\n');
fprintf('  - 如果Python代码需要匹配此频率，请在acc_updated.py中\n');
fprintf('    将控制周期设置为 %.4f 秒\n', fixed_step);
fprintf('  - 频率越高，控制精度越高，但计算负担也越大\n');
fprintf('  - 常用频率: 20Hz(0.05s), 50Hz(0.02s), 100Hz(0.01s)\n');
fprintf('========================================\n');

end