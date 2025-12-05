function embed_data_to_model()
%EMBED_DATA_TO_MODEL 将查找表数据固化到Simulink模型工作区
%   这样模型就可以独立运行，不再依赖外部的decision_lookup_data.mat文件
%   也不会因为Base Workspace变量丢失而报错

fprintf('🔨 开始将数据固化到模型工作区...\n');

model_name = 'acc_decision_core';
data_file = 'decision_lookup_data.mat';

%% 1. 检查文件
if ~exist([model_name '.slx'], 'file')
    error('❌ 未找到模型文件: %s.slx', model_name);
end

if ~exist(data_file, 'file')
    error('❌ 未找到数据文件: %s', data_file);
end

%% 2. 加载数据
fprintf('📂 加载数据文件: %s\n', data_file);
data = load(data_file);

%% 3. 加载模型
fprintf('📂 加载模型: %s\n', model_name);
load_system(model_name);

%% 4. 获取模型工作区并写入数据
fprintf('💾 写入数据到 Model Workspace...\n');
hws = get_param(model_name, 'ModelWorkspace');

% 设置数据源为模型文件（这样数据会保存在slx内部）
hws.DataSource = 'Model File';

% 写入变量
vars = fieldnames(data);
for i = 1:length(vars)
    var_name = vars{i};
    var_value = data.(var_name);
    
    hws.assignin(var_name, var_value);
    fprintf('   ✓ 已添加变量: %s\n', var_name);
end

%% 5. 保存模型
fprintf('💾 保存模型...\n');
save_system(model_name);
close_system(model_name);

fprintf('\n✅ 操作完成！\n');
fprintf('   现在 %s.slx 已经包含了所有查找表数据。\n', model_name);
fprintf('   你不再需要手动 load(''%s'') 了。\n', data_file);

end
