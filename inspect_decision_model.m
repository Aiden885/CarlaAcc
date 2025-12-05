%% 检查acc_decision_core模型的内部结构
fprintf('======================================================================\n');
fprintf('检查ACC决策模型内部结构\n');
fprintf('======================================================================\n\n');

model_name = 'acc_decision_core';

%% 加载模型
fprintf('📂 加载模型: %s\n', model_name);
load_system(model_name);
fprintf('✅ 模型加载成功\n\n');

%% 查找所有模块
fprintf('🔍 查找模型中的所有模块:\n');
fprintf('----------------------------------------------------------------------\n');

blocks = find_system(model_name, 'Type', 'Block');
fprintf('模型包含 %d 个模块\n\n', length(blocks));

%% 查找关键模块类型
fprintf('🔍 关键模块类型:\n');
fprintf('----------------------------------------------------------------------\n');

% 查找Lookup Table
lookup_blocks = find_system(model_name, 'BlockType', 'Lookup_n-D');
fprintf('Lookup Table 模块数量: %d\n', length(lookup_blocks));
if ~isempty(lookup_blocks)
    for i = 1:length(lookup_blocks)
        fprintf('  [%d] %s\n', i, lookup_blocks{i});
    end
end

% 查找Stateflow Chart
chart_blocks = find_system(model_name, 'MaskType', 'Stateflow');
fprintf('\nStateflow Chart 模块数量: %d\n', length(chart_blocks));
if ~isempty(chart_blocks)
    for i = 1:length(chart_blocks)
        fprintf('  [%d] %s\n', i, chart_blocks{i});
    end
end

% 查找MATLAB Function
matlab_fcn_blocks = find_system(model_name, 'BlockType', 'MATLABFcn');
fprintf('\nMATLAB Function 模块数量: %d\n', length(matlab_fcn_blocks));
if ~isempty(matlab_fcn_blocks)
    for i = 1:length(matlab_fcn_blocks)
        fprintf('  [%d] %s\n', i, matlab_fcn_blocks{i});
    end
end

% 查找Subsystem
subsystem_blocks = find_system(model_name, 'BlockType', 'SubSystem');
fprintf('\nSubsystem 模块数量: %d\n', length(subsystem_blocks));
if ~isempty(subsystem_blocks)
    for i = 1:min(5, length(subsystem_blocks))  % 只显示前5个
        fprintf('  [%d] %s\n', i, subsystem_blocks{i});
    end
    if length(subsystem_blocks) > 5
        fprintf('  ... 还有 %d 个\n', length(subsystem_blocks) - 5);
    end
end

%% 检查UDP Receive和Send模块
fprintf('\n🔍 UDP通信模块:\n');
fprintf('----------------------------------------------------------------------\n');

udp_receive = find_system(model_name, 'BlockType', 'UDPReceive');
udp_send = find_system(model_name, 'BlockType', 'UDPSend');

fprintf('UDP Receive 模块数量: %d\n', length(udp_receive));
if ~isempty(udp_receive)
    for i = 1:length(udp_receive)
        fprintf('  [%d] %s\n', i, udp_receive{i});
        % 获取端口配置
        try
            port = get_param(udp_receive{i}, 'LocalIPPort');
            fprintf('      端口: %s\n', port);
        catch
        end
    end
end

fprintf('\nUDP Send 模块数量: %d\n', length(udp_send));
if ~isempty(udp_send)
    for i = 1:length(udp_send)
        fprintf('  [%d] %s\n', i, udp_send{i});
        % 获取端口配置
        try
            port = get_param(udp_send{i}, 'RemoteIPPort');
            fprintf('      端口: %s\n', port);
        catch
        end
    end
end

%% 检查输入输出端口
fprintf('\n🔍 模型输入输出端口:\n');
fprintf('----------------------------------------------------------------------\n');

inports = find_system(model_name, 'SearchDepth', 1, 'BlockType', 'Inport');
outports = find_system(model_name, 'SearchDepth', 1, 'BlockType', 'Outport');

fprintf('输入端口数量: %d\n', length(inports));
for i = 1:length(inports)
    fprintf('  In%d: %s\n', i, get_param(inports{i}, 'Name'));
end

fprintf('\n输出端口数量: %d\n', length(outports));
for i = 1:length(outports)
    fprintf('  Out%d: %s\n', i, get_param(outports{i}, 'Name'));
end

%% 总结
fprintf('\n======================================================================\n');
fprintf('诊断建议\n');
fprintf('======================================================================\n\n');

if isempty(lookup_blocks)
    fprintf('⚠️  警告：模型中没有Lookup Table模块！\n');
    fprintf('   这说明模型可能使用了Stateflow或MATLAB Function实现决策逻辑\n');
    fprintf('   但没有正确配置为使用decision_lookup_data.mat中的数据\n\n');
else
    fprintf('✅ 模型包含Lookup Table模块\n');
    fprintf('   需要检查这些模块是否正确引用了工作区变量\n\n');
end

if ~isempty(chart_blocks)
    fprintf('⚠️  注意：模型包含Stateflow Chart\n');
    fprintf('   Stateflow可能覆盖了查找表逻辑\n\n');
end

close_system(model_name, 0);
fprintf('✅ 检查完成\n');
