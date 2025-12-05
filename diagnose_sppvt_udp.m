%% 诊断SPPVT模型UDP配置
% 详细检查UDP Receive和UDP Send模块的配置

fprintf('=== 诊断SPPVT模型UDP配置 ===\n\n');

model = 'sppvt_control_model';

%% 1. 检查模型是否加载和运行
fprintf('1️⃣  检查模型状态\n');
fprintf('--------------------------------------------------\n');

if ~bdIsLoaded(model)
    fprintf('⚠️  模型未加载，正在加载...\n');
    load_system(model);
end

sim_status = get_param(model, 'SimulationStatus');
fprintf('仿真状态: %s\n', sim_status);

if ~strcmp(sim_status, 'running')
    fprintf('❌ 模型没有运行！这就是问题所在！\n');
    fprintf('   解决方法: set_param(''%s'', ''SimulationCommand'', ''start'');\n', model);
    should_start = true;
else
    fprintf('✅ 模型正在运行\n');
    should_start = false;
end

fprintf('\n');

%% 2. 查找UDP模块
fprintf('2️⃣  查找UDP Receive和UDP Send模块\n');
fprintf('--------------------------------------------------\n');

% 查找所有UDP Receive模块
udp_receive_blocks = find_system(model, 'BlockType', 'UDPReceive');
fprintf('找到 %d 个UDP Receive模块:\n', length(udp_receive_blocks));
for i = 1:length(udp_receive_blocks)
    fprintf('  %d. %s\n', i, udp_receive_blocks{i});
end

% 查找所有UDP Send模块
udp_send_blocks = find_system(model, 'BlockType', 'UDPSend');
fprintf('找到 %d 个UDP Send模块:\n', length(udp_send_blocks));
for i = 1:length(udp_send_blocks)
    fprintf('  %d. %s\n', i, udp_send_blocks{i});
end

if isempty(udp_receive_blocks) || isempty(udp_send_blocks)
    fprintf('❌ 错误: 找不到UDP模块！\n');
    fprintf('   模型中应该有UDP Receive和UDP Send模块\n');
    return;
end

fprintf('\n');

%% 3. 检查UDP Receive配置
fprintf('3️⃣  检查UDP Receive配置（Python发送到此端口）\n');
fprintf('--------------------------------------------------\n');

for i = 1:length(udp_receive_blocks)
    block = udp_receive_blocks{i};
    fprintf('模块: %s\n', block);

    try
        % 获取关键参数
        local_port = get_param(block, 'LocalIPPort');
        timeout = get_param(block, 'Timeout');

        fprintf('  LocalIPPort (接收端口): %s\n', local_port);
        fprintf('  Timeout: %s 秒\n', timeout);

        % 检查端口是否正确
        if strcmp(local_port, '26000')
            fprintf('  ✅ 端口配置正确\n');
        else
            fprintf('  ❌ 端口错误！应该是 26000，当前是 %s\n', local_port);
            fprintf('     修复: set_param(''%s'', ''LocalIPPort'', ''26000'');\n', block);
        end

        % 检查超时设置
        timeout_val = str2double(timeout);
        if timeout_val < 5
            fprintf('  ⚠️  超时设置较短: %s秒\n', timeout);
            fprintf('     建议增加: set_param(''%s'', ''Timeout'', ''10'');\n', block);
        end

    catch e
        fprintf('  ❌ 无法读取参数: %s\n', e.message);
    end
end

fprintf('\n');

%% 4. 检查UDP Send配置
fprintf('4️⃣  检查UDP Send配置（发送到Python端口）\n');
fprintf('--------------------------------------------------\n');

for i = 1:length(udp_send_blocks)
    block = udp_send_blocks{i};
    fprintf('模块: %s\n', block);

    try
        % 获取关键参数
        remote_addr = get_param(block, 'RemoteIPAddress');
        remote_port = get_param(block, 'RemoteIPPort');

        fprintf('  RemoteIPAddress: %s\n', remote_addr);
        fprintf('  RemoteIPPort (发送端口): %s\n', remote_port);

        % 检查配置是否正确
        if strcmp(remote_addr, '127.0.0.1') || strcmp(remote_addr, 'localhost')
            fprintf('  ✅ IP地址正确\n');
        else
            fprintf('  ❌ IP地址错误！应该是 127.0.0.1，当前是 %s\n', remote_addr);
        end

        if strcmp(remote_port, '26001')
            fprintf('  ✅ 端口配置正确\n');
        else
            fprintf('  ❌ 端口错误！应该是 26001，当前是 %s\n', remote_port);
            fprintf('     修复: set_param(''%s'', ''RemoteIPPort'', ''26001'');\n', block);
        end

    catch e
        fprintf('  ❌ 无法读取参数: %s\n', e.message);
    end
end

fprintf('\n');

%% 5. 检查数据类型配置
fprintf('5️⃣  检查UDP数据类型配置\n');
fprintf('--------------------------------------------------\n');

for i = 1:length(udp_receive_blocks)
    block = udp_receive_blocks{i};
    try
        data_size = get_param(block, 'DataSize');
        data_type = get_param(block, 'DataType');

        fprintf('UDP Receive: DataSize=%s, DataType=%s\n', data_size, data_type);

        % SPPVT应该接收5个double (5*8=40 bytes)
        if strcmp(data_size, '5') && strcmp(data_type, 'double')
            fprintf('  ✅ 接收配置正确 (5 doubles = 40 bytes)\n');
        else
            fprintf('  ⚠️  配置可能不对，期望: DataSize=5, DataType=double\n');
        end
    catch
        fprintf('  无法读取数据类型配置\n');
    end
end

for i = 1:length(udp_send_blocks)
    block = udp_send_blocks{i};
    try
        % UDP Send从输入推断，检查输入端口
        fprintf('UDP Send: 检查输入信号维度...\n');
        % 应该发送5个double
    catch
        fprintf('  无法读取配置\n');
    end
end

fprintf('\n');

%% 6. 总结和建议
fprintf('=== 诊断总结 ===\n');

if should_start
    fprintf('❌ 主要问题: 模型没有运行\n');
    fprintf('\n立即修复:\n');
    fprintf('  set_param(''%s'', ''SimulationCommand'', ''start'');\n', model);
    fprintf('  然后重新运行 Python 测试\n');
else
    fprintf('模型正在运行，如果还有问题，请检查:\n');
    fprintf('  1. UDP Receive端口是否是26000\n');
    fprintf('  2. UDP Send端口是否是26001\n');
    fprintf('  3. 防火墙是否阻止了本地UDP通信\n');
    fprintf('  4. 是否有其他程序占用了26000或26001端口\n');
    fprintf('\n测试端口占用:\n');
    if ispc
        fprintf('  netstat -ano | findstr "26000"\n');
        fprintf('  netstat -ano | findstr "26001"\n');
    else
        fprintf('  lsof -i :26000\n');
        fprintf('  lsof -i :26001\n');
    end
end

%% 7. 提供启动命令
fprintf('\n=== 完整启动流程 ===\n');
fprintf('1. 确保配置正确后，停止模型:\n');
fprintf('   set_param(''%s'', ''SimulationCommand'', ''stop'');\n', model);
fprintf('\n2. 重新启动模型:\n');
fprintf('   set_param(''%s'', ''SimulationCommand'', ''start'');\n', model);
fprintf('\n3. 检查模型状态:\n');
fprintf('   get_param(''%s'', ''SimulationStatus'')\n', model);
fprintf('   应该显示: running\n');
fprintf('\n4. 运行Python测试:\n');
fprintf('   python test_udp_stress.py\n');
