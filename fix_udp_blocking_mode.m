%% 检查和修复Simulink模型中的UDP配置
% 重点检查UDP Receive的阻塞模式设置

fprintf('======================================================================\n');
fprintf('检查并修复Simulink UDP配置\n');
fprintf('======================================================================\n\n');

models_to_check = {'acc_decision_core', 'sppvt_control_model'};

for m = 1:length(models_to_check)
    model_name = models_to_check{m};

    if ~exist([model_name '.slx'], 'file')
        fprintf('⚠️  模型不存在: %s\n\n', model_name);
        continue;
    end

    fprintf('🔍 检查模型: %s\n', model_name);
    fprintf('======================================================================\n');

    % 加载模型
    load_system(model_name);

    %% 查找UDP Receive模块
    udp_receive_blocks = find_system(model_name, 'BlockType', 'UDPReceive');

    fprintf('\nUDP Receive模块数量: %d\n', length(udp_receive_blocks));

    for i = 1:length(udp_receive_blocks)
        block = udp_receive_blocks{i};
        fprintf('\n【UDP Receive %d】 %s\n', i, block);
        fprintf('----------------------------------------------------------------------\n');

        try
            % 获取配置参数
            local_port = get_param(block, 'LocalIPPort');
            blocking = get_param(block, 'Blocking');  % 'on' 或 'off'
            sample_time = get_param(block, 'SampleTime');

            fprintf('本地端口: %s\n', local_port);
            fprintf('阻塞模式: %s\n', blocking);
            fprintf('采样时间: %s\n', sample_time);

            % 检查是否正确配置
            if strcmp(blocking, 'on')
                fprintf('❌ 错误！UDP Receive设置为阻塞模式\n');
                fprintf('   这会导致Simulink等待数据，可能卡住\n');
                fprintf('   正在修复为非阻塞模式...\n');

                set_param(block, 'Blocking', 'off');
                fprintf('   ✅ 已修改为非阻塞模式\n');
            else
                fprintf('✅ 正确：UDP Receive已设置为非阻塞模式\n');
            end

            % 检查采样时间
            if strcmp(sample_time, '0.05')
                fprintf('✅ 采样时间正确: 0.05s (20 Hz)\n');
            else
                fprintf('⚠️  采样时间: %s (建议0.05s)\n', sample_time);
            end

        catch ME
            fprintf('❌ 获取参数失败: %s\n', ME.message);
        end
    end

    %% 查找UDP Send模块
    udp_send_blocks = find_system(model_name, 'BlockType', 'UDPSend');

    fprintf('\nUDP Send模块数量: %d\n', length(udp_send_blocks));

    for i = 1:length(udp_send_blocks)
        block = udp_send_blocks{i};
        fprintf('\n【UDP Send %d】 %s\n', i, block);
        fprintf('----------------------------------------------------------------------\n');

        try
            remote_addr = get_param(block, 'RemoteIPAddress');
            remote_port = get_param(block, 'RemoteIPPort');
            sample_time = get_param(block, 'SampleTime');

            fprintf('远程地址: %s\n', remote_addr);
            fprintf('远程端口: %s\n', remote_port);
            fprintf('采样时间: %s\n', sample_time);

            % 检查配置
            if strcmp(remote_addr, '127.0.0.1') || strcmp(remote_addr, 'localhost')
                fprintf('✅ 远程地址正确\n');
            else
                fprintf('⚠️  远程地址不是localhost\n');
            end

        catch ME
            fprintf('❌ 获取参数失败: %s\n', ME.message);
        end
    end

    %% 保存模型（如果有修改）
    if ~isempty(udp_receive_blocks)
        fprintf('\n💾 保存模型修改...\n');
        save_system(model_name);
        fprintf('✅ 模型已保存\n');
    end

    close_system(model_name, 0);
    fprintf('\n');
end

fprintf('======================================================================\n');
fprintf('总结\n');
fprintf('======================================================================\n\n');

fprintf('关键配置要求：\n');
fprintf('  1. UDP Receive: Blocking = off (非阻塞) ✅ 最重要！\n');
fprintf('  2. UDP Receive: Sample Time = 0.05 (20 Hz)\n');
fprintf('  3. UDP Send: Sample Time = 0.05 (20 Hz)\n');
fprintf('  4. 模型: StopTime = inf, Solver = FixedStepDiscrete\n\n');

fprintf('为什么非阻塞模式重要：\n');
fprintf('  - Simulink持续运行，不会等待Python发送数据\n');
fprintf('  - Python按需发送UDP请求，Simulink立即响应\n');
fprintf('  - 避免死锁和超时问题\n\n');

fprintf('下一步：\n');
fprintf('  1. 重新启动Simulink模型（关闭并重新打开）\n');
fprintf('  2. 运行测试: python test_e_key_auto.py\n\n');