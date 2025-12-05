function check_udp_send_config()
%CHECK_UDP_SEND_CONFIG 检查两个模型的UDP Send模块配置
%   查看是否有端口冲突（例如绑定了相同的本地端口）

models = {'acc_decision_core', 'sppvt_control_model'};

fprintf('======================================================================\n');
fprintf('检查 UDP Send 模块配置\n');
fprintf('======================================================================\n\n');

for i = 1:length(models)
    model_name = models{i};
    fprintf('🔍 模型: %s\n', model_name);

    if ~exist([model_name '.slx'], 'file')
        fprintf('   ❌ 文件不存在\n');
        continue;
    end

    try
        load_system(model_name);

        % 查找 UDP Send 模块
        udp_sends = find_system(model_name, 'BlockType', 'UDPSend');

        if isempty(udp_sends)
            fprintf('   ⚠️  未找到 UDP Send 模块\n');
        else
            for j = 1:length(udp_sends)
                blk = udp_sends{j};
                fprintf('   模块: %s\n', blk);

                % 获取关键参数
                remote_host = get_param(blk, 'RemoteIPAddress');
                remote_port = get_param(blk, 'RemoteIPPort');
                local_port = get_param(blk, 'LocalIPPort'); % 发送方的本地端口

                fprintf('      目标: %s:%s\n', remote_host, remote_port);
                fprintf('      本地绑定端口 (LocalIPPort): %s\n', local_port);

                if ~strcmp(local_port, '-1')
                    fprintf('      ⚠️  注意: 绑定了固定本地端口! 如果两个模型都用这个端口，会冲突。\n');
                else
                    fprintf('      ✅ 自动分配本地端口 (-1)\n');
                end
            end
        end

        % 同时也检查 UDP Receive
        udp_recvs = find_system(model_name, 'BlockType', 'UDPReceive');
        for j = 1:length(udp_recvs)
            blk = udp_recvs{j};
            local_port = get_param(blk, 'LocalIPPort');
            fprintf('   [Receive] 监听端口: %s\n', local_port);
        end

    catch ME
        fprintf('   ❌ 检查失败: %s\n', ME.message);
    end
    fprintf('\n');
end

end
